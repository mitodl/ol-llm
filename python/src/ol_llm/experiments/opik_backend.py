"""Opik as the experiment backend, and Opik's judge metrics as ol-llm metrics.

Datasets and experiments live in the Opik that tracing is configured for, an app's
namespace is its Opik project, and ``opik.evaluate`` runs each experiment, so the
experiment links the prompt versions under test and each item has a trace. ol-llm
metrics run as Opik ``BaseMetric`` objects.

The judge metrics (``hallucination`` and the others below) are Opik's metric
classes, which run in this process and need only a model: the judge, wrapped as an
``OpikBaseModel``. Given one, they do not import litellm. They keep Opik's score
names, e.g. ``hallucination_metric``, whatever backend runs the experiment.

Run experiments where the Opik SDK is installed whole, not with the slimming
overrides: when a task or metric raises anything else, ``opik.evaluate`` checks
whether it was a rate limit by importing ``openai`` and ``litellm``
(``opik/evaluation/engine/exception_analyzer.py``), so without them that failure
aborts the run with ``ModuleNotFoundError``.
"""

import functools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import httpx
import opik
import pydantic
from opik import context_storage, url_helpers
from opik.api_objects.dataset.dataset_item import DatasetItem
from opik.evaluation.evaluation_result import EvaluationResult
from opik.evaluation.metrics import (
    AnswerRelevance,
    BaseMetric,
    ContextPrecision,
    ContextRecall,
    Hallucination,
    arguments_helpers,
    score_result,
)
from opik.evaluation.metrics.arguments_validator import validate_score_arguments
from opik.evaluation.models import OpikBaseModel
from opik.evaluation.models.base_model import ConversationDict
from opik.exceptions import DatasetVersionNotFound
from opik.rest_api.core import ApiError
from opik.types import BatchFeedbackScoreDict

from ol_llm import tracing
from ol_llm.experiments.interface import (
    MAX_EXPERIMENT_ITEMS,
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
    ExperimentBackend,
    ExperimentItem,
    ExperimentResult,
    ExperimentRun,
    ExperimentSummary,
    Metric,
    Score,
    Task,
    TraceScore,
)
from ol_llm.judge import JudgeModel
from ol_llm.prompt_registry.opik_registry import opik_prompt, opik_version_id
from ol_llm.prompts import ManagedPrompt
from ol_llm.runs import LLMResult
from ol_llm.settings import get_settings

_ITEM_FIELDS = frozenset(
    {
        "id",
        "trace_id",
        "span_id",
        "source",
        "description",
        "evaluators",
        "execution_policy",
    }
)
"""The keys ``opik.api_objects.dataset.dataset_item.DatasetItem`` takes as its
own fields rather than as the item's data."""

_TRACE_TOOL_CONTEXT = "trace_tool_context"
"""What ``opik.evaluate`` adds to the inputs of a metric that takes ``**kwargs``,
which an ol-llm metric does not score on."""

_PAGE_SIZE = 100


class OpikExperimentBackend(ExperimentBackend):
    """Datasets and experiments in the Opik that tracing is configured for."""

    def disabled_reason(self) -> str | None:
        """Say that tracing is off, since then there is no Opik to run against."""
        if tracing.enabled():
            return None
        return "Tracing is off (OPIK_TRACK_DISABLE), so there is no Opik to run against"

    def seed_dataset(
        self,
        name: str,
        items: Sequence[Mapping[str, Any]],
        *,
        app: str,
        description: str | None = None,
    ) -> str | None:
        """Insert the items into the dataset in the app's Opik project.

        The SDK hashes each item's content and skips one the dataset already
        holds, reading the existing items' hashes before the first insert into a
        dataset it fetched, and sends nothing when every item is already there.
        Each insert that adds items makes a new dataset version. Items removed
        from the file stay in the dataset: it is curated in Opik (spec §5.1 item
        6), and deleting there is a deliberate change.

        :raises ValueError: if an item has a field Opik keeps for itself, e.g.
            ``id`` or ``description``.
        """
        rows = [dict(item) for item in items]
        if reserved := sorted({key for row in rows for key in row} & _ITEM_FIELDS):
            msg = (
                f"Golden items may not have the fields {reserved}: Opik reads them "
                "as the dataset item's own (an id must be a UUID, a description is "
                "dropped from what the task sees). Rename them."
            )
            raise ValueError(msg)
        dataset = tracing.opik_client().get_or_create_dataset(
            name, description=description, project_name=app
        )
        dataset.insert(rows)
        return dataset.get_current_version_name()

    def latest_dataset_version(self, name: str, *, app: str) -> str | None:
        """Return the dataset's current version name."""
        return _dataset(name, app).get_current_version_name()

    def missing_items(
        self,
        name: str,
        items: Sequence[Mapping[str, Any]],
        *,
        app: str,
        version: str,
    ) -> int:
        """Count the items the version lacks, by the SDK's content hash."""
        dataset = _dataset(name, app)
        try:
            pinned = dataset.get_version_view(version)
        except DatasetVersionNotFound as exc:
            msg = f"Dataset {name!r} has no version {version}"
            raise DatasetVersionNotFoundError(
                msg, latest=dataset.get_current_version_name()
            ) from exc
        if not items:
            return 0
        held = {DatasetItem(**item).content_hash() for item in pinned.get_items()}
        return len({DatasetItem(**item).content_hash() for item in items} - held)

    def linked_version(self, prompt: ManagedPrompt[Any]) -> str | None:
        """Return the Opik prompt version id, which experiments link by."""
        return opik_version_id(prompt.registry_ref)

    def run_experiment(self, run: ExperimentRun) -> ExperimentResult:
        """Run ``opik.evaluate`` with the managed prompts' Opik versions linked."""
        source = _dataset(run.dataset, run.app)
        result = opik.evaluate(
            dataset=source
            if run.dataset_version is None
            else source.get_version_view(run.dataset_version),
            task=_in_environment(run.task),
            scoring_metrics=[_OpikMetric(metric) for metric in run.metrics],
            experiment_name=run.name,
            experiment_config=dict(run.config),
            prompts=[
                prompt
                for p in run.prompts
                if (prompt := opik_prompt(p.registry_ref)) is not None
            ],
            scoring_key_mapping=dict(run.scoring_key_mapping),
            experiment_tags=list(run.tags),
            task_threads=run.threads,
        )
        return _result(result)

    def read_experiment(self, experiment_id: str) -> ExperimentResult:
        """Read the experiment's items; Opik keeps only the scores that succeeded."""
        experiment = tracing.opik_client().get_experiment_by_id(experiment_id)
        # One past the SDK's cap, which max_results=None does not lift.
        found = experiment.get_items(max_results=MAX_EXPERIMENT_ITEMS + 1)
        if len(found) > MAX_EXPERIMENT_ITEMS:
            msg = (
                f"Experiment {experiment_id} has more than {MAX_EXPERIMENT_ITEMS} "
                "items, more than the Opik SDK reads"
            )
            raise ValueError(msg)
        return ExperimentResult(
            experiment_id=experiment_id,
            name=experiment.name,
            url=None,
            items=tuple(
                ExperimentItem(
                    dataset_item_id=item.dataset_item_id,
                    trace_id=item.trace_id,
                    item=item.dataset_item_data or {},
                    output=item.evaluation_task_output or {},
                    # Opik sends a null value for a score it has no value for.
                    scores=tuple(
                        Score(name=s["name"], value=0.0, failed=True)
                        if s["value"] is None
                        else Score(name=s["name"], value=s["value"])
                        for s in item.feedback_scores
                    ),
                )
                for item in found
            ),
        )

    def experiment_dataset_version(self, experiment_id: str) -> str | None:
        """Return the dataset and dataset version ids the experiment records."""
        experiment = tracing.opik_client().rest_client.experiments.get_experiment_by_id(
            id=experiment_id
        )
        if experiment.dataset_version_id is None:
            return None
        return f"{experiment.dataset_id}/{experiment.dataset_version_id}"

    def log_scores(self, scores: Sequence[TraceScore], *, app: str) -> None:
        """Log the scores on the traces as feedback scores."""
        tracing.opik_client().log_traces_feedback_scores(
            scores=[
                BatchFeedbackScoreDict(
                    id=score.trace_id,
                    name=score.name,
                    value=score.value,
                    reason=score.reason,
                    category_name=score.category,
                    project_name=app,
                )
                for score in scores
            ]
        )

    def tag_experiment(self, experiment_id: str, tag: str) -> None:
        """Add the tag through Opik's REST API."""
        tracing.opik_client().rest_client.experiments.update_experiment(
            experiment_id, tags_to_add=[tag]
        )

    def find_experiments(
        self, *, app: str, dataset: str, tags: Sequence[str]
    ) -> Iterator[ExperimentSummary]:
        """Page through the dataset's experiments that carry every tag.

        Opik filters on each tag (``=`` on a list is membership) and returns
        experiments by id, descending, when no sorting is given; ids are UUIDv7,
        so that is newest first. Checked against the Opik backend at 2.2.81 and
        2.2.103 (``ExperimentDAO.FIND``, ``FilterQueryBuilder``).
        """
        dataset_id = _dataset(dataset, app).id
        filters = json.dumps(
            [{"field": "tags", "operator": "=", "value": tag} for tag in tags]
        )
        rest = tracing.opik_client().rest_client
        page = 1
        while True:
            content = (
                rest.experiments.find_experiments(
                    page=page, size=_PAGE_SIZE, dataset_id=dataset_id, filters=filters
                ).content
                or []
            )
            for experiment in content:
                # Checked here too, so a backend that ignored the filter cannot
                # hand back a pull request's run as a baseline.
                if experiment.id is None or not set(tags) <= set(experiment.tags or []):
                    continue
                summary = experiment.dataset_version_summary
                yield ExperimentSummary(
                    experiment_id=experiment.id,
                    name=experiment.name,
                    url=_experiment_url(dataset_id, experiment.id),
                    tags=frozenset(experiment.tags or []),
                    prompt_versions=frozenset(
                        v.prompt_version_id
                        for v in experiment.prompt_versions or []
                        if v.prompt_version_id is not None
                    ),
                    dataset_version=None if summary is None else summary.version_name,
                    means={s.name: s.value for s in experiment.feedback_scores or []},
                )
            if len(content) < _PAGE_SIZE:
                return
            page += 1


def _dataset(name: str, app: str) -> opik.Dataset:
    try:
        return tracing.opik_client().get_dataset(name, project_name=app)
    except ApiError as exc:
        if exc.status_code == httpx.codes.NOT_FOUND:
            msg = f"App {app!r} has no dataset {name!r}"
            raise DatasetNotFoundError(msg) from exc
        raise


def _experiment_url(dataset_id: str, experiment_id: str) -> str | None:
    client = tracing.opik_client()
    base_url = client.config.url_override
    if not base_url:
        return None
    return url_helpers.get_experiment_url_by_id(
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        base_url=base_url,
        workspace=client.config.workspace,
    )


def _carry_environment() -> None:
    """Put the deployment environment on the item's trace and its open span.

    ``opik.evaluate`` opens each item's trace with no environment (``engine.py``),
    and Opik gives a nested span its parent's environment, so without this the
    task's and the judges' calls land in Opik's default environment instead of
    ``Settings.environment``, with a warning for each.
    ``opik_context`` hands out copies; these are the objects Opik sends when the
    span and trace close. The exact opik pin keeps ``context_storage`` stable.
    """
    environment = get_settings().environment
    if (trace := context_storage.get_trace_data()) is not None:
        trace.environment = environment
    if (span := context_storage.top_span_data()) is not None:
        span.environment = environment


def _in_environment(task: Task) -> Task:
    @functools.wraps(task)
    def run(item: dict[str, Any]) -> dict[str, Any]:
        _carry_environment()
        return task(item)

    return run


class _OpikMetric(BaseMetric):
    """An ol-llm metric as an Opik metric, so ``opik.evaluate`` scores with it.

    :param metric: the metric; its ``track`` decides whether Opik records its
        scoring as a span.
    """

    def __init__(self, metric: Metric) -> None:
        """Take the metric's name and tracking."""
        super().__init__(name=metric.name, track=metric.track)
        self.metric = metric

    def score(
        self,
        **inputs: Any,  # noqa: ANN401
    ) -> score_result.ScoreResult | list[score_result.ScoreResult]:
        """Score the item with the ol-llm metric."""
        _carry_environment()
        inputs.pop(_TRACE_TOOL_CONTEXT, None)
        found = self.metric.score(**inputs)
        if isinstance(found, Score):
            return _to_opik(found)
        return [_to_opik(score) for score in found]


def _to_opik(score: Score) -> score_result.ScoreResult:
    return score_result.ScoreResult(
        name=score.name,
        value=score.value,
        reason=score.reason,
        scoring_failed=score.failed,
    )


def _from_opik(result: score_result.ScoreResult) -> Score:
    return Score(
        name=result.name,
        value=result.value,
        reason=result.reason,
        failed=result.scoring_failed,
    )


def _result(result: EvaluationResult) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=result.experiment_id,
        name=result.experiment_name,
        url=result.experiment_url,
        items=tuple(
            ExperimentItem(
                dataset_item_id=test.test_case.dataset_item_id,
                trace_id=test.test_case.trace_id,
                item=test.test_case.dataset_item_content or {},
                output=test.test_case.task_output or {},
                scores=tuple(_from_opik(s) for s in test.score_results),
            )
            for test in result.test_results
        ),
    )


class _OpikJudge(OpikBaseModel):
    """A ``JudgeModel`` as an ``OpikBaseModel``, the model Opik's judge metrics take.

    :param judge: the judge.
    """

    def __init__(self, judge: JudgeModel) -> None:
        """Take the judge's model id as the model name."""
        super().__init__(model_name=judge.model_name)
        self.judge = judge

    def generate_string(
        self,
        input: str,  # noqa: A002 - the base class's parameter name
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Answer one user message through the judge."""
        return self.judge.generate_string(input, response_format, **kwargs)

    async def agenerate_string(
        self,
        input: str,  # noqa: A002 - the base class's parameter name
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Answer one user message through the judge."""
        return await self.judge.agenerate_string(input, response_format, **kwargs)

    def generate_chat_completion(
        self,
        messages: list[ConversationDict],
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ConversationDict:
        """Answer a conversation through the judge."""
        reply = self.judge.generate_chat_completion(messages, response_format, **kwargs)
        return {"role": "assistant", "content": reply["content"]}

    async def agenerate_chat_completion(
        self,
        messages: list[ConversationDict],
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ConversationDict:
        """Answer a conversation through the judge."""
        reply = await self.judge.agenerate_chat_completion(
            messages, response_format, **kwargs
        )
        return {"role": "assistant", "content": reply["content"]}

    def generate_provider_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,  # noqa: ANN401
    ) -> LLMResult[str]:
        """Answer a conversation through the judge and return the whole result."""
        return self.judge.respond(messages, **kwargs)

    async def agenerate_provider_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,  # noqa: ANN401
    ) -> LLMResult[str]:
        """Answer a conversation through the judge and return the whole result."""
        return await self.judge.arespond(messages, **kwargs)


class OpikJudgeMetric(Metric):
    """One of Opik's judge metrics as an ol-llm metric.

    Opik's metric runs in this process with the judge as its model. A missing
    input raises Opik's ``ScoreMethodMissingArguments``, which aborts an Opik
    experiment rather than failing each item's score.

    :param metric: the Opik metric, built untracked: the backend that runs the
        experiment traces the ol-llm metric instead.
    :param track: see ``Metric``.
    """

    def __init__(self, metric: BaseMetric, *, track: bool = True) -> None:
        """Take the Opik metric's name."""
        super().__init__(metric.name, track=track)
        self.metric = metric

    def score(self, **inputs: Any) -> Score | list[Score]:  # noqa: ANN401
        """Score the item with the Opik metric.

        :raises ScoreMethodMissingArguments: if an input it needs is missing.
        """
        validate_score_arguments(self.metric, inputs, None)
        positional, keyword = arguments_helpers.select_score_arguments(
            self.metric.score, inputs, self.name
        )
        found = self.metric.score(*positional, **keyword)
        if isinstance(found, list):
            return [_from_opik(result) for result in found]
        return _from_opik(found)


def _judge_metric(
    build: Callable[..., BaseMetric], judge: JudgeModel, *, track: bool
) -> OpikJudgeMetric:
    return OpikJudgeMetric(build(model=_OpikJudge(judge), track=False), track=track)


def hallucination(judge: JudgeModel, *, track: bool = True) -> OpikJudgeMetric:
    """Return Opik's ``Hallucination``, scored ``hallucination_metric``.

    1.0 when the output makes claims the context does not support, so lower is
    better. Needs ``input``, ``output`` and ``context``; without a context it
    judges the output against general knowledge.

    :param judge: the judge, from ``judge_model``.
    :param track: whether scoring is a span of the item's trace.
    :returns: the metric.
    :rtype: OpikJudgeMetric
    """
    return _judge_metric(Hallucination, judge, track=track)


def answer_relevance(judge: JudgeModel, *, track: bool = True) -> OpikJudgeMetric:
    """Return Opik's ``AnswerRelevance``, scored ``answer_relevance_metric``.

    Higher is better. Needs ``input``, ``output`` and ``context``.

    :param judge: the judge, from ``judge_model``.
    :param track: whether scoring is a span of the item's trace.
    :returns: the metric.
    :rtype: OpikJudgeMetric
    """
    return _judge_metric(AnswerRelevance, judge, track=track)


def context_precision(judge: JudgeModel, *, track: bool = True) -> OpikJudgeMetric:
    """Return Opik's ``ContextPrecision``, scored ``context_precision_metric``.

    Higher is better. Needs ``input``, ``output``, ``expected_output`` and
    ``context``.

    :param judge: the judge, from ``judge_model``.
    :param track: whether scoring is a span of the item's trace.
    :returns: the metric.
    :rtype: OpikJudgeMetric
    """
    return _judge_metric(ContextPrecision, judge, track=track)


def context_recall(judge: JudgeModel, *, track: bool = True) -> OpikJudgeMetric:
    """Return Opik's ``ContextRecall``, scored ``context_recall_metric``.

    Higher is better. Needs ``input``, ``output``, ``expected_output`` and
    ``context``.

    :param judge: the judge, from ``judge_model``.
    :param track: whether scoring is a span of the item's trace.
    :returns: the metric.
    :rtype: OpikJudgeMetric
    """
    return _judge_metric(ContextRecall, judge, track=track)
