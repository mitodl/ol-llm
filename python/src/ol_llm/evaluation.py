"""Evaluation in Opik (lifecycle spec §4.2 and §5): datasets, experiments, gates.

A use case whose output a learner or instructor sees keeps a golden dataset in Opik
and runs an experiment on every prompt or model change. This module is the path
for that:

- ``load_items`` and ``seed_dataset`` put a golden set from a local file into an
  Opik dataset. Seeding again is a no-op for items already there, since the SDK
  skips an item whose content hash the dataset already holds, so a seed job can
  run on every change to the file. Each insert that adds items makes a new
  dataset version, which a gate pins.
- ``evaluate`` runs ``opik.evaluate`` with the managed prompts the task uses, so
  the experiment links their versions. An ``LLMError`` from the task or a judge
  fails that item's scores instead of aborting the run.
- ``ValidatorMetric`` turns a deterministic check (an SRT structure check, a
  placeholder check) into an Opik metric, and ``judge_model`` gives Opik's
  LLM-judge metrics a model that calls through this library, so the judge is
  catalog-checked, routed, retried, and traced like any other call, under the
  ``eval-judge`` use case. Opik's judge metrics that take a ``model`` accept any
  ``OpikBaseModel``, and given one they do not import litellm.
- ``check_thresholds`` compares an experiment's scores to the declared
  thresholds. Gates are advisory until a use case's owner makes them blocking
  (spec §5.1 item 4), so by default it reports and does not raise.

``ol-llm-eval seed`` and ``ol-llm-eval check`` do the seeding and the threshold
check from a CI job. Needs the ``pydantic-ai`` extra, and the provider extra for
the judge's model.

Run ``evaluate`` where the Opik SDK is installed whole, not with the slimming
overrides: when a task or metric raises anything else, ``opik.evaluate`` checks
whether it was a rate limit by importing ``openai`` and ``litellm``
(``opik/evaluation/engine/exception_analyzer.py``), so without them that failure
aborts the run with ``ModuleNotFoundError``.
"""

import argparse
import functools
import inspect
import json
import logging
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import opik
import pydantic
from opik.evaluation.evaluation_result import EvaluationResult
from opik.evaluation.metrics import BaseMetric, arguments_helpers, score_result
from opik.evaluation.metrics.arguments_validator import validate_score_arguments
from opik.evaluation.models import OpikBaseModel
from opik.evaluation.models.base_model import ConversationDict
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from ol_llm import tracing
from ol_llm.catalog import load_catalog
from ol_llm.errors import LLMError
from ol_llm.policy import UseCase
from ol_llm.prompts import ManagedPrompt
from ol_llm.runs import LLMResult, complete, complete_sync
from ol_llm.settings import get_settings

log = logging.getLogger(__name__)

JUDGE_USE_CASE = "eval-judge"
"""The use case every judge call is attributed to (spec §4.2)."""

GateMode = Literal["advisory", "blocking"]
"""Advisory reports and does not fail; blocking fails the check (spec §5.1)."""

LLM_ERROR_FIELD = "ol_llm_error"
"""The output field ``evaluate`` records, as the error's class name, for an item
whose task raised an ``LLMError``. It shows on the item's trace in Opik."""

Verdict = bool | Sequence[str]
"""What a validator returns: whether the output passed, or the problems it found,
where none means it passed."""


def judge_model(app: str, model: str) -> "JudgeModel":
    """Return a judge for Opik's LLM-judge metrics that calls through ol-llm.

    Pass it as the ``model`` of an Opik judge metric that takes one, e.g.
    ``Hallucination(model=judge_model("learn-ai", "openai:gpt-4o-mini"))``.
    ``SycEval`` also needs it as ``rebuttal_model``, or it builds a LiteLLM model
    for the rebuttal; Opik's dataset-item evaluators (``LLMJudge``) take only a
    model name and cannot use it. The judge should not be the model under test
    (spec §5.1 item 7).

    :param app: the app being evaluated; the judge's calls are traced and costed
        under its project.
    :param model: the judge's model id, in ``provider:model`` form.
    :returns: the judge.
    :rtype: JudgeModel
    :raises ValueError: if the catalog does not list the model, or lists it as
        retired or as an embedding model. Unlike a call site, a judge may not use
        an uncataloged model: its spend is shared evaluation cost, and the catalog
        is where that is priced.
    """
    return JudgeModel(
        UseCase(app=app, name=JUDGE_USE_CASE, profile="batch", model=model)
    )


class JudgeModel(OpikBaseModel):
    """An ``OpikBaseModel`` that answers through ``complete``, for judge metrics.

    Opik's judge metrics send a system and a user message and, for a structured
    verdict, a pydantic model as ``response_format``. The system messages become
    the instructions, the rest the conversation, and a structured verdict comes
    back as its JSON, which is what the metrics parse.

    Build it with ``judge_model``, which names the use case ``eval-judge``.

    :param use_case: the judge's use case.
    :raises ValueError: if the catalog does not list the use case's model, or
        lists it as retired or as an embedding model.
    """

    def __init__(self, use_case: UseCase) -> None:
        """Check the model against the catalog; it is built on the first call."""
        if load_catalog().buildable(use_case.model, kind="chat") is None:
            msg = (
                f"{use_case.model!r} is not in the model catalog; a judge must use "
                "a listed model"
            )
            raise ValueError(msg)
        super().__init__(model_name=use_case.model)
        self.use_case = use_case

    def generate_string(
        self,
        input: str,  # noqa: A002 - the base class's parameter name
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Answer one user message.

        :param input: the message.
        :param response_format: the verdict's model, for a structured answer.
        :returns: the answer, as JSON for a structured one.
        :rtype: str
        :raises LLMError: on a failed call.
        """
        _no_options(kwargs)
        return _content(
            complete_sync(self.use_case, input, output_type=response_format or str)
        )

    async def agenerate_string(
        self,
        input: str,  # noqa: A002 - the base class's parameter name
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Answer one user message; see ``generate_string``."""
        _no_options(kwargs)
        result = await complete(
            self.use_case, input, output_type=response_format or str
        )
        return _content(result)

    def generate_chat_completion(
        self,
        messages: list[ConversationDict],
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ConversationDict:
        """Answer a conversation, as every Opik judge metric asks.

        :param messages: the conversation, in OpenAI's chat shape.
        :param response_format: the verdict's model, for a structured answer.
        :returns: the assistant's turn.
        :rtype: ConversationDict
        :raises ValueError: on a tool message, or a conversation that does not
            end with a user message.
        :raises LLMError: on a failed call.
        """
        _no_options(kwargs)
        instructions, history = _conversation(messages)
        result = complete_sync(
            self.use_case,
            history,
            output_type=response_format or str,
            instructions=instructions,
        )
        return {"role": "assistant", "content": _content(result)}

    async def agenerate_chat_completion(
        self,
        messages: list[ConversationDict],
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ConversationDict:
        """Answer a conversation; see ``generate_chat_completion``."""
        _no_options(kwargs)
        instructions, history = _conversation(messages)
        result = await complete(
            self.use_case,
            history,
            output_type=response_format or str,
            instructions=instructions,
        )
        return {"role": "assistant", "content": _content(result)}

    def generate_provider_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,  # noqa: ANN401
    ) -> LLMResult[str]:
        """Answer a conversation and return the whole result.

        :param messages: the conversation, in OpenAI's chat shape.
        :returns: the result.
        :rtype: LLMResult
        """
        _no_options(kwargs)
        instructions, history = _conversation(cast("list[ConversationDict]", messages))
        return complete_sync(self.use_case, history, instructions=instructions)

    async def agenerate_provider_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,  # noqa: ANN401
    ) -> LLMResult[str]:
        """Answer a conversation and return the whole result; see the sync form."""
        _no_options(kwargs)
        instructions, history = _conversation(cast("list[ConversationDict]", messages))
        return await complete(self.use_case, history, instructions=instructions)


def _no_options(kwargs: Mapping[str, Any]) -> None:
    # Sampling options belong to the use case and the call policy, not to each
    # metric; Opik's judge metrics pass none to a model they did not build.
    if kwargs:
        msg = f"JudgeModel takes no provider options, got {sorted(kwargs)}"
        raise TypeError(msg)


def _conversation(
    messages: Sequence[ConversationDict],
) -> tuple[str | None, list[ModelMessage]]:
    """Split a chat conversation into instructions and PydanticAI messages."""
    instructions: list[str] = []
    history: list[ModelMessage] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            instructions.append(message["content"])
        elif role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(message["content"])]))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(message["content"])]))
        else:
            msg = f"A judge conversation has no {role!r} messages"
            raise ValueError(msg)
    if not history or not isinstance(history[-1], ModelRequest):
        msg = "A judge conversation must end with a user message"
        raise ValueError(msg)
    return "\n\n".join(instructions) or None, history


def _content(result: LLMResult[Any]) -> str:
    output = result.output
    if isinstance(output, pydantic.BaseModel):
        return output.model_dump_json()
    return cast("str", output)


class ValidatorMetric(BaseMetric):
    """A deterministic check as an Opik metric, scored 1.0 on a pass and 0.0 not.

    The check takes the inputs it names from the dataset item and the task output
    (after ``scoring_key_mapping``) and returns a ``Verdict``: a bool, or the
    problems it found, which become the score's reason. A runtime validator is
    reused as it is, e.g. a placeholder check
    ``lambda source, output: placeholders(source) == placeholders(output)``.

    Opik records an exception raised by the check as a failed score, which
    ``check_thresholds`` counts against the metric. An item whose task raised an
    ``LLMError`` in ``evaluate`` scores 0.0 without running the check.

    :param name: the score's name.
    :param check: the validator.
    :param track: whether each check is a span of the item's trace.
    """

    def __init__(
        self, name: str, check: Callable[..., Verdict], *, track: bool = True
    ) -> None:
        """Hold the check and read which inputs it takes."""
        super().__init__(name=name, track=track)
        self._check = check
        parameters = inspect.signature(check).parameters.values()
        self._takes_any = any(p.kind is p.VAR_KEYWORD for p in parameters)
        self._accepted = {
            p.name
            for p in parameters
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        self._required = {
            p.name
            for p in parameters
            if p.name in self._accepted and p.default is p.empty
        }

    def score(self, **inputs: Any) -> score_result.ScoreResult:  # noqa: ANN401
        """Run the check on one item.

        :param inputs: the item's fields and the task's output.
        :returns: 1.0 when the check passed, 0.0 with the problems when not.
        :rtype: ScoreResult
        :raises ValueError: if an input the check needs is missing.
        :raises TypeError: if the check returns a string, which reads as neither a
            pass nor a list of problems.
        """
        if (error := inputs.get(LLM_ERROR_FIELD)) is not None:
            return score_result.ScoreResult(
                name=self.name, value=0.0, reason=f"the task raised {error}"
            )
        if missing := sorted(self._required - inputs.keys()):
            msg = (
                f"{self.name} needs {missing}; the item and output have "
                f"{sorted(inputs)}. Map them with scoring_key_mapping."
            )
            raise ValueError(msg)
        arguments = (
            inputs
            if self._takes_any
            else {k: v for k, v in inputs.items() if k in self._accepted}
        )
        verdict = self._check(**arguments)
        if isinstance(verdict, str):
            msg = f"{self.name} returned a string; return a bool or a list of problems"
            raise TypeError(msg)
        if isinstance(verdict, bool):
            passed, problems = verdict, []
        else:
            problems = list(verdict)
            passed = not problems
        return score_result.ScoreResult(
            name=self.name,
            value=1.0 if passed else 0.0,
            reason="; ".join(problems) or (None if passed else "check failed"),
        )


def load_items(path: Path, *, group: str | None = None) -> list[dict[str, Any]]:
    """Read golden items from a JSON file.

    The file holds a list of items, or an object of named lists, one per bot or
    use case, as learn-ai's ``test_json/rag_evaluation.json`` does; ``group``
    picks one of those.

    :param path: the file.
    :param group: the list to read from a file of named lists.
    :returns: the items, as the file has them.
    :rtype: list[dict[str, Any]]
    :raises ValueError: if the file is not one of those shapes, an item is not an
        object, ``group`` is missing for a file of named lists or given for a
        plain list, or names no list in it.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if group is None:
            msg = f"{path} holds named lists {sorted(data)}; pick one with group"
            raise ValueError(msg)
        if group not in data:
            msg = f"{path} has no list {group!r}; it has {sorted(data)}"
            raise ValueError(msg)
        data = data[group]
    elif group is not None:
        msg = f"{path} holds one list, so group {group!r} does not apply"
        raise ValueError(msg)
    if not isinstance(data, list) or not all(isinstance(i, dict) for i in data):
        msg = f"{path} must hold a list of objects, or an object of such lists"
        raise ValueError(msg)
    return data


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


def seed_dataset(
    name: str,
    items: Iterable[Mapping[str, Any]],
    *,
    project_name: str,
    description: str | None = None,
) -> opik.Dataset:
    """Add golden items to an Opik dataset, creating it if needed.

    Idempotent: the SDK hashes each item's content and skips one the dataset
    already holds, reading the existing items' hashes before the first insert into
    a dataset it fetched, and sends nothing when every item is already there. So
    seeding the same file again adds no items and no version; changing or adding
    items adds just those, as a new version. Items removed from the file stay in
    the dataset: it is curated in Opik (spec §5.1 item 6), and deleting there is a
    deliberate change.

    :param name: the dataset, e.g. ``<use_case>-golden``.
    :param items: the items; an item's keys are its fields.
    :param project_name: the app's Opik project.
    :param description: set on a dataset this creates.
    :returns: the dataset; ``get_current_version_name()`` gives the version to
        pin.
    :rtype: opik.Dataset
    :raises ValueError: if an item has a field Opik keeps for itself, e.g. ``id``
        or ``description``.
    """
    rows = [dict(item) for item in items]
    if reserved := sorted({key for row in rows for key in row} & _ITEM_FIELDS):
        msg = (
            f"Golden items may not have the fields {reserved}: Opik reads them as "
            "the dataset item's own (an id must be a UUID, a description is dropped "
            "from what the task sees). Rename them."
        )
        raise ValueError(msg)
    dataset = tracing.opik_client().get_or_create_dataset(
        name, description=description, project_name=project_name
    )
    dataset.insert(rows)
    return dataset


def _guard_task(
    task: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return the task with an ``LLMError`` turned into an output naming it.

    Opik aborts the whole run on any task exception (``engine.py``), so one
    content filter or exhausted retry would leave a gate with nothing to report.
    """

    @functools.wraps(task)
    def guarded(item: dict[str, Any]) -> dict[str, Any]:
        try:
            return task(item)
        except LLMError as exc:
            # The class, not the message: a provider's message can quote the request.
            log.warning(
                "The task raised %s; its item scores as a failure", type(exc).__name__
            )
            return {LLM_ERROR_FIELD: type(exc).__name__}

    return guarded


class _FailsAfterLLMError(BaseMetric):
    """A metric that reports a failed score for an item an ``LLMError`` cut short.

    Wraps every metric ``evaluate`` runs except ``ValidatorMetric``, which scores
    such an item 0.0 itself. A judge metric cannot do that: 0.0 is the passing
    end of some, e.g. ``Hallucination``. So the item gets a failed score, which
    Opik leaves out of the mean and does not store, and ``check_thresholds``
    counts as unscored. A judge's own ``LLMError`` is reported the same way,
    which also keeps Opik's rate-limit check, and its litellm import, off that
    path.

    :param metric: the metric.
    """

    def __init__(self, metric: BaseMetric) -> None:
        """Take the metric's name; the metric keeps its own tracking."""
        super().__init__(name=metric.name, track=False)
        self.metric = metric

    def score(
        self,
        **inputs: Any,  # noqa: ANN401
    ) -> score_result.ScoreResult | list[score_result.ScoreResult]:
        """Score the item with the metric, or fail it after an ``LLMError``."""
        if (error := inputs.get(LLM_ERROR_FIELD)) is not None:
            return self._failed(f"the task raised {error}")
        # What opik.evaluate does before calling a metric, so a missing input
        # still aborts the run as it would without the wrapper.
        validate_score_arguments(self.metric, inputs, None)
        positional, keyword = arguments_helpers.select_score_arguments(
            self.metric.score, inputs, self.name
        )
        try:
            return self.metric.score(*positional, **keyword)
        except LLMError as exc:
            return self._failed(f"the judge raised {type(exc).__name__}")

    def _failed(self, reason: str) -> score_result.ScoreResult:
        return score_result.ScoreResult(
            name=self.name, value=0.0, reason=reason, scoring_failed=True
        )


def _guard_metric(metric: BaseMetric) -> BaseMetric:
    if isinstance(metric, ValidatorMetric):
        return metric
    # Opik passes the task span only to a metric whose score names it, which a
    # wrapper would hide.
    if "task_span" in inspect.signature(metric.score).parameters:
        return metric
    return _FailsAfterLLMError(metric)


def evaluate(  # noqa: PLR0913
    task: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    dataset: str,
    project_name: str,
    metrics: Sequence[BaseMetric],
    prompts: Sequence[ManagedPrompt[Any]],
    dataset_version: str | None = None,
    experiment_name: str | None = None,
    experiment_config: Mapping[str, Any] | None = None,
    scoring_key_mapping: Mapping[str, str] | None = None,
    task_threads: int = 16,
) -> EvaluationResult:
    """Run an experiment on a dataset, linked to the prompt versions under test.

    ``opik.evaluate`` with ``prompts=`` set from the managed prompts, so Opik
    records which prompt versions and which dataset version the experiment ran
    against. The experiment is tagged with the app and environment, and its
    config records the environment. Buffered traces and spans are sent before it
    returns, so a CI job can exit straight after.

    An ``LLMError`` from the task does not abort the run, as Opik would: the
    item's output becomes ``{"ol_llm_error": "<error class>"}``, a
    ``ValidatorMetric`` scores it 0.0, and every other metric reports a failed
    score, which ``check_thresholds`` counts as unscored. A judge metric's
    ``LLMError`` is a failed score too. Any other exception from the task still
    aborts the run.

    :param task: called with each dataset item's fields; returns the output the
        metrics score.
    :param dataset: the dataset's name.
    :param project_name: the app's Opik project, which holds the dataset.
    :param metrics: the metrics, e.g. ``ValidatorMetric`` and Opik judge metrics
        built on ``judge_model``.
    :param prompts: every managed prompt the task uses, fetched from Opik. For a
        prompt change, the version under review; for a model change, the
        production version.
    :param dataset_version: the dataset version to run against, e.g. ``v3``; the
        latest when omitted. A gate pins it (spec §5.1 item 6).
    :param experiment_name: the experiment's name; Opik generates one when omitted.
    :param experiment_config: what describes the run, e.g. the model under test.
    :param scoring_key_mapping: metric input name to item or output field, e.g.
        ``{"input": "question"}``.
    :param task_threads: items run at once; 1 runs them in this thread.
    :returns: Opik's result, with the experiment's id and URL.
    :rtype: EvaluationResult
    :raises ValueError: if ``prompts`` is empty or holds an embedded default,
        which has no Opik version to link.
    """
    if not prompts:
        msg = "Pass every managed prompt the task uses, so the experiment links them"
        raise ValueError(msg)
    if embedded := [p.name for p in prompts if p.opik_prompt is None]:
        msg = (
            f"Prompts {embedded} are embedded defaults, not Opik versions; an "
            "evaluation needs Opik reachable to link what it ran"
        )
        raise ValueError(msg)
    environment = get_settings().environment
    source = tracing.opik_client().get_dataset(dataset, project_name=project_name)
    result = opik.evaluate(
        dataset=source
        if dataset_version is None
        else source.get_version_view(dataset_version),
        task=_guard_task(task),
        scoring_metrics=[_guard_metric(metric) for metric in metrics],
        experiment_name=experiment_name,
        experiment_config={"environment": environment, **(experiment_config or {})},
        prompts=[p.opik_prompt for p in prompts if p.opik_prompt is not None],
        scoring_key_mapping=dict(scoring_key_mapping or {}),
        experiment_tags=[f"app:{project_name}", f"env:{environment}"],
        task_threads=task_threads,
    )
    tracing.flush()
    return result


_THRESHOLD = re.compile(r"\s*(?P<metric>[^<>=\s]+)\s*(?P<op>>=|<=)\s*(?P<bound>\S+)\s*")


@dataclass(frozen=True, kw_only=True)
class Threshold:
    """A bound on one metric's mean over an experiment.

    A deterministic validator that must pass on every item (spec §5.1 item 5) is
    ``minimum=1.0``. A judge metric where higher is worse, such as Opik's
    ``Hallucination``, takes a ``maximum``.

    :param metric: the score's name.
    :param minimum: the lowest mean that passes.
    :param maximum: the highest mean that passes.
    :raises ValueError: if neither bound is set, or the minimum is above the
        maximum.
    """

    metric: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        """Reject a threshold that bounds nothing or nothing could pass."""
        if self.minimum is None and self.maximum is None:
            msg = f"Threshold for {self.metric!r} needs a minimum or a maximum"
            raise ValueError(msg)
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            msg = f"Threshold for {self.metric!r} has its minimum above its maximum"
            raise ValueError(msg)

    @classmethod
    def parse(cls, text: str) -> "Threshold":
        """Read a threshold written ``metric>=bound`` or ``metric<=bound``.

        :param text: e.g. ``hallucination_metric<=0.2``.
        :returns: the threshold.
        :rtype: Threshold
        :raises ValueError: if the text is not in that form.
        """
        if (match := _THRESHOLD.fullmatch(text)) is None:
            msg = f"{text!r} is not metric>=bound or metric<=bound"
            raise ValueError(msg)
        bound = float(match["bound"])
        if match["op"] == ">=":
            return cls(metric=match["metric"], minimum=bound)
        return cls(metric=match["metric"], maximum=bound)

    def describe(self) -> str:
        """Return the bound as text, e.g. ``>= 0.80``."""
        parts = []
        if self.minimum is not None:
            parts.append(f">= {self.minimum:.2f}")
        if self.maximum is not None:
            parts.append(f"<= {self.maximum:.2f}")
        return " and ".join(parts)


@dataclass(frozen=True, kw_only=True)
class MetricOutcome:
    """How one metric did against its threshold.

    :param threshold: the threshold.
    :param mean: the mean of the scored items, ``None`` when none was scored.
    :param scored: items with a score.
    :param unscored: items without one, because scoring failed or never ran.
    """

    threshold: Threshold
    mean: float | None
    scored: int
    unscored: int

    @property
    def passed(self) -> bool:
        """Whether every item was scored and the mean is within the bounds."""
        if self.mean is None or self.unscored:
            return False
        low, high = self.threshold.minimum, self.threshold.maximum
        return (low is None or self.mean >= low) and (high is None or self.mean <= high)

    def describe(self) -> str:
        """Return one report line."""
        status = "PASS" if self.passed else "FAIL"
        mean = "no scores" if self.mean is None else f"mean {self.mean:.2f}"
        counts = f"{self.scored} scored"
        if self.unscored:
            counts += f", {self.unscored} unscored"
        return (
            f"{status} {self.threshold.metric}: {mean}, needs "
            f"{self.threshold.describe()} ({counts})"
        )


@dataclass(frozen=True, kw_only=True)
class GateReport:
    """An experiment's scores against a use case's thresholds.

    :param experiment: the experiment's name, or its id when it has none.
    :param url: the experiment in Opik, when known.
    :param mode: whether a miss fails the check.
    :param outcomes: one per threshold, in the order given.
    """

    experiment: str
    url: str | None
    mode: GateMode
    outcomes: tuple[MetricOutcome, ...]

    @property
    def passed(self) -> bool:
        """Whether every threshold was met."""
        return all(outcome.passed for outcome in self.outcomes)

    def describe(self) -> str:
        """Return the report as text, one line per threshold."""
        heading = f"Experiment {self.experiment}" + (
            f" ({self.url})" if self.url else ""
        )
        missed = sum(not outcome.passed for outcome in self.outcomes)
        if not missed:
            verdict = "All thresholds met."
        elif self.mode == "advisory":
            verdict = f"{missed} threshold(s) not met; advisory, so not failing."
        else:
            verdict = f"{missed} threshold(s) not met."
        lines = [heading, *(o.describe() for o in self.outcomes), verdict]
        return "\n".join(lines)


class EvaluationGateError(Exception):
    """A blocking gate's thresholds were not met.

    :param report: the report.
    """

    def __init__(self, report: GateReport) -> None:
        """Keep the report, and use its text as the message."""
        super().__init__(report.describe())
        self.report = report


def check_thresholds(
    result: EvaluationResult,
    thresholds: Sequence[Threshold],
    *,
    mode: GateMode = "advisory",
) -> GateReport:
    """Compare an experiment's scores to its thresholds.

    A metric fails its threshold when any item has no score for it, because its
    scoring raised or the metric returned no score of that name, since a gate
    cannot vouch for items it did not score.

    :param result: what ``evaluate`` returned.
    :param thresholds: the use case's thresholds.
    :param mode: ``advisory`` logs a miss as a warning; ``blocking`` raises.
    :returns: the report.
    :rtype: GateReport
    :raises EvaluationGateError: in blocking mode, if a threshold was not met.
    """
    scores: dict[str, list[float]] = {}
    for test_result in result.test_results:
        for score in test_result.score_results:
            if not score.scoring_failed:
                scores.setdefault(score.name, []).append(score.value)
    return _enforce(
        _report(
            experiment=result.experiment_name or result.experiment_id,
            url=result.experiment_url,
            items=len(result.test_results),
            scores=scores,
            thresholds=thresholds,
            mode=mode,
        )
    )


def check_experiment(
    experiment_id: str,
    thresholds: Sequence[Threshold],
    *,
    mode: GateMode = "advisory",
) -> GateReport:
    """Compare a finished experiment's scores in Opik to its thresholds.

    For a CI step separate from the one that ran the experiment. Opik keeps only
    the scores that succeeded, so an item whose scoring failed counts as
    unscored, as in ``check_thresholds``. Reads the first 10,000 items, the SDK's
    default limit.

    :param experiment_id: the experiment's id, from ``EvaluationResult``.
    :param thresholds: the use case's thresholds.
    :param mode: ``advisory`` logs a miss as a warning; ``blocking`` raises.
    :returns: the report.
    :rtype: GateReport
    :raises EvaluationGateError: in blocking mode, if a threshold was not met.
    """
    return _enforce(_experiment_report(experiment_id, thresholds, mode=mode))


def _experiment_report(
    experiment_id: str, thresholds: Sequence[Threshold], *, mode: GateMode
) -> GateReport:
    experiment = tracing.opik_client().get_experiment_by_id(experiment_id)
    items = experiment.get_items()
    scores: dict[str, list[float]] = {}
    for item in items:
        for score in item.feedback_scores:
            scores.setdefault(score["name"], []).append(score["value"])
    return _report(
        experiment=experiment.name or experiment_id,
        url=None,
        items=len(items),
        scores=scores,
        thresholds=thresholds,
        mode=mode,
    )


def _report(  # noqa: PLR0913
    *,
    experiment: str,
    url: str | None,
    items: int,
    scores: Mapping[str, Sequence[float]],
    thresholds: Sequence[Threshold],
    mode: GateMode,
) -> GateReport:
    outcomes = []
    for threshold in thresholds:
        values = scores.get(threshold.metric, ())
        outcomes.append(
            MetricOutcome(
                threshold=threshold,
                mean=sum(values) / len(values) if values else None,
                scored=len(values),
                unscored=max(items - len(values), 0),
            )
        )
    return GateReport(
        experiment=experiment, url=url, mode=mode, outcomes=tuple(outcomes)
    )


def _enforce(report: GateReport) -> GateReport:
    """Raise for a blocking miss, log an advisory one, and return the report."""
    if not report.passed:
        if report.mode == "blocking":
            raise EvaluationGateError(report)
        log.warning("%s", report.describe())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``ol-llm-eval`` command.

    ``seed`` adds a file's golden items to a dataset and prints its version.
    ``check`` prints an experiment's scores against thresholds and exits 1 on a
    miss only with ``--mode blocking``.

    :param argv: the arguments; ``sys.argv[1:]`` when omitted.
    :returns: the exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        prog="ol-llm-eval", description="Golden datasets and gates in Opik."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser("seed", help="add a file's golden items to a dataset")
    seed.add_argument("file", type=Path, help="a JSON list of items, or named lists")
    seed.add_argument("--dataset", required=True, help="the dataset's name")
    seed.add_argument("--project", required=True, help="the app's Opik project")
    seed.add_argument("--group", help="the named list to read from the file")
    check = commands.add_parser(
        "check", help="check an experiment's scores against thresholds"
    )
    check.add_argument("experiment_id", help="the experiment's id")
    check.add_argument(
        "--threshold",
        type=Threshold.parse,
        action="append",
        required=True,
        help="metric>=bound or metric<=bound; repeat for each metric",
    )
    check.add_argument(
        "--mode",
        choices=["advisory", "blocking"],
        default="advisory",
        help="blocking exits 1 when a threshold is not met",
    )
    args = parser.parse_args(argv)
    if args.command == "seed":
        dataset = seed_dataset(
            args.dataset,
            load_items(args.file, group=args.group),
            project_name=args.project,
        )
        version = dataset.get_current_version_name() or "unversioned"
        sys.stdout.write(f"{args.dataset} {version}\n")
        return 0
    report = _experiment_report(
        args.experiment_id, args.threshold, mode=cast("GateMode", args.mode)
    )
    sys.stdout.write(f"{report.describe()}\n")
    return 1 if report.mode == "blocking" and not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
