import json
import logging
import math
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ol_llm.catalog as catalog_module
import opik
import pytest
from conftest import PromptLibrary, preferring
from ol_llm import TextTemplate, UseCase, get_prompt, tracing
from ol_llm.catalog import Catalog, CatalogEntry, load_catalog
from ol_llm.errors import LLMContentFiltered, LLMUnavailable
from ol_llm.evaluation import (
    JUDGE_USE_CASE,
    LLM_ERROR_FIELD,
    MAX_EXPERIMENT_ITEMS,
    PAIRWISE_METRIC,
    EvaluationGateError,
    JudgeModel,
    MetricOutcome,
    SkippedItem,
    Threshold,
    ValidatorMetric,
    check_experiment,
    check_thresholds,
    evaluate,
    judge_model,
    judge_pairwise,
    load_items,
    main,
    seed_dataset,
)
from ol_llm.models import PolicyModel, override_models, shared_model
from ol_llm.prompts import ManagedPrompt
from opik.api_objects.dataset.dataset_item import DatasetItem
from opik.api_objects.experiment.experiment_item import ExperimentItemContent
from opik.evaluation import test_case, test_result
from opik.evaluation.evaluation_result import EvaluationResult
from opik.evaluation.metrics import BaseMetric, Hallucination
from opik.evaluation.metrics.score_result import ScoreResult
from opik.exceptions import ScoreMethodMissingArguments
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

APP = "mit-learn"
JUDGE = "openai:gpt-4o-mini"
UNCATALOGED = "openai:gpt-9-judge"
GOLDEN = [
    {"question": "Who teaches 8.01?", "expected_output": "Prof. Lewin"},
    {"question": "How is 8.01 graded?", "expected_output": "Exams: 70%"},
]


class FakeDataset:
    """An Opik dataset, keyed by the SDK's own content hash as its insert dedupes.

    Each insert that adds an item makes a new version, as Opik does.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.items: dict[str, dict[str, Any]] = {}
        self.versions = 0

    def insert(self, items: list[dict[str, Any]], deduplication: bool = True) -> None:  # noqa: FBT001, FBT002
        assert deduplication
        new = {DatasetItem(**item).content_hash(): item for item in items}
        if added := {h: i for h, i in new.items() if h not in self.items}:
            self.items.update(added)
            self.versions += 1

    def get_current_version_name(self) -> str | None:
        return f"v{self.versions}" if self.versions else None

    def get_version_view(self, version_name: str) -> tuple[str, str]:
        return (self.name, version_name)


class FakeClient:
    """Datasets and experiments; anything else goes to the SDK's real client."""

    def __init__(self, real: Callable[[], opik.Opik]) -> None:
        self._real = real
        self.datasets: dict[tuple[str, str], FakeDataset] = {}
        self.experiments: dict[str, Any] = {}
        self.records: dict[str, SimpleNamespace] = {}
        self.feedback_scores: list[dict[str, Any]] = []
        self.rest_client = SimpleNamespace(
            experiments=SimpleNamespace(
                get_experiment_by_id=lambda id: self.records[id]  # noqa: A006
            )
        )

    def get_or_create_dataset(
        self, name: str, description: str | None, project_name: str
    ) -> FakeDataset:
        del description
        return self.datasets.setdefault((project_name, name), FakeDataset(name))

    def get_dataset(self, name: str, project_name: str) -> FakeDataset:
        return self.datasets[project_name, name]

    def get_experiment_by_id(self, experiment_id: str) -> Any:  # noqa: ANN401
        return self.experiments[experiment_id]

    def log_traces_feedback_scores(self, scores: list[dict[str, Any]]) -> None:
        self.feedback_scores.extend(scores)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._real(), name)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient(tracing.opik_client)
    monkeypatch.setattr(tracing, "opik_client", lambda: fake)
    return fake


def scored(*rows: list[ScoreResult]) -> EvaluationResult:
    return EvaluationResult(
        experiment_id="exp-1",
        dataset_id="ds-1",
        experiment_name="summary-ci",
        test_results=[
            test_result.TestResult(
                test_case=test_case.TestCase(
                    trace_id=f"t{i}", dataset_item_id=f"i{i}", task_output={}
                ),
                score_results=row,
                trial_id=0,
            )
            for i, row in enumerate(rows)
        ],
        experiment_url="https://opik.example/experiments/exp-1",
        trial_count=1,
    )


def nonempty(output: str) -> bool:
    return bool(output)


def score(name: str, value: float, *, failed: bool = False) -> ScoreResult:
    return ScoreResult(name=name, value=value, scoring_failed=failed)


# Datasets


def test_seeding_twice_adds_nothing_the_second_time(client: FakeClient) -> None:
    first = seed_dataset("summary-golden", GOLDEN, project_name=APP)
    again = seed_dataset("summary-golden", [dict(i) for i in GOLDEN], project_name=APP)

    assert again is first
    assert len(client.datasets[APP, "summary-golden"].items) == len(GOLDEN)
    assert again.get_current_version_name() == "v1"


def test_seeding_a_changed_file_adds_only_the_new_items(client: FakeClient) -> None:
    seed_dataset("summary-golden", GOLDEN, project_name=APP)
    extra = {"question": "Is 8.01 free?", "expected_output": "Yes"}

    dataset = seed_dataset("summary-golden", [*GOLDEN, extra], project_name=APP)

    assert len(client.datasets[APP, "summary-golden"].items) == len(GOLDEN) + 1
    assert dataset.get_current_version_name() == "v2"


@pytest.mark.parametrize("field", ["id", "description"])
def test_seeding_refuses_a_field_opik_keeps_for_itself(
    client: FakeClient, field: str
) -> None:
    with pytest.raises(ValueError, match=f"fields \\['{field}'\\]"):
        seed_dataset("summary-golden", [{**GOLDEN[0], field: "x"}], project_name=APP)
    assert not client.datasets


def test_load_items_reads_one_named_list(tmp_path: Path) -> None:
    path = tmp_path / "rag_evaluation.json"
    path.write_text(json.dumps({"syllabus": GOLDEN, "tutor": [{"question": "?"}]}))

    assert load_items(path, group="syllabus") == GOLDEN


def test_load_items_reads_a_plain_list(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(GOLDEN))

    assert load_items(path) == GOLDEN


@pytest.mark.parametrize(
    ("content", "group", "message"),
    [
        ({"syllabus": GOLDEN}, None, "pick one with group"),
        ({"syllabus": GOLDEN}, "tutor", "no list 'tutor'"),
        (GOLDEN, "syllabus", "does not apply"),
        (["not an object"], None, "list of objects"),
    ],
)
def test_load_items_refuses_the_wrong_shape(
    tmp_path: Path, content: object, group: str | None, message: str
) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(content))

    with pytest.raises(ValueError, match=message):
        load_items(path, group=group)


# Experiments


@pytest.fixture
def production_prompt(
    prompt_library: PromptLibrary,
) -> ManagedPrompt[TextTemplate]:
    prompt_library.add("summary", "Summarize {{text}}.", project_name=APP)
    prompt_library.add(
        "summary",
        "Summarize {{text}} briefly.",
        project_name=APP,
        environments=("production",),
    )
    return get_prompt(
        "summary",
        project_name=APP,
        default=TextTemplate(template="Summarize {{text}}."),
        environment="production",
    )


@pytest.mark.usefixtures("client")
def test_evaluate_links_the_prompt_versions_to_the_experiment(
    monkeypatch: pytest.MonkeyPatch,
    production_prompt: ManagedPrompt[TextTemplate],
) -> None:
    seed_dataset("summary-golden", GOLDEN, project_name=APP)
    calls: list[dict[str, Any]] = []
    expected = scored([score("correct", 1.0)])

    def fake_evaluate(**kwargs: Any) -> EvaluationResult:  # noqa: ANN401
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(opik, "evaluate", fake_evaluate)
    metric = ValidatorMetric("correct", nonempty, track=False)

    def task(item: dict[str, Any]) -> dict[str, Any]:
        return {"output": item["question"]}

    result = evaluate(
        task,
        dataset="summary-golden",
        project_name=APP,
        metrics=[metric],
        prompts=[production_prompt],
        dataset_version="v1",
        experiment_config={"model": "openai:gpt-4o"},
        scoring_key_mapping={"input": "question"},
    )

    assert result is expected
    [call] = calls
    assert production_prompt.opik_prompt is not None
    assert production_prompt.version == "v2"
    assert call["prompts"] == [production_prompt.opik_prompt]
    assert call["dataset"] == ("summary-golden", "v1")
    assert call["task"].__wrapped__ is task
    assert call["scoring_metrics"] == [metric]
    assert call["experiment_tags"] == [f"app:{APP}", "env:ci"]
    assert call["experiment_config"] == {
        "environment": "ci",
        "model": "openai:gpt-4o",
    }
    assert call["scoring_key_mapping"] == {"input": "question"}


@pytest.mark.usefixtures("client")
def test_evaluate_runs_the_latest_dataset_version_unless_pinned(
    monkeypatch: pytest.MonkeyPatch,
    production_prompt: ManagedPrompt[TextTemplate],
) -> None:
    dataset = seed_dataset("summary-golden", GOLDEN, project_name=APP)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(opik, "evaluate", lambda **kwargs: calls.append(kwargs))

    evaluate(
        lambda item: item,
        dataset="summary-golden",
        project_name=APP,
        metrics=[],
        prompts=[production_prompt],
    )

    assert calls[0]["dataset"] is dataset


@pytest.mark.usefixtures("client")
def test_evaluate_refuses_an_embedded_default() -> None:
    embedded = ManagedPrompt(
        name="summary",
        template=TextTemplate(template="Summarize {{text}}."),
        source="embedded",
        degraded_reasons=("embedded_prompt",),
    )

    with pytest.raises(ValueError, match="embedded defaults"):
        evaluate(
            lambda item: item,
            dataset="summary-golden",
            project_name=APP,
            metrics=[],
            prompts=[embedded],
        )


@pytest.mark.usefixtures("client")
def test_evaluate_needs_the_prompts_it_runs() -> None:
    with pytest.raises(ValueError, match="every managed prompt"):
        evaluate(
            lambda item: item,
            dataset="summary-golden",
            project_name=APP,
            metrics=[],
            prompts=[],
        )


# Task and judge failures, through Opik's own evaluation engine

ON_OPIKS_ENGINE = pytest.mark.usefixtures("client", "opiks_engine")
VERDICT = {"score": 0.0, "reason": ["grounded"]}
FILTERED = "Who wrote the essay?"
RAG = [
    {"question": "Who teaches 8.01?", "context": ["Prof. Lewin teaches 8.01."]},
    {"question": FILTERED, "context": ["A learner wrote it."]},
]


@pytest.fixture
def opiks_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``opik.evaluate``'s item engine on the fake dataset's items.

    ``evaluate_on_dict_items`` drives the same ``EvaluationEngine.run_and_score``
    as ``opik.evaluate``, without the dataset and experiment calls the fake Opik
    does not answer.
    """
    on_items = opik.evaluate_on_dict_items

    def evaluate_items(
        *,
        dataset: FakeDataset,
        task: Callable[[dict[str, Any]], dict[str, Any]],
        scoring_metrics: list[Any],
        scoring_key_mapping: dict[str, str],
        task_threads: int,
        **_: object,
    ) -> EvaluationResult:
        result = on_items(
            list(dataset.items.values()),
            task,
            scoring_metrics=scoring_metrics,
            scoring_key_mapping=scoring_key_mapping,
            scoring_threads=task_threads,
        )
        return EvaluationResult(
            experiment_id="exp-1",
            dataset_id="ds-1",
            experiment_name="summary-ci",
            test_results=result.test_results,
            experiment_url=None,
            trial_count=1,
        )

    monkeypatch.setattr(opik, "evaluate", evaluate_items)


def answer(item: dict[str, Any]) -> dict[str, Any]:
    if item["question"] == FILTERED:
        msg = "filtered"
        raise LLMContentFiltered(msg, use_case="summary", model=JUDGE)
    return {"output": "Prof. Lewin."}


def run_rag(
    production_prompt: ManagedPrompt[TextTemplate],
    task: Callable[[dict[str, Any]], dict[str, Any]] = answer,
    mapping: dict[str, str] | None = None,
) -> EvaluationResult:
    seed_dataset("rag-golden", RAG, project_name=APP)
    return evaluate(
        task,
        dataset="rag-golden",
        project_name=APP,
        metrics=[
            ValidatorMetric("nonempty", nonempty, track=False),
            Hallucination(model=judge_model(APP, JUDGE), track=False),
        ],
        prompts=[production_prompt],
        scoring_key_mapping={"input": "question"} if mapping is None else mapping,
        task_threads=1,
    )


def by_question(result: EvaluationResult) -> dict[str, dict[str, ScoreResult]]:
    return {
        r.test_case.dataset_item_content["question"]: {
            s.name: s for s in r.score_results
        }
        for r in result.test_results
    }


@ON_OPIKS_ENGINE
def test_a_task_llm_error_fails_its_item_instead_of_the_run(
    production_prompt: ManagedPrompt[TextTemplate],
) -> None:
    with override_models(TestModel(custom_output_args=VERDICT)):
        result = run_rag(production_prompt)

    scores = by_question(result)
    good, filtered = scores[RAG[0]["question"]], scores[FILTERED]
    assert (good["nonempty"].value, good["hallucination_metric"].value) == (1.0, 0.0)
    assert not good["hallucination_metric"].scoring_failed
    assert (filtered["nonempty"].value, filtered["nonempty"].reason) == (
        0.0,
        "the task raised LLMContentFiltered",
    )
    assert not filtered["nonempty"].scoring_failed
    judged = filtered["hallucination_metric"]
    assert judged.scoring_failed
    assert judged.reason == "the task raised LLMContentFiltered"
    [failed] = [
        r for r in result.test_results if r.test_case.task_output.get(LLM_ERROR_FIELD)
    ]
    assert failed.test_case.task_output == {LLM_ERROR_FIELD: "LLMContentFiltered"}

    report = check_thresholds(
        result,
        [
            Threshold(metric="nonempty", minimum=1.0),
            Threshold(metric="hallucination_metric", maximum=0.2),
        ],
    )
    nonempty, hallucination = report.outcomes
    assert (nonempty.mean, nonempty.unscored, nonempty.passed) == (0.5, 0, False)
    assert (hallucination.mean, hallucination.unscored) == (0.0, 1)
    assert not hallucination.passed


@ON_OPIKS_ENGINE
def test_a_judge_llm_error_is_a_failed_score(
    production_prompt: ManagedPrompt[TextTemplate],
) -> None:
    def refuse(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(400, "gpt-4o-mini", body="no")

    with override_models(FunctionModel(refuse)):
        result = run_rag(production_prompt, task=lambda _item: {"output": "x"})

    for scores in by_question(result).values():
        judged = scores["hallucination_metric"]
        assert judged.scoring_failed
        assert judged.reason == "the judge raised LLMRejected"
        assert scores["nonempty"].value == 1.0


class SpanJudge(BaseMetric):
    """A judge that reads the task span, which Opik scores on a separate path."""

    def __init__(self) -> None:
        super().__init__(name="span_judge", track=False)

    def score(
        self,
        input: str,  # noqa: A002
        task_span: object,
        **_: object,
    ) -> ScoreResult:
        if input == FILTERED:
            msg = "judge down"
            raise LLMUnavailable(msg, use_case=JUDGE_USE_CASE, model=JUDGE)
        return ScoreResult(name=self.name, value=float(task_span is not None))


@ON_OPIKS_ENGINE
def test_a_span_reading_judges_llm_error_is_a_failed_score(
    production_prompt: ManagedPrompt[TextTemplate],
) -> None:
    seed_dataset("rag-golden", RAG, project_name=APP)
    result = evaluate(
        lambda _item: {"output": "x"},
        dataset="rag-golden",
        project_name=APP,
        metrics=[SpanJudge()],
        prompts=[production_prompt],
        scoring_key_mapping={"input": "question"},
        task_threads=1,
    )

    scores = by_question(result)
    assert scores[RAG[0]["question"]]["span_judge"].value == 1.0
    judged = scores[FILTERED]["span_judge"]
    assert judged.scoring_failed
    assert judged.reason == "the judge raised LLMUnavailable"


@ON_OPIKS_ENGINE
def test_any_other_task_error_still_aborts_the_run(
    production_prompt: ManagedPrompt[TextTemplate],
) -> None:
    def broken(_item: dict[str, Any]) -> dict[str, Any]:
        msg = "a bug in the task"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="a bug in the task"):
        run_rag(production_prompt, task=broken)


@ON_OPIKS_ENGINE
def test_a_missing_metric_input_still_aborts_the_run(
    production_prompt: ManagedPrompt[TextTemplate],
) -> None:
    with (
        override_models(TestModel(custom_output_args=VERDICT)),
        pytest.raises(ScoreMethodMissingArguments),
    ):
        run_rag(production_prompt, task=lambda _item: {"output": "x"}, mapping={})


# Metrics


def braces(text: str) -> set[str]:
    return {part.split("}")[0] for part in text.split("{")[1:]}


def same_placeholders(source: str, output: str) -> bool:
    return braces(source) == braces(output)


def test_a_validator_scores_a_pass_and_a_fail() -> None:
    metric = ValidatorMetric("placeholders", same_placeholders, track=False)

    passed = metric.score(source="Hi {name}", output="Hola {name}", extra="ignored")
    failed = metric.score(source="Hi {name}", output="Hola", extra="ignored")

    assert (passed.name, passed.value, passed.reason) == ("placeholders", 1.0, None)
    assert (failed.value, failed.reason) == (0.0, "check failed")


def test_a_validator_reports_the_problems_it_found() -> None:
    def cues(original: list[int], translated: list[int]) -> list[str]:
        return [f"cue {o}: missing" for o in original if o not in translated]

    metric = ValidatorMetric("srt_structure", cues, track=False)

    assert metric.score(original=[1, 2], translated=[1, 2]).value == 1.0
    result = metric.score(original=[1, 2, 3], translated=[1])
    assert (result.value, result.reason) == (0.0, "cue 2: missing; cue 3: missing")


def test_a_validator_names_the_inputs_it_is_missing() -> None:
    metric = ValidatorMetric("placeholders", same_placeholders, track=False)

    with pytest.raises(ValueError, match=r"needs \['source'\].*scoring_key_mapping"):
        metric.score(output="Hola")


def test_a_validator_may_not_return_a_string() -> None:
    metric = ValidatorMetric("placeholders", lambda output: output, track=False)

    with pytest.raises(TypeError, match="returned a string"):
        metric.score(output="looks fine")


def test_an_opik_judge_metric_scores_through_the_library() -> None:
    judge = judge_model(APP, JUDGE)
    verdict = {"score": 0.0, "reason": ["Every claim is in the context."]}
    metric = Hallucination(model=judge, track=False)

    with (
        override_models(TestModel(custom_output_args=verdict)),
        opik.record_traces_locally() as storage,
    ):
        result = metric.score(
            input="Who teaches 8.01?",
            output="Prof. Lewin.",
            context=["8.01 is taught by Prof. Lewin."],
        )
        [trace] = storage.trace_trees
    tracing.flush()

    assert isinstance(result, ScoreResult)
    assert (result.name, result.value) == ("hallucination_metric", 0.0)
    assert result.reason == "Every claim is in the context."
    assert trace.name == JUDGE_USE_CASE
    assert trace.project_name == APP
    assert {f"app:{APP}", f"use_case:{JUDGE_USE_CASE}", "env:ci"} <= set(trace.tags)


def test_the_judge_sends_system_messages_as_instructions() -> None:
    seen: list[tuple[list[ModelMessage], AgentInfo]] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append((messages, info))
        [tool] = info.output_tools
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {"score": 1.0, "reason": ["x"]})]
        )

    with override_models(FunctionModel(reply)):
        Hallucination(model=judge_model(APP, JUDGE), track=False).score(
            input="q", output="a", context=["c"]
        )

    [(messages, info)] = seen
    [request] = messages
    assert isinstance(request, ModelRequest)
    assert info.instructions is not None
    assert "hallucination" in info.instructions.lower()
    [part] = request.parts
    assert isinstance(part, UserPromptPart)
    assert "c" in str(part.content)


def test_the_judge_answers_free_text() -> None:
    judge = judge_model(APP, JUDGE)

    with override_models(TestModel(custom_output_text="Clarity, accuracy.")):
        reply = judge.generate_chat_completion(
            [
                {"role": "system", "content": "List criteria."},
                {"role": "user", "content": "For a summary."},
            ]
        )
        text = judge.generate_string("For a summary.")

    assert reply == {"role": "assistant", "content": "Clarity, accuracy."}
    assert text == "Clarity, accuracy."


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ([{"role": "tool", "content": "x", "tool_call_id": "1"}], ValueError),
        (
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            ValueError,
        ),
    ],
)
def test_the_judge_refuses_a_conversation_it_cannot_send(
    messages: list[Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        judge_model(APP, JUDGE).generate_chat_completion(messages)


def test_the_judge_takes_no_provider_options() -> None:
    with pytest.raises(TypeError, match="no provider options"):
        judge_model(APP, JUDGE).generate_string("q", temperature=0)


@pytest.mark.usefixtures("provider_env")
def test_the_judge_model_comes_from_the_catalog_under_the_judge_use_case() -> None:
    judge = judge_model(APP, JUDGE)

    assert (judge.use_case.app, judge.use_case.name) == (APP, JUDGE_USE_CASE)
    assert judge.use_case.profile == "batch"
    assert judge.model_name == JUDGE
    model = shared_model(judge.use_case)
    assert isinstance(model, PolicyModel)
    assert model.catalog_entry is not None
    assert model.catalog_entry.id == JUDGE


def test_an_uncataloged_judge_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in the model catalog"):
        judge_model(APP, UNCATALOGED)
    use_case = UseCase(app=APP, name="judge", profile="batch", model=UNCATALOGED)
    with pytest.raises(ValueError, match="not in the model catalog"):
        JudgeModel(use_case)


def test_a_retired_judge_model_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    retired = CatalogEntry(
        id=UNCATALOGED,
        kind="chat",
        status="retired",
        input_usd_per_mtok=Decimal(1),
        output_usd_per_mtok=Decimal(1),
    )
    catalog = Catalog({**load_catalog().entries, UNCATALOGED: retired})
    monkeypatch.setattr(catalog_module, "load_catalog", lambda: catalog)
    monkeypatch.setattr("ol_llm.evaluation.load_catalog", lambda: catalog)

    with pytest.raises(ValueError, match="retired"):
        judge_model(APP, UNCATALOGED)


# Thresholds

THRESHOLDS = [
    Threshold(metric="placeholders", minimum=1.0),
    Threshold(metric="hallucination_metric", maximum=0.2),
]


def test_met_thresholds_pass_in_either_mode() -> None:
    result = scored(
        [score("placeholders", 1.0), score("hallucination_metric", 0.0)],
        [score("placeholders", 1.0), score("hallucination_metric", 0.2)],
    )

    for mode in ("advisory", "blocking"):
        report = check_thresholds(result, THRESHOLDS, mode=mode)
        assert report.passed
        assert [o.mean for o in report.outcomes] == [1.0, 0.1]


def test_a_blocking_gate_raises_on_a_miss() -> None:
    result = scored(
        [score("placeholders", 1.0), score("hallucination_metric", 0.0)],
        [score("placeholders", 0.0), score("hallucination_metric", 1.0)],
    )

    with pytest.raises(EvaluationGateError) as caught:
        check_thresholds(result, THRESHOLDS, mode="blocking")

    report = caught.value.report
    assert not report.passed
    assert [o.passed for o in report.outcomes] == [False, False]
    assert "FAIL placeholders: mean 0.50, needs >= 1.00 (2 scored)" in str(caught.value)
    assert "https://opik.example/experiments/exp-1" in str(caught.value)


def test_an_advisory_gate_reports_a_miss_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = scored([score("placeholders", 0.0), score("hallucination_metric", 0.0)])

    with caplog.at_level(logging.WARNING, logger="ol_llm.evaluation"):
        report = check_thresholds(result, THRESHOLDS)

    assert report.mode == "advisory"
    assert not report.passed
    assert "advisory, so not failing" in report.describe()
    assert "FAIL placeholders" in caplog.text


def test_an_unscored_item_fails_its_metric() -> None:
    result = scored(
        [score("placeholders", 1.0), score("hallucination_metric", 0.0)],
        [score("placeholders", 1.0), score("hallucination_metric", 0.0, failed=True)],
        [score("placeholders", 1.0)],
    )

    report = check_thresholds(result, THRESHOLDS)

    placeholders, hallucination = report.outcomes
    assert placeholders.passed
    assert (hallucination.mean, hallucination.scored, hallucination.unscored) == (
        0.0,
        1,
        2,
    )
    assert not hallucination.passed


def test_an_item_with_two_scores_of_one_name_is_refused() -> None:
    result = scored(
        [score("placeholders", 1.0), score("placeholders", 1.0)],
        [score("placeholders", 0.0, failed=True)],
    )

    with pytest.raises(ValueError, match="two scores named 'placeholders'"):
        check_thresholds(result, THRESHOLDS)


def test_a_metric_with_no_scores_fails() -> None:
    report = check_thresholds(scored([score("placeholders", 1.0)]), THRESHOLDS)

    assert "FAIL hallucination_metric: no scores" in report.describe()


@pytest.mark.parametrize(
    ("text", "threshold"),
    [
        ("placeholders>=1", Threshold(metric="placeholders", minimum=1.0)),
        (
            " hallucination_metric <= 0.2 ",
            Threshold(metric="hallucination_metric", maximum=0.2),
        ),
    ],
)
def test_parse_a_threshold(text: str, threshold: Threshold) -> None:
    assert Threshold.parse(text) == threshold


@pytest.mark.parametrize(
    "text", ["placeholders=1", "placeholders>=", ">=1", "placeholders>=nan"]
)
def test_parse_refuses_a_malformed_threshold(text: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        Threshold.parse(text)


def test_a_threshold_needs_a_satisfiable_bound() -> None:
    with pytest.raises(ValueError, match="a minimum, a maximum, or a tolerance"):
        Threshold(metric="placeholders")
    with pytest.raises(ValueError, match="above its maximum"):
        Threshold(metric="placeholders", minimum=0.9, maximum=0.1)
    with pytest.raises(ValueError, match="negative tolerance"):
        Threshold(metric="placeholders", tolerance=-0.1, direction="higher")


@pytest.mark.parametrize(
    "limits",
    [
        {"minimum": math.nan},
        {"maximum": math.nan},
        {"minimum": -math.inf},
        {"minimum": 0.5, "tolerance": math.nan},
    ],
)
def test_a_threshold_refuses_a_bound_that_is_not_finite(
    limits: dict[str, float],
) -> None:
    # A NaN minimum would pass any finite mean, since mean < nan is false.
    with pytest.raises(ValueError, match="not a finite number"):
        Threshold(metric="placeholders", **limits)


def test_a_tolerance_with_no_bound_needs_a_direction() -> None:
    # Guessing "higher" would pass a rise in hallucination_metric.
    with pytest.raises(ValueError, match='needs direction = "higher" or "lower"'):
        Threshold(metric="hallucination_metric", tolerance=0.05)


@pytest.mark.parametrize(
    ("threshold", "better"),
    [
        (Threshold(metric="m", tolerance=0.1, direction="higher"), "higher"),
        (Threshold(metric="m", minimum=0.5, tolerance=0.1), "higher"),
        (Threshold(metric="m", maximum=0.5, tolerance=0.1), "lower"),
        (Threshold(metric="m", tolerance=0.1, direction="lower"), "lower"),
        (Threshold(metric="m", maximum=0.5, direction="higher"), "higher"),
    ],
)
def test_a_threshold_knows_which_way_is_better(
    threshold: Threshold, better: str
) -> None:
    assert threshold.better == better


HIGHER = Threshold(metric="m", tolerance=0.1, direction="higher")


@pytest.mark.parametrize(
    ("threshold", "values", "status"),
    [
        # Higher is better: at most 0.1 below the baseline of 0.8.
        (HIGHER, [0.7, 0.7], "pass"),
        (HIGHER, [0.6, 0.7], "fail"),
        # Lower is better: at most 0.1 above the baseline of 0.8.
        (Threshold(metric="m", maximum=1.0, tolerance=0.1), [0.9, 0.9], "pass"),
        (Threshold(metric="m", maximum=1.0, tolerance=0.1), [1.0, 0.9], "fail"),
        # The fixed bound still applies within the tolerance.
        (Threshold(metric="m", minimum=0.75, tolerance=0.1), [0.7, 0.7], "fail"),
        # A mean that is not a number never passes.
        (HIGHER, [math.nan], "fail"),
    ],
)
def test_a_tolerance_is_measured_from_the_baseline(
    threshold: Threshold, values: list[float], status: str
) -> None:
    result = scored(*([score("m", v)] for v in values))

    [outcome] = check_thresholds(result, [threshold], baselines={"m": 0.8}).outcomes

    assert outcome.baseline == 0.8
    assert outcome.status == status


@pytest.mark.parametrize("value", [0.0, math.nan])
def test_a_tolerance_with_no_baseline_is_not_compared(value: float) -> None:
    result = scored([score("m", value)])

    report = check_thresholds(result, [HIGHER], baselines={"other": 1.0})

    [outcome] = report.outcomes
    if math.isnan(value):
        assert outcome.status == "fail"
        assert not report.passed
        return
    assert outcome.status == "not compared"
    assert not outcome.passed
    assert not outcome.failed
    assert report.passed
    assert outcome.describe() == (
        "NOT COMPARED m: mean 0.00, needs at most 0.10 below baseline "
        "(1 scored, no baseline)"
    )
    assert report.describe().endswith(
        "No threshold missed; 1 not compared, for want of a baseline."
    )


def test_without_a_baseline_the_fixed_bounds_still_apply() -> None:
    threshold = Threshold(metric="m", minimum=0.5, tolerance=0.1)

    [low] = check_thresholds(scored([score("m", 0.4)]), [threshold]).outcomes
    [high] = check_thresholds(scored([score("m", 0.6)]), [threshold]).outcomes

    assert (low.status, high.status) == ("fail", "pass")


# The command


def test_the_seed_command_prints_the_version(
    client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "rag_evaluation.json"
    path.write_text(json.dumps({"syllabus": GOLDEN}))
    argv = [
        "seed",
        str(path),
        "--dataset",
        "syllabus-golden",
        "--project",
        APP,
        "--group",
        "syllabus",
    ]

    assert main(argv) == 0
    assert main(argv) == 0

    assert capsys.readouterr().out.splitlines() == [
        "syllabus-golden v1",
        "syllabus-golden v1",
    ]
    assert len(client.datasets[APP, "syllabus-golden"].items) == len(GOLDEN)


@pytest.mark.parametrize(("mode", "status"), [("advisory", 0), ("blocking", 1)])
def test_the_check_command_fails_only_when_blocking(
    client: FakeClient,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    mode: str,
    status: int,
) -> None:
    items = [
        SimpleNamespace(feedback_scores=[{"name": "placeholders", "value": 1.0}]),
        SimpleNamespace(feedback_scores=[]),
    ]
    client.experiments["exp-1"] = SimpleNamespace(
        name="summary-ci", get_items=lambda max_results: items[:max_results]
    )

    argv = ["check", "exp-1", "--threshold", "placeholders>=1", "--mode", mode]

    assert main(argv) == status
    out = capsys.readouterr().out
    assert "Experiment summary-ci" in out
    assert "FAIL placeholders: mean 1.00, needs >= 1.00 (1 scored, 1 unscored)" in out
    assert out.count("FAIL placeholders") == 1
    assert not caplog.records


def test_a_null_experiment_score_counts_as_unscored(client: FakeClient) -> None:
    items = [
        SimpleNamespace(feedback_scores=[{"name": "placeholders", "value": 1.0}]),
        SimpleNamespace(feedback_scores=[{"name": "placeholders", "value": None}]),
    ]
    client.experiments["exp-1"] = SimpleNamespace(
        name="summary-ci", get_items=lambda max_results: items[:max_results]
    )

    [outcome] = check_experiment(
        "exp-1", [Threshold(metric="placeholders", minimum=1.0)]
    ).outcomes

    assert (outcome.mean, outcome.scored, outcome.unscored) == (1.0, 1, 1)
    assert not outcome.passed


def test_an_experiment_past_the_sdks_item_limit_is_refused(
    client: FakeClient,
) -> None:
    item = SimpleNamespace(feedback_scores=[{"name": "placeholders", "value": 1.0}])
    items = [item] * (MAX_EXPERIMENT_ITEMS + 5)
    client.experiments["exp-1"] = SimpleNamespace(
        name="summary-ci", get_items=lambda max_results: items[:max_results]
    )

    with pytest.raises(ValueError, match="more than 10000 items"):
        check_experiment(
            "exp-1", [Threshold(metric="placeholders", minimum=1.0)], mode="blocking"
        )


def test_the_check_command_passes_a_met_threshold(
    client: FakeClient, capsys: pytest.CaptureFixture[str]
) -> None:
    items = [SimpleNamespace(feedback_scores=[{"name": "placeholders", "value": 1.0}])]
    client.experiments["exp-1"] = SimpleNamespace(
        name="summary-ci", get_items=lambda max_results: items[:max_results]
    )

    argv = ["check", "exp-1", "--threshold", "placeholders>=1", "--mode", "blocking"]

    assert main(argv) == 0
    assert "All thresholds met." in capsys.readouterr().out


# Pairwise


def store_experiment(
    client: FakeClient,
    experiment_id: str,
    outputs: dict[str, dict[str, Any]],
    *,
    dataset_version_id: str | None = "dv-1",
) -> None:
    """An experiment in Opik whose item for each dataset item id has an output."""
    items = [
        ExperimentItemContent(
            id=f"{experiment_id}-{item_id}",
            dataset_item_id=item_id,
            trace_id=f"{experiment_id}-trace-{item_id}",
            dataset_item_data={"question": f"Question {item_id}"},
            evaluation_task_output=output,
            feedback_scores=[],
        )
        for item_id, output in outputs.items()
    ]
    client.experiments[experiment_id] = SimpleNamespace(
        name=experiment_id, get_items=lambda max_results: items[:max_results]
    )
    client.records[experiment_id] = SimpleNamespace(
        dataset_id="ds-1", dataset_version_id=dataset_version_id
    )


def candidate_result(outputs: dict[str, dict[str, Any]]) -> EvaluationResult:
    return EvaluationResult(
        experiment_id="candidate",
        dataset_id="ds-1",
        experiment_name="summary pr-1",
        test_results=[
            test_result.TestResult(
                test_case=test_case.TestCase(
                    trace_id=f"candidate-trace-{item_id}",
                    dataset_item_id=item_id,
                    task_output=output,
                    dataset_item_content={"question": f"Question {item_id}"},
                ),
                score_results=[],
                trial_id=0,
            )
            for item_id, output in outputs.items()
        ],
        experiment_url=None,
        trial_count=1,
    )


PAIRWISE_CRITERIA = "Prefer the answer that names the lecturer."
LEWIN = {"output": "Prof. Lewin"}
UNKNOWN = {"output": "Not sure"}


def test_pairwise_judges_each_pair_in_both_orders(client: FakeClient) -> None:
    store_experiment(
        client,
        "baseline",
        {
            "won": UNKNOWN,
            "lost": LEWIN,
            "tied": LEWIN,
            "reference-only": LEWIN,
            "reference-raised": {LLM_ERROR_FIELD: "LLMContentFiltered"},
            "candidate-raised": LEWIN,
        },
    )
    client.records["candidate"] = client.records["baseline"]
    candidate = candidate_result(
        {
            "won": LEWIN,
            "lost": UNKNOWN,
            "tied": LEWIN,
            "candidate-only": LEWIN,
            "reference-raised": LEWIN,
            "candidate-raised": {LLM_ERROR_FIELD: "LLMRejected"},
        }
    )
    seen: list[str] = []

    with (
        override_models(preferring("Lewin", seen)),
        opik.record_traces_locally() as storage,
    ):
        comparison = judge_pairwise(
            candidate,
            "baseline",
            judge=judge_model(APP, JUDGE),
            criteria=PAIRWISE_CRITERIA,
            project_name=APP,
        )
        tracing.flush()
        traces = storage.trace_trees

    assert [
        (v.dataset_item_id, v.outcome, v.orders_agreed) for v in comparison.verdicts
    ] == [
        ("won", "win", True),
        ("lost", "loss", True),
        ("tied", "tie", False),
    ]
    assert comparison.win_rate == 0.5
    assert comparison.disagreements == 1
    assert sorted(comparison.unpaired, key=lambda s: s.dataset_item_id) == [
        SkippedItem(dataset_item_id="candidate-only", reason="only in the candidate"),
        SkippedItem(dataset_item_id="reference-only", reason="only in the reference"),
        SkippedItem(
            dataset_item_id="reference-raised",
            reason="the reference's task raised LLMContentFiltered",
        ),
    ]
    assert comparison.unjudged == (
        SkippedItem(
            dataset_item_id="candidate-raised",
            reason="the candidate's task raised LLMRejected",
        ),
    )
    # Two calls per judged pair: the candidate's answer as A, then as B.
    assert len(seen) == 6
    won = [text for text in seen if "Question won" in text]
    assert sorted(
        text.index("Prof. Lewin") < text.index("Not sure") for text in won
    ) == [
        False,
        True,
    ]
    assert len(traces) == 6
    assert {trace.name for trace in traces} == {JUDGE_USE_CASE}
    assert {trace.project_name for trace in traces} == {APP}

    assert [
        (s["id"], s["name"], s["value"], s["category_name"], s["project_name"])
        for s in client.feedback_scores
    ] == [
        ("candidate-trace-won", PAIRWISE_METRIC, 1.0, "win", APP),
        ("candidate-trace-lost", PAIRWISE_METRIC, 0.0, "loss", APP),
        ("candidate-trace-tied", PAIRWISE_METRIC, 0.5, "tie", APP),
    ]
    tied = client.feedback_scores[2]["reason"]
    assert tied.startswith("The two answer orders disagreed, so this is a tie.")
    assert "Candidate as A: A is it" in tied
    assert "Candidate as B: A is it" in tied


def test_the_win_rate_is_a_thresholdable_outcome(client: FakeClient) -> None:
    store_experiment(client, "baseline", {"a": UNKNOWN, "b": UNKNOWN, "c": LEWIN})
    store_experiment(client, "candidate", {"a": LEWIN, "b": UNKNOWN, "c": UNKNOWN})
    threshold = Threshold(metric=PAIRWISE_METRIC, minimum=0.45)

    with override_models(preferring("Lewin")):
        comparison = judge_pairwise(
            "candidate",
            "baseline",
            judge=judge_model(APP, JUDGE),
            criteria=PAIRWISE_CRITERIA,
            project_name=APP,
            threads=1,
        )
    outcome = comparison.outcome(threshold)

    # A win, a tie, and a loss: (1 + 0.5 + 0) / 3.
    assert outcome.mean == 0.5
    assert outcome.passed
    assert outcome.describe() == (
        "PASS win_rate_vs_baseline: mean 0.50, needs >= 0.45 "
        "(3 scored, 1 won, 1 tied, 1 lost)"
    )
    assert [s["id"] for s in client.feedback_scores] == [
        "candidate-trace-a",
        "candidate-trace-b",
        "candidate-trace-c",
    ]
    assert comparison.describe() == (
        "Judged 3 pair(s) in both answer orders: 1 won, 1 tied (1 because the "
        "orders disagreed), 1 lost."
    )


def test_a_judge_error_leaves_its_pair_unjudged(client: FakeClient) -> None:
    store_experiment(client, "baseline", {"a": UNKNOWN})
    store_experiment(client, "candidate", {"a": LEWIN})

    def refuse(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(400, "gpt-4o-mini", body="no")

    with override_models(FunctionModel(refuse)):
        comparison = judge_pairwise(
            "candidate",
            "baseline",
            judge=judge_model(APP, JUDGE),
            criteria=PAIRWISE_CRITERIA,
            project_name=APP,
        )

    assert comparison.verdicts == ()
    assert comparison.unjudged == (
        SkippedItem(dataset_item_id="a", reason="the judge raised LLMRejected"),
    )
    assert comparison.win_rate is None
    assert not client.feedback_scores
    outcome = comparison.outcome(Threshold(metric=PAIRWISE_METRIC, minimum=0.45))
    assert (outcome.scored, outcome.unscored, outcome.status) == (0, 1, "fail")


@pytest.mark.parametrize("version", ["dv-2", None])
def test_pairwise_needs_one_dataset_version(
    client: FakeClient, version: str | None
) -> None:
    store_experiment(client, "baseline", {"a": UNKNOWN})
    store_experiment(client, "candidate", {"a": LEWIN}, dataset_version_id=version)

    with pytest.raises(ValueError, match="same dataset version"):
        judge_pairwise(
            "candidate",
            "baseline",
            judge=judge_model(APP, JUDGE),
            criteria=PAIRWISE_CRITERIA,
            project_name=APP,
        )


def test_pairwise_refuses_two_trials_of_one_item(client: FakeClient) -> None:
    store_experiment(client, "baseline", {"a": UNKNOWN})
    client.records["candidate"] = client.records["baseline"]
    twice = candidate_result({"a": LEWIN})
    twice.test_results.extend(twice.test_results)

    with pytest.raises(ValueError, match="one trial per item"):
        judge_pairwise(
            twice,
            "baseline",
            judge=judge_model(APP, JUDGE),
            criteria=PAIRWISE_CRITERIA,
            project_name=APP,
        )


def test_pairwise_names_a_candidate_answer_field_it_lacks(client: FakeClient) -> None:
    store_experiment(client, "baseline", {"a": {"answer": "Prof. Lewin"}})
    store_experiment(client, "candidate", {"a": UNKNOWN})

    with pytest.raises(
        ValueError,
        match=r"candidate's answer to dataset item a has no \['answer'\]; it has "
        r"\['output'\]",
    ):
        judge_pairwise(
            "candidate",
            "baseline",
            judge=judge_model(APP, JUDGE),
            criteria=PAIRWISE_CRITERIA,
            project_name=APP,
            answer_fields=["answer"],
        )


def test_pairwise_names_a_field_neither_experiment_has(client: FakeClient) -> None:
    # Unpaired, every item would make the metric "not compared" and let a
    # default-branch run record a baseline that still lacks the field.
    store_experiment(client, "baseline", {"a": UNKNOWN})
    store_experiment(client, "candidate", {"a": LEWIN})

    with pytest.raises(ValueError, match=r"has no \['answer'\]"):
        judge_pairwise(
            "candidate",
            "baseline",
            judge=judge_model(APP, JUDGE),
            criteria=PAIRWISE_CRITERIA,
            project_name=APP,
            answer_fields=["answer"],
        )


def test_a_reference_without_the_answer_fields_is_not_compared(
    client: FakeClient,
) -> None:
    # The suite now returns "answer"; the reference ran before it did.
    store_experiment(client, "baseline", {"a": UNKNOWN})
    store_experiment(client, "candidate", {"a": {"answer": "Prof. Lewin"}})

    comparison = judge_pairwise(
        "candidate",
        "baseline",
        judge=judge_model(APP, JUDGE),
        criteria=PAIRWISE_CRITERIA,
        project_name=APP,
        answer_fields=["answer"],
    )

    assert comparison.unpaired == (
        SkippedItem(
            dataset_item_id="a", reason="the reference's answer has no ['answer']"
        ),
    )
    assert not client.feedback_scores
    outcome = comparison.outcome(Threshold(metric=PAIRWISE_METRIC, minimum=0.45))
    assert outcome.status == "not compared"
    assert outcome.describe().endswith("(0 scored, no item paired, 1 unpaired)")


def test_an_uncompared_outcome_neither_passes_nor_fails() -> None:
    outcome = MetricOutcome(
        threshold=Threshold(metric=PAIRWISE_METRIC, minimum=0.45),
        mean=None,
        scored=0,
        unscored=0,
        compared=False,
        detail="no baseline",
    )

    assert outcome.status == "not compared"
    assert not outcome.failed
    assert outcome.describe() == (
        "NOT COMPARED win_rate_vs_baseline: no scores, needs >= 0.45 "
        "(0 scored, no baseline)"
    )
