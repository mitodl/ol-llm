import json
import logging
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ol_llm.catalog as catalog_module
import opik
import pytest
from conftest import PromptLibrary
from ol_llm import TextTemplate, UseCase, get_prompt, tracing
from ol_llm.catalog import Catalog, CatalogEntry, load_catalog
from ol_llm.evaluation import (
    JUDGE_USE_CASE,
    EvaluationGateError,
    JudgeModel,
    Threshold,
    ValidatorMetric,
    check_thresholds,
    evaluate,
    judge_model,
    load_items,
    main,
    seed_dataset,
)
from ol_llm.models import PolicyModel, override_models, shared_model
from ol_llm.prompts import ManagedPrompt
from opik.api_objects.dataset.dataset_item import DatasetItem
from opik.evaluation import test_case, test_result
from opik.evaluation.evaluation_result import EvaluationResult
from opik.evaluation.metrics import Hallucination
from opik.evaluation.metrics.score_result import ScoreResult
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

    def get_or_create_dataset(
        self, name: str, description: str | None, project_name: str
    ) -> FakeDataset:
        del description
        return self.datasets.setdefault((project_name, name), FakeDataset(name))

    def get_dataset(self, name: str, project_name: str) -> FakeDataset:
        return self.datasets[project_name, name]

    def get_experiment_by_id(self, experiment_id: str) -> Any:  # noqa: ANN401
        return self.experiments[experiment_id]

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
    assert call["task"] is task
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


@pytest.mark.parametrize("text", ["placeholders=1", "placeholders>=", ">=1"])
def test_parse_refuses_a_malformed_threshold(text: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        Threshold.parse(text)


def test_a_threshold_needs_a_satisfiable_bound() -> None:
    with pytest.raises(ValueError, match="minimum or a maximum"):
        Threshold(metric="placeholders")
    with pytest.raises(ValueError, match="above its maximum"):
        Threshold(metric="placeholders", minimum=0.9, maximum=0.1)


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
        name="summary-ci", get_items=lambda: items
    )

    argv = ["check", "exp-1", "--threshold", "placeholders>=1", "--mode", mode]

    assert main(argv) == status
    out = capsys.readouterr().out
    assert "Experiment summary-ci" in out
    assert "FAIL placeholders: mean 1.00, needs >= 1.00 (1 scored, 1 unscored)" in out
    assert out.count("FAIL placeholders") == 1
    assert not caplog.records


def test_the_check_command_passes_a_met_threshold(
    client: FakeClient, capsys: pytest.CaptureFixture[str]
) -> None:
    items = [SimpleNamespace(feedback_scores=[{"name": "placeholders", "value": 1.0}])]
    client.experiments["exp-1"] = SimpleNamespace(
        name="summary-ci", get_items=lambda: items
    )

    argv = ["check", "exp-1", "--threshold", "placeholders>=1", "--mode", "blocking"]

    assert main(argv) == 0
    assert "All thresholds met." in capsys.readouterr().out
