"""The experiment backend interface, its in-memory backend, and the gate on it."""

from collections.abc import Generator, Iterator, Mapping, Sequence
from typing import Any

import pytest
from conftest import PromptLibrary, preferring
from ol_llm import TextTemplate, get_prompt
from ol_llm.errors import LLMContentFiltered
from ol_llm.evaluation import (
    PAIRWISE_METRIC,
    Score,
    Threshold,
    ValidatorMetric,
    check_experiment,
    evaluate,
    hallucination,
    judge_model,
    judge_pairwise,
    seed_dataset,
)
from ol_llm.experiments import (
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
    ExperimentBackend,
    ExperimentRun,
    ExperimentSummary,
    MemoryExperimentBackend,
    Metric,
    MissingMetricInputError,
    OpikExperimentBackend,
    TraceScore,
    configure_experiment_backend,
    experiment_backend,
    register_experiment_backend,
)
from ol_llm.gate import (
    BASELINE_TAG,
    GateConfigurationError,
    GateContext,
    GateDeclaration,
    Suite,
    run_gate,
)
from ol_llm.models import override_models
from ol_llm.prompts import ManagedPrompt

APP = "mit-learn"
USE_CASE = "summary"
DATASET = "summary-golden"
GOLDEN = [
    {"text": "The course was {great}.", "expected": "{great}"},
    {"text": "Too long; {lectures} dragged.", "expected": "{lectures}"},
]


@pytest.fixture
def memory() -> Generator[MemoryExperimentBackend]:
    backend = MemoryExperimentBackend()
    configure_experiment_backend(backend)
    yield backend
    configure_experiment_backend(None)


@pytest.fixture
def prompts(prompt_library: PromptLibrary) -> PromptLibrary:
    """v1 in production, v2 under review."""
    prompt_library.add(
        USE_CASE, "Summarize {{text}}.", project_name=APP, environments=("production",)
    )
    prompt_library.add(USE_CASE, "Summarize {{text}}, briefly.", project_name=APP)
    return prompt_library


def production() -> ManagedPrompt[TextTemplate]:
    return get_prompt(
        USE_CASE,
        app=APP,
        default=TextTemplate(template="Summarize {{text}}."),
        environment="production",
    )


# Selection


def test_the_backend_is_opik_unless_the_environment_names_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(experiment_backend(), OpikExperimentBackend)

    configure_experiment_backend(None)
    monkeypatch.setenv("OL_LLM_EXPERIMENT_BACKEND", "memory")

    assert isinstance(experiment_backend(), MemoryExperimentBackend)
    assert experiment_backend() is experiment_backend()


def test_an_unknown_backend_name_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OL_LLM_EXPERIMENT_BACKEND", "nonesuch")

    with pytest.raises(ValueError, match="No experiment backend named 'nonesuch'"):
        experiment_backend()


def test_a_registered_backend_is_built_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[Mapping[str, str]] = []

    def factory(env: Mapping[str, str]) -> ExperimentBackend:
        built.append(env)
        return MemoryExperimentBackend()

    register_experiment_backend("custom", factory)
    monkeypatch.setenv("OL_LLM_EXPERIMENT_BACKEND", "custom")

    experiment_backend()

    [env] = built
    assert env["OL_LLM_EXPERIMENT_BACKEND"] == "custom"


# The memory backend


def test_seeding_is_idempotent_and_versions_what_it_adds(
    memory: MemoryExperimentBackend,
) -> None:
    assert memory.seed_dataset(DATASET, GOLDEN, app=APP) == "v1"
    assert memory.seed_dataset(DATASET, list(reversed(GOLDEN)), app=APP) == "v1"
    extra = {"text": "{x}", "expected": "{x}"}
    assert memory.seed_dataset(DATASET, [*GOLDEN, extra], app=APP) == "v2"

    assert memory.latest_dataset_version(DATASET, app=APP) == "v2"
    assert memory.missing_items(DATASET, [extra], app=APP, version="v1") == 1
    assert memory.missing_items(DATASET, [extra], app=APP, version="v2") == 0


def test_a_missing_dataset_or_version_raises_its_own_error(
    memory: MemoryExperimentBackend,
) -> None:
    with pytest.raises(DatasetNotFoundError):
        memory.latest_dataset_version(DATASET, app=APP)
    memory.seed_dataset(DATASET, GOLDEN, app=APP)

    with pytest.raises(DatasetVersionNotFoundError) as caught:
        memory.missing_items(DATASET, [], app=APP, version="v3")
    assert caught.value.latest == "v1"


class Several(Metric):
    """A metric that scores two things, and fails on one item."""

    def __init__(self) -> None:
        super().__init__("several", track=False)

    def score(self, text: str, output: str, **_: object) -> list[Score]:  # type: ignore[override]
        if "lectures" in text:
            msg = "cannot score"
            raise RuntimeError(msg)
        return [
            Score(name="length", value=float(len(output))),
            Score(name="echo", value=float(text in output)),
        ]


@pytest.mark.usefixtures("prompts")
def test_an_experiment_maps_inputs_and_fails_a_metric_that_raises(
    memory: MemoryExperimentBackend,
) -> None:
    memory.seed_dataset(DATASET, GOLDEN, app=APP)

    result = memory.run_experiment(
        ExperimentRun(
            task=lambda item: {"output": item["expected"]},
            metrics=[Several()],
            app=APP,
            dataset=DATASET,
            prompts=[production()],
            scoring_key_mapping={"text": "expected"},
            tags=["gate:x"],
        )
    )

    great, lectures = result.items
    assert {s.name: s.value for s in great.scores} == {"length": 7.0, "echo": 1.0}
    [failed] = lectures.scores
    assert (failed.name, failed.failed, failed.reason) == (
        "several",
        True,
        "RuntimeError",
    )
    # A read keeps only the scores that succeeded, as Opik does.
    assert memory.read_experiment(result.experiment_id).items[1].scores == ()
    [summary] = memory.find_experiments(app=APP, dataset=DATASET, tags=["gate:x"])
    assert summary.prompt_versions == {f"{USE_CASE}@v1"}
    assert summary.dataset_version == "v1"
    assert summary.means == {"length": 7.0, "echo": 1.0}


@pytest.mark.usefixtures("prompts")
def test_experiments_are_found_by_tag_newest_first(
    memory: MemoryExperimentBackend,
) -> None:
    memory.seed_dataset(DATASET, GOLDEN, app=APP)

    def run(*tags: str) -> str:
        return memory.run_experiment(
            ExperimentRun(
                task=lambda item: {"output": item["expected"]},
                metrics=[],
                app=APP,
                dataset=DATASET,
                prompts=[production()],
                tags=tags,
            )
        ).experiment_id

    first, _, third = run("a", "b"), run("a"), run("a")
    memory.tag_experiment(third, "b")

    found = memory.find_experiments(app=APP, dataset=DATASET, tags=["a", "b"])
    assert [s.experiment_id for s in found] == [third, first]
    assert memory.experiment_dataset_version(first) == (
        memory.experiment_dataset_version(third)
    )


def test_trace_scores_join_their_experiments_items(
    memory: MemoryExperimentBackend, prompts: PromptLibrary
) -> None:
    del prompts
    memory.seed_dataset(DATASET, GOLDEN[:1], app=APP)
    result = memory.run_experiment(
        ExperimentRun(
            task=lambda _item: {"output": "x"},
            metrics=[],
            app=APP,
            dataset=DATASET,
            prompts=[production()],
        )
    )
    [item] = result.items

    memory.log_scores(
        [TraceScore(trace_id=item.trace_id, name="pairwise", value=0.5)], app=APP
    )

    [read] = memory.read_experiment(result.experiment_id).items
    assert [(s.name, s.value) for s in read.scores] == [("pairwise", 0.5)]


def test_only_a_registry_version_links() -> None:
    backend = MemoryExperimentBackend()
    embedded = ManagedPrompt(
        name=USE_CASE,
        template=TextTemplate(template="Hi."),
        source="embedded",
        degraded_reasons=("embedded_prompt",),
    )

    assert backend.linked_version(embedded) is None


# The evaluation API on the memory backend


def kept(candidate: str) -> bool:
    return bool(candidate)


@pytest.mark.usefixtures("prompts")
def test_a_mapping_whose_field_is_missing_is_skipped(
    memory: MemoryExperimentBackend,
) -> None:
    memory.seed_dataset(DATASET, GOLDEN, app=APP)

    def filtered(_item: dict[str, Any]) -> dict[str, Any]:
        msg = "filtered"
        raise LLMContentFiltered(msg, use_case=USE_CASE, model="openai:x")

    result = evaluate(
        filtered,
        dataset=DATASET,
        app=APP,
        metrics=[ValidatorMetric("kept", kept)],
        prompts=[production()],
        scoring_key_mapping={"candidate": "summary"},
    )

    assert [(s.value, s.reason) for i in result.items for s in i.scores] == [
        (0.0, "the task raised LLMContentFiltered"),
    ] * len(GOLDEN)


@pytest.mark.usefixtures("prompts")
@pytest.mark.parametrize("judged", [False, True])
def test_a_missing_metric_input_aborts_the_run(
    memory: MemoryExperimentBackend,
    judged: bool,  # noqa: FBT001
) -> None:
    memory.seed_dataset(DATASET, GOLDEN, app=APP)
    metric = (
        hallucination(judge_model(APP, "openai:gpt-4o-mini"))
        if judged
        else ValidatorMetric("placeholders", lambda source, output: source == output)
    )

    with pytest.raises(MissingMetricInputError) as caught:
        evaluate(
            lambda _item: {"output": "x"},
            dataset=DATASET,
            app=APP,
            metrics=[metric],
            prompts=[production()],
        )
    assert caught.value.metric == metric.name


class NotScores(Metric):
    def __init__(self) -> None:
        super().__init__("not_scores", track=False)

    def score(self, **_: object) -> Score:  # type: ignore[override]
        return 1.0  # type: ignore[return-value]


@pytest.mark.usefixtures("prompts")
def test_a_metric_that_returns_no_score_fails_its_score(
    memory: MemoryExperimentBackend,
) -> None:
    memory.seed_dataset(DATASET, GOLDEN, app=APP)

    result = evaluate(
        lambda _item: {"output": "x"},
        dataset=DATASET,
        app=APP,
        metrics=[NotScores()],
        prompts=[production()],
    )

    assert {(s.failed, s.reason) for i in result.items for s in i.scores} == {
        (True, "TypeError")
    }


@pytest.mark.usefixtures("prompts")
def test_evaluate_and_check_run_on_the_configured_backend(
    memory: MemoryExperimentBackend,
) -> None:
    assert seed_dataset(DATASET, GOLDEN, app=APP) == "v1"

    result = evaluate(
        lambda item: {"output": item["expected"]},
        dataset=DATASET,
        app=APP,
        metrics=[ValidatorMetric("short", lambda output: len(output) < 9)],
        prompts=[production()],
    )

    stored = memory.experiments[result.experiment_id]
    assert stored.tags == {f"app:{APP}", "env:ci"}
    report = check_experiment(
        result.experiment_id, [Threshold(metric="short", minimum=1.0)]
    )
    [outcome] = report.outcomes
    assert (outcome.mean, outcome.scored) == (0.5, 2)


# The gate on the memory backend


def suite(context: GateContext) -> Suite:
    prompt = context.prompts[USE_CASE]

    def task(item: dict[str, Any]) -> dict[str, Any]:
        return {"output": f"{prompt.version}: {item['expected']}"}

    return Suite(
        task=task,
        metrics=[
            ValidatorMetric("brevity", lambda output: len(output) < 14, track=False)
        ],
    )


def declaration(**changes: Any) -> GateDeclaration:  # noqa: ANN401
    data = {
        "app": APP,
        "use_case": USE_CASE,
        "suite": "unused:suite",
        "prompts": {USE_CASE: "v2"},
        "dataset": {"name": DATASET, "version": "v1"},
        "thresholds": [{"metric": "brevity", "tolerance": 0.1, "direction": "higher"}],
    }
    return GateDeclaration.model_validate(data | changes)


@pytest.mark.usefixtures("prompts")
def test_a_gate_records_and_then_measures_from_its_baseline(
    memory: MemoryExperimentBackend,
) -> None:
    memory.seed_dataset(DATASET, GOLDEN, app=APP)

    recorded = run_gate(
        declaration(prompts={USE_CASE: "v1"}),
        suite=suite,
        change="main",
        record_baseline=True,
    )
    checked = run_gate(declaration(), suite=suite, change="pr-1")

    assert recorded.recorded_baseline
    assert BASELINE_TAG in memory.experiments[recorded.experiment_id].tags
    assert checked.baseline is not None
    assert checked.baseline.experiment_id == recorded.experiment_id
    [brevity] = checked.report.outcomes
    # "v1: {great}" and "v1: {lectures}" score 1.0 and 0.0, and so does v2.
    assert (brevity.mean, brevity.baseline, brevity.status) == (0.5, 0.5, "pass")


class IgnoresTags(MemoryExperimentBackend):
    """A backend whose search does not filter on tags."""

    def find_experiments(
        self, *, app: str, dataset: str, tags: Sequence[str]
    ) -> Iterator[ExperimentSummary]:
        del tags
        return super().find_experiments(app=app, dataset=dataset, tags=[])


@pytest.mark.usefixtures("prompts")
def test_a_pull_request_run_is_never_the_baseline() -> None:
    backend = IgnoresTags()
    configure_experiment_backend(backend)
    backend.seed_dataset(DATASET, GOLDEN, app=APP)
    run_gate(declaration(prompts={USE_CASE: "v1"}), suite=suite, change="pr-1")

    outcome = run_gate(declaration(), suite=suite, change="pr-2")

    assert outcome.baseline is None


@pytest.mark.usefixtures("prompts")
def test_a_gate_on_a_missing_dataset_says_how_to_seed_it(
    memory: MemoryExperimentBackend,
) -> None:
    del memory

    with pytest.raises(GateConfigurationError, match=f"--app {APP}"):
        run_gate(declaration(), suite=suite, change="pr-1")


@pytest.mark.usefixtures("prompts")
def test_a_pairwise_gate_judges_against_the_baseline(
    memory: MemoryExperimentBackend,
) -> None:
    memory.seed_dataset(DATASET, GOLDEN, app=APP)
    gate = declaration(
        judge_model="openai:gpt-4o-mini",
        pairwise={"criteria": "Prefer the answer from v2."},
        thresholds=[{"metric": PAIRWISE_METRIC, "minimum": 0.6}],
    )
    run_gate(
        gate.model_copy(update={"prompts": {USE_CASE: "v1"}}),
        suite=suite,
        change="main",
        record_baseline=True,
    )

    with override_models(preferring("v2")):
        outcome = run_gate(gate, suite=suite, change="pr-1")

    assert outcome.pairwise is not None
    assert outcome.pairwise.win_rate == 1.0
    assert outcome.report.passed
    logged = {score.name for _, score in memory.trace_scores}
    assert logged == {PAIRWISE_METRIC}


@pytest.mark.usefixtures("prompts")
def test_judge_pairwise_reads_both_experiments_from_the_backend(
    memory: MemoryExperimentBackend,
) -> None:
    memory.seed_dataset(DATASET, GOLDEN, app=APP)

    def run(version: str) -> str:
        return evaluate(
            lambda item: {"output": f"{version}: {item['expected']}"},
            dataset=DATASET,
            app=APP,
            metrics=[],
            prompts=[production()],
        ).experiment_id

    reference, candidate = run("v1"), run("v2")

    with override_models(preferring("v1")):
        comparison = judge_pairwise(
            candidate,
            reference,
            judge=judge_model(APP, "openai:gpt-4o-mini"),
            criteria="Prefer v1.",
            app=APP,
            threads=1,
        )

    assert comparison.win_rate == 0.0
    assert [v.trace_id for v in comparison.verdicts] == [
        item.trace_id for item in memory.experiments[candidate].result.items
    ]
