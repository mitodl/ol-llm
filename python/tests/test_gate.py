import json
import re
import sys
import textwrap
import tomllib
import uuid
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ol_llm.gate as gate_module
import opik
import pydantic
import pytest
from conftest import PromptLibrary
from ol_llm import fetch_prompt, tracing
from ol_llm.errors import LLMError, LLMRejected
from ol_llm.evaluation import Threshold, ValidatorMetric, main
from ol_llm.gate import (
    BASELINE_TAG,
    GATE_ERROR_STATUS,
    GateConfigurationError,
    GateContext,
    GateDeclaration,
    Suite,
    load_suite,
    run_gate,
)
from ol_llm.settings import Settings
from ol_llm.tracing import TracingConfig
from opik.api_objects.dataset.dataset_item import DatasetItem
from opik.evaluation import test_case, test_result
from opik.evaluation.evaluation_result import EvaluationResult
from opik.exceptions import DatasetVersionNotFound
from opik.rest_api.core import ApiError
from opik.rest_api.types import (
    DatasetVersionSummaryPublic,
    ExperimentPublic,
    FeedbackScoreAveragePublic,
    PromptVersionLinkPublic,
)

APP = "dagster-ml"
USE_CASE = "feedback-summary"
DATASET = "feedback-summary-golden"
GOLDEN = [
    {"text": "The course was {great}.", "expected": "{great}"},
    {"text": "Too long; {lectures} dragged.", "expected": "{lectures}"},
]

GATE_TOML = f"""
app = "{APP}"
use_case = "{USE_CASE}"
suite = "unused:suite"

[prompts]
{USE_CASE} = "v2"

[dataset]
name = "{DATASET}"
version = "v1"

[[thresholds]]
metric = "placeholders"
minimum = 1.0

[[thresholds]]
metric = "brevity"
tolerance = 0.1
direction = "higher"
"""


class FakeDataset:
    """An Opik dataset whose versions hold every item inserted up to them."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.id = f"ds-{name}"
        self.versions: list[dict[str, dict[str, Any]]] = []

    def insert(self, items: list[dict[str, Any]], deduplication: bool = True) -> None:  # noqa: FBT001, FBT002
        assert deduplication
        current = dict(self.versions[-1]) if self.versions else {}
        new = {DatasetItem(**item).content_hash(): item for item in items}
        if any(key not in current for key in new):
            self.versions.append(current | new)

    def get_current_version_name(self) -> str | None:
        return f"v{len(self.versions)}" if self.versions else None

    def get_version_view(self, version_name: str) -> list[dict[str, Any]]:
        number = int(version_name.removeprefix("v"))
        if not 0 < number <= len(self.versions):
            raise DatasetVersionNotFound(version_name)
        return list(self.versions[number - 1].values())


class FakeExperiments:
    def __init__(self) -> None:
        self.stored: list[ExperimentPublic] = []
        self.updates: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[list[dict[str, str]]] = []

    def find_experiments(
        self, *, page: int, size: int, dataset_id: str, filters: str
    ) -> SimpleNamespace:
        """Filter on tags with ``=`` as Opik does, newest first as its id order is."""
        self.queries.append(json.loads(filters))
        tags = {f["value"] for f in json.loads(filters) if f["field"] == "tags"}
        assert all(f["operator"] == "=" for f in json.loads(filters))
        matching = sorted(
            (
                e
                for e in self.stored
                if e.dataset_id == dataset_id and tags <= set(e.tags or [])
            ),
            key=lambda e: e.created_at,
            reverse=True,
        )
        return SimpleNamespace(content=matching[(page - 1) * size : page * size])

    def update_experiment(self, experiment_id: str, **fields: Any) -> None:  # noqa: ANN401
        self.updates.append((experiment_id, fields))


class FakeClient:
    """Datasets and experiments; prompts go to the SDK's real client."""

    def __init__(self, real: Callable[[], opik.Opik]) -> None:
        self._real = real
        self.datasets: dict[tuple[str, str], FakeDataset] = {}
        self.rest_client = SimpleNamespace(experiments=FakeExperiments())

    def get_or_create_dataset(
        self, name: str, description: str | None, project_name: str
    ) -> FakeDataset:
        del description
        return self.datasets.setdefault((project_name, name), FakeDataset(name))

    def get_dataset(self, name: str, project_name: str) -> FakeDataset:
        if (project_name, name) not in self.datasets:
            raise ApiError(status_code=404, body={"errors": ["not found"]})
        return self.datasets[project_name, name]

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._real(), name)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient(tracing.opik_client)
    monkeypatch.setattr(tracing, "opik_client", lambda: fake)
    return fake


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """``opik.evaluate``, scoring each item with the task and metrics in turn."""
    calls: list[dict[str, Any]] = []

    def fake_evaluate(**kwargs: Any) -> EvaluationResult:  # noqa: ANN401
        calls.append(kwargs)
        return fake_result(kwargs, number=len(calls))

    monkeypatch.setattr(opik, "evaluate", fake_evaluate)
    return calls


def fake_result(kwargs: dict[str, Any], *, number: int = 1) -> EvaluationResult:
    """Score each item with the task and metrics ``opik.evaluate`` was given."""
    results = []
    for i, item in enumerate(kwargs["dataset"]):
        output = kwargs["task"](item)
        inputs = {**item, **output}
        results.append(
            test_result.TestResult(
                test_case=test_case.TestCase(
                    trace_id=f"t{i}", dataset_item_id=f"i{i}", task_output=output
                ),
                score_results=[m.score(**inputs) for m in kwargs["scoring_metrics"]],
                trial_id=0,
            )
        )
    return EvaluationResult(
        experiment_id=f"exp-{number}",
        dataset_id=f"ds-{DATASET}",
        experiment_name=kwargs["experiment_name"],
        test_results=results,
        experiment_url=f"https://opik.example/exp-{number}",
        trial_count=1,
    )


@pytest.fixture
def prompts(prompt_library: PromptLibrary) -> PromptLibrary:
    """v1 in production, v2 under review, v3 a later draft."""
    prompt_library.add(
        USE_CASE, "Summarize {{text}}.", project_name=APP, environments=("production",)
    )
    prompt_library.add(USE_CASE, "Summarize {{text}}, briefly.", project_name=APP)
    prompt_library.add(USE_CASE, "Summarize {{text}} in a word.", project_name=APP)
    return prompt_library


def braces(text: str) -> set[str]:
    return {part.split("}")[0] for part in text.split("{")[1:]}


def suite(context: GateContext) -> Suite:
    prompt = context.prompts[USE_CASE]

    def task(item: dict[str, Any]) -> dict[str, Any]:
        return {"output": f"{prompt.version}: {item['expected']}"}

    return Suite(
        task=task,
        metrics=[
            ValidatorMetric(
                "placeholders",
                lambda text, output: braces(text) == braces(output),
                track=False,
            ),
            ValidatorMetric("brevity", lambda output: len(output) < 14, track=False),
        ],
        experiment_config={"model": "anthropic:claude-haiku-4-5"},
    )


def declaration(**changes: Any) -> GateDeclaration:  # noqa: ANN401
    data = {
        "app": APP,
        "use_case": USE_CASE,
        "suite": "unused:suite",
        "prompts": {USE_CASE: "v2"},
        "dataset": {"name": DATASET, "version": "v1"},
        "thresholds": [
            {"metric": "placeholders", "minimum": 1.0},
            {"metric": "brevity", "tolerance": 0.1, "direction": "higher"},
        ],
    }
    return GateDeclaration.model_validate(data | changes)


def seeded(client: FakeClient) -> FakeDataset:
    dataset = client.get_or_create_dataset(DATASET, None, APP)
    dataset.insert(GOLDEN)
    return dataset


def experiment(  # noqa: PLR0913
    prompt_library: PromptLibrary,
    *,
    version: str = "v1",
    tags: tuple[str, ...] = (f"gate:{USE_CASE}", BASELINE_TAG),
    dataset_version: str = "v1",
    brevity: float = 1.0,
    age_hours: int = 1,
    name: str | None = None,
) -> ExperimentPublic:
    detail = next(
        v
        for v in prompt_library.prompts[APP, USE_CASE]
        if v["version_number"] == version
    )
    return ExperimentPublic(
        id=str(uuid.uuid4()),
        name=name,
        dataset_id=f"ds-{DATASET}",
        tags=list(tags),
        created_at=datetime.now(UTC) - timedelta(hours=age_hours),
        prompt_versions=[
            PromptVersionLinkPublic(
                prompt_version_id=detail["id"], prompt_id=detail["prompt_id"]
            )
        ],
        dataset_version_summary=DatasetVersionSummaryPublic(
            version_name=dataset_version
        ),
        feedback_scores=[
            FeedbackScoreAveragePublic(name="placeholders", value=1.0),
            FeedbackScoreAveragePublic(name="brevity", value=brevity),
        ],
    )


# The declaration


def test_the_documented_declarations_are_valid() -> None:
    doc = (WORKFLOW.parents[2] / "docs" / "evaluation-gate.md").read_text()
    [example] = re.findall(r"```toml\n(.*?)```", doc, flags=re.DOTALL)
    docstring = (gate_module.__doc__ or "").split("::\n", 1)[1].split("``suite``")[0]

    for text in (example, textwrap.dedent(docstring)):
        assert GateDeclaration.model_validate(tomllib.loads(text)).thresholds


def test_a_declaration_loads_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "gate.toml"
    path.write_text(GATE_TOML + "\n[dataset.extra]\n", encoding="utf-8")
    with pytest.raises(pydantic.ValidationError, match="extra"):
        GateDeclaration.load(path)

    path.write_text(
        GATE_TOML.replace('version = "v1"', 'version = "v1"\nfile = "golden.json"'),
        encoding="utf-8",
    )
    loaded = GateDeclaration.load(path)

    assert loaded.mode == "advisory"
    assert loaded.prompts == {USE_CASE: "v2"}
    assert loaded.dataset.file == tmp_path / "golden.json"
    assert loaded.thresholds == [
        Threshold(metric="placeholders", minimum=1.0),
        Threshold(metric="brevity", tolerance=0.1, direction="higher"),
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompts": {USE_CASE: "2"}}, "should match pattern"),
        ({"prompts": {}}, "at least 1 item"),
        ({"dataset": {"name": DATASET, "version": "latest"}}, "should match pattern"),
        ({"thresholds": []}, "at least 1 item"),
        ({"thresholds": [{"metric": "brevity"}]}, "a minimum, a maximum, or a"),
        (
            {"thresholds": [{"metric": "hallucination_metric", "tolerance": 0.05}]},
            'needs direction = "higher" or "lower"',
        ),
        (
            {
                "thresholds": [
                    {"metric": "b", "minimum": 1},
                    {"metric": "b", "maximum": 1},
                ]
            },
            "one threshold",
        ),
        ({"mode": "strict"}, "'advisory' or 'blocking'"),
        ({"suite": "no_attribute"}, "should match pattern"),
    ],
)
def test_a_declaration_refuses_what_cannot_gate(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        declaration(**changes)


# Running


@pytest.mark.usefixtures("prompts")
def test_a_gate_runs_the_pinned_versions_and_tags_the_change(
    client: FakeClient, runs: list[dict[str, Any]]
) -> None:
    seeded(client)
    seeded(client).insert([{"text": "{later}", "expected": "{later}"}])

    outcome = run_gate(declaration(), suite=suite, change="pr-12")

    [call] = runs
    assert [p.version for p in call["prompts"]] == ["v2"]
    assert len(call["dataset"]) == len(GOLDEN)
    assert call["experiment_name"] == f"{USE_CASE} pr-12"
    assert call["experiment_tags"] == [
        f"app:{APP}",
        "env:ci",
        f"gate:{USE_CASE}",
        "change:pr-12",
    ]
    assert call["experiment_config"] == {
        "environment": "ci",
        "model": "anthropic:claude-haiku-4-5",
        "gate": USE_CASE,
        "gate_mode": "advisory",
        "change": "pr-12",
        "judge_model": None,
    }
    assert outcome.report.passed
    assert not outcome.failed
    assert outcome.exit_status == 0
    assert outcome.prompt_versions == {USE_CASE: "v2"}
    assert outcome.baseline is None
    assert "no default-branch run" in (outcome.baseline_note or "")
    assert not client.rest_client.experiments.updates
    # brevity has only a tolerance, so with no baseline its 0.5 is not compared.
    _, brevity = outcome.report.outcomes
    assert brevity.status == "not compared"
    markdown = outcome.markdown()
    assert "no threshold missed, 1 not compared (advisory)" in markdown
    assert "| 2 of 2 | not compared |" in markdown


@pytest.mark.usefixtures("prompts")
def test_a_missing_dataset_fails_with_how_to_create_it(
    client: FakeClient, runs: list[dict[str, Any]]
) -> None:
    del client
    with pytest.raises(GateConfigurationError, match="ol-llm-eval seed"):
        run_gate(declaration(), suite=suite, change="pr-12")
    assert not runs


@pytest.mark.usefixtures("prompts")
def test_a_missing_dataset_version_fails(
    client: FakeClient, runs: list[dict[str, Any]]
) -> None:
    seeded(client)
    pin = {"name": DATASET, "version": "v4"}

    with pytest.raises(GateConfigurationError, match="no version v4; its latest is v1"):
        run_gate(declaration(dataset=pin), suite=suite, change="pr-12")
    assert not runs


@pytest.mark.usefixtures("client", "prompts")
def test_a_declared_file_seeds_the_dataset(
    tmp_path: Path, runs: list[dict[str, Any]]
) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(GOLDEN), encoding="utf-8")
    pin = {"name": DATASET, "version": "v1", "file": str(golden)}

    run_gate(declaration(dataset=pin), suite=suite, change="pr-12")
    run_gate(declaration(dataset=pin), suite=suite, change="pr-12")

    assert [len(call["dataset"]) for call in runs] == [2, 2]


@pytest.mark.usefixtures("prompts")
def test_seeding_new_items_under_an_old_pin_fails(
    client: FakeClient, tmp_path: Path, runs: list[dict[str, Any]]
) -> None:
    seeded(client)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([*GOLDEN, {"text": "{x}", "expected": "{x}"}]))
    pin = {"name": DATASET, "version": "v1", "file": str(golden)}

    with pytest.raises(GateConfigurationError, match=r'dataset\.version = "v2"'):
        run_gate(declaration(dataset=pin), suite=suite, change="pr-12")
    assert not runs


@pytest.mark.usefixtures("prompts")
def test_a_missing_prompt_version_fails(
    client: FakeClient, runs: list[dict[str, Any]]
) -> None:
    seeded(client)

    with pytest.raises(
        GateConfigurationError, match="no prompt 'feedback-summary' at v9"
    ):
        run_gate(declaration(prompts={USE_CASE: "v9"}), suite=suite, change="pr-12")
    assert not runs


@pytest.mark.usefixtures("prompts")
def test_a_suite_that_needs_a_judge_says_so(client: FakeClient) -> None:
    seeded(client)

    def judged(context: GateContext) -> Suite:
        context.require_judge()
        raise AssertionError

    with pytest.raises(GateConfigurationError, match="set judge_model"):
        run_gate(declaration(), suite=judged, change="pr-12")


@pytest.mark.usefixtures("prompts")
def test_the_judge_is_built_through_the_library(client: FakeClient) -> None:
    seeded(client)
    judges = []

    def judged(context: GateContext) -> Suite:
        judges.append(context.require_judge())
        msg = "stop"
        raise GateConfigurationError(msg)

    with pytest.raises(GateConfigurationError, match="stop"):
        run_gate(
            declaration(judge_model="openai:gpt-4o-mini"), suite=judged, change="pr-1"
        )

    [judge] = judges
    assert (judge.use_case.app, judge.use_case.name) == (APP, "eval-judge")


@pytest.mark.usefixtures("prompts")
def test_an_uncataloged_judge_cannot_run(client: FakeClient) -> None:
    seeded(client)

    with pytest.raises(GateConfigurationError, match="not in the model catalog"):
        run_gate(
            declaration(judge_model="openai:gpt-9-judge"), suite=suite, change="pr-1"
        )


@pytest.mark.usefixtures("client", "prompts")
@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "No such file"), ('{"a": []}', "pick one with group")],
)
def test_a_golden_file_that_cannot_be_read_cannot_run(
    tmp_path: Path, content: str | None, message: str
) -> None:
    golden = tmp_path / "golden.json"
    if content is not None:
        golden.write_text(content)
    pin = {"name": DATASET, "version": "v1", "file": str(golden)}

    with pytest.raises(GateConfigurationError, match=message):
        run_gate(declaration(dataset=pin), suite=suite, change="pr-1")


def failing_suite(error: Callable[[str], LLMError]) -> Callable[[GateContext], Suite]:
    def build(context: GateContext) -> Suite:
        built = suite(context)

        def task(item: dict[str, Any]) -> dict[str, Any]:
            raise error(item["text"])

        return Suite(task=task, metrics=built.metrics)

    return build


@pytest.mark.usefixtures("prompts", "runs")
@pytest.mark.parametrize("mode", ["advisory", "blocking"])
def test_a_gate_whose_every_task_raised_could_not_run(
    client: FakeClient, mode: str
) -> None:
    seeded(client)

    # evaluate's guard turns each LLMError into an output naming it.
    outcome = run_gate(
        declaration(mode=mode),
        suite=failing_suite(
            lambda text: LLMRejected(
                text, use_case=USE_CASE, model="bedrock:m", status_code=401
            )
        ),
        change="main",
        record_baseline=True,
    )

    assert outcome.not_run == (
        "the task raised an LLMError on every item (LLMRejected x2); the job log "
        "has each one."
    )
    assert outcome.exit_status == GATE_ERROR_STATUS
    assert not client.rest_client.experiments.updates
    assert not outcome.recorded_baseline
    assert outcome.markdown().startswith(
        f"### Evaluation gate `{USE_CASE}`: could not run ({mode})"
    )
    assert "Advisory" not in outcome.markdown()


@pytest.mark.usefixtures("prompts", "runs")
def test_a_metric_that_scored_no_item_could_not_run(client: FakeClient) -> None:
    seeded(client)

    def unscored(context: GateContext) -> Suite:
        built = suite(context)
        return Suite(task=built.task, metrics=built.metrics[:1])

    outcome = run_gate(declaration(), suite=unscored, change="pr-1")

    assert outcome.not_run is not None
    assert "no item has a score for ['brevity']" in outcome.not_run
    assert outcome.exit_status == GATE_ERROR_STATUS


# Baselines


@pytest.mark.usefixtures("runs")
def test_a_tolerance_is_measured_from_the_newest_baseline(
    client: FakeClient, prompts: PromptLibrary
) -> None:
    seeded(client)
    stored = client.rest_client.experiments.stored
    newest = experiment(prompts, brevity=1.0, age_hours=1, name="main abc123")
    stored.extend(
        [
            experiment(prompts, brevity=0.5, age_hours=5),
            newest,
            # Not baselines: a pull request's run, another gate, another prompt
            # version, another dataset version, all newer.
            experiment(
                prompts,
                tags=(f"gate:{USE_CASE}", "change:pr-11"),
                brevity=0.5,
                age_hours=0,
            ),
            experiment(
                prompts, tags=("gate:other", BASELINE_TAG), brevity=0.5, age_hours=0
            ),
            experiment(prompts, version="v3", brevity=0.5, age_hours=0),
            experiment(prompts, dataset_version="v2", brevity=0.5, age_hours=0),
        ]
    )

    outcome = run_gate(declaration(), suite=suite, change="pr-12")

    assert outcome.baseline is not None
    assert outcome.baseline.experiment_id == newest.id
    assert client.rest_client.experiments.queries == [
        [
            {"field": "tags", "operator": "=", "value": f"gate:{USE_CASE}"},
            {"field": "tags", "operator": "=", "value": BASELINE_TAG},
        ]
    ]
    assert outcome.baseline.prompt_versions == {USE_CASE: "v1"}
    # "v2: {great}" is short enough; "v2: {lectures}" is not, so brevity is 0.5,
    # more than 0.1 below the baseline's 1.0.
    placeholders, brevity = outcome.report.outcomes
    assert placeholders.passed
    assert (brevity.mean, brevity.baseline, brevity.passed) == (0.5, 1.0, False)
    assert not outcome.failed
    markdown = outcome.markdown()
    assert "1 of 2 thresholds not met (advisory)" in markdown
    assert (
        "| `brevity` | 0.50 | at most 0.10 below baseline | 1.00 | 2 of 2 | fail |"
        in markdown
    )
    assert "Baseline: [main abc123](" in markdown
    assert "Advisory: this gate reports and does not fail the check" in markdown


@pytest.mark.usefixtures("runs")
def test_a_blocking_gate_fails_below_the_baseline(
    client: FakeClient, prompts: PromptLibrary
) -> None:
    seeded(client)
    client.rest_client.experiments.stored.append(experiment(prompts, brevity=1.0))

    outcome = run_gate(declaration(mode="blocking"), suite=suite, change="pr-12")

    assert outcome.failed
    assert "Advisory" not in outcome.markdown()


@pytest.mark.usefixtures("runs")
def test_no_production_label_means_no_baseline(
    client: FakeClient, prompt_library: PromptLibrary
) -> None:
    prompt_library.add(USE_CASE, "Summarize {{text}}.", project_name=APP)
    prompt_library.add(USE_CASE, "Summarize {{text}}, briefly.", project_name=APP)
    seeded(client)

    outcome = run_gate(declaration(), suite=suite, change="pr-12")

    assert outcome.baseline is None
    assert outcome.baseline_note == (
        "no version of `feedback-summary` carries the production label."
    )
    assert "No baseline: no version" in outcome.markdown()


@pytest.mark.usefixtures("prompts", "runs")
def test_a_default_branch_run_records_itself_as_the_baseline(
    client: FakeClient,
) -> None:
    seeded(client)

    outcome = run_gate(
        declaration(prompts={USE_CASE: "v1"}),
        suite=suite,
        change="main",
        record_baseline=True,
    )

    assert client.rest_client.experiments.updates == [
        ("exp-1", {"tags_to_add": [BASELINE_TAG]})
    ]
    assert outcome.recorded_baseline
    assert "recorded as the baseline" in outcome.markdown()


# The command


@pytest.fixture
def gate_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """A declaration with its suite module beside it."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    module = f"gate_suite_{uuid.uuid4().hex}"
    (tmp_path / f"{module}.py").write_text(
        "from test_gate import suite\n", encoding="utf-8"
    )
    path = tmp_path / "gate.toml"
    path.write_text(GATE_TOML.replace("unused:suite", f"{module}:suite"))
    yield path
    sys.modules.pop(module, None)


@pytest.mark.usefixtures("runs")
@pytest.mark.parametrize("case", [("advisory", 0), ("blocking", 1)])
def test_the_gate_command_fails_only_a_blocking_miss(
    client: FakeClient,
    prompts: PromptLibrary,
    gate_file: Path,
    capsys: pytest.CaptureFixture[str],
    case: tuple[str, int],
) -> None:
    mode, status = case
    seeded(client)
    client.rest_client.experiments.stored.append(experiment(prompts, brevity=1.0))
    gate_file.write_text(f'mode = "{mode}"\n' + gate_file.read_text())
    report = gate_file.with_name("report.md")

    argv = ["gate", str(gate_file), "--change", "pr-12", "--report", str(report)]

    assert main(argv) == status
    assert "FAIL brevity: mean 0.50, needs at most 0.10 below baseline" in (
        capsys.readouterr().out
    )
    assert report.read_text().startswith(
        f"### Evaluation gate `{USE_CASE}`: 1 of 2 thresholds not met ({mode})"
    )


@pytest.mark.usefixtures("client", "prompts", "runs")
@pytest.mark.parametrize("mode", ["advisory", "blocking"])
def test_the_gate_command_fails_a_gate_that_cannot_run(
    gate_file: Path, tmp_path: Path, mode: str
) -> None:
    gate_file.write_text(f'mode = "{mode}"\n' + gate_file.read_text())
    report = tmp_path / "report.md"

    argv = ["gate", str(gate_file), "--report", str(report)]

    assert main(argv) == GATE_ERROR_STATUS
    text = report.read_text()
    assert "could not run" in text
    assert f"has no dataset '{DATASET}'" in text


def test_the_gate_command_fails_a_malformed_declaration(tmp_path: Path) -> None:
    path = tmp_path / "gate.toml"
    path.write_text('app = "x"\n')
    report = tmp_path / "report.md"

    assert main(["gate", str(path), "--report", str(report)]) == GATE_ERROR_STATUS
    assert "use_case" in report.read_text()


@pytest.mark.usefixtures("prompts")
def test_the_gate_command_reports_a_crash_without_its_message(
    client: FakeClient, gate_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(client)

    def refuse(**_kwargs: Any) -> None:  # noqa: ANN401
        raise ApiError(status_code=401, body={"errors": ["secret request body"]})

    monkeypatch.setattr(opik, "evaluate", refuse)
    report = gate_file.with_name("report.md")

    with pytest.raises(ApiError):
        main(["gate", str(gate_file), "--report", str(report)])
    assert "raised `ApiError`" in report.read_text()
    assert "secret" not in report.read_text()


def test_a_suite_that_does_not_import_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(GateConfigurationError, match="Could not import the suite"):
        load_suite("no_such_module_here:suite", tmp_path)


# fetch_prompt


def test_fetch_prompt_reads_text_and_chat_prompts(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add(
        "text", "Hi {{name}}.", project_name=APP, environments=("production",)
    )
    prompt_library.add(
        "chat", [{"role": "user", "content": "Hi {{name}}."}], project_name=APP
    )

    text = fetch_prompt("text", project_name=APP, environment="production")
    chat = fetch_prompt("chat", project_name=APP, version="v1")

    assert text is not None
    assert (text.version, text.format(name="Ada")) == ("v1", "Hi Ada.")
    assert chat is not None
    assert chat.format(name="Ada") == [{"role": "user", "content": "Hi Ada."}]
    assert fetch_prompt("text", project_name=APP, version="v2") is None
    assert fetch_prompt("chat", project_name=APP, environment="production") is None


def test_fetch_prompt_takes_one_selector() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        fetch_prompt("text", project_name=APP)
    with pytest.raises(ValueError, match="exactly one"):
        fetch_prompt("text", project_name=APP, version="v1", environment="production")


# The workflow

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "evaluation-gate.yml"
ENV_LINE = re.compile(r"^ +([A-Z][A-Z0-9_]*): (\S+)", re.MULTILINE)


def test_the_workflow_sets_what_the_library_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate step's environment names the variables ol-llm reads, by name."""
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    step = WORKFLOW.read_text().split("- name: Run the gate\n", 1)[1]
    step = step.split("\n      - ", 1)[0]
    env = {name: "set" for name, _ in ENV_LINE.findall(step)}
    env["OL_LLM_ENVIRONMENT"] = dict(ENV_LINE.findall(step))["OL_LLM_ENVIRONMENT"]

    keycloak = TracingConfig.from_env(env).keycloak

    assert keycloak is not None
    assert keycloak.client_secret == "set"  # noqa: S105
    assert Settings.from_env(env).environment == "ci"
    assert {"OPIK_URL_OVERRIDE", "OPIK_WORKSPACE"} <= env.keys()
