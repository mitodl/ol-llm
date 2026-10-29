"""Evaluation gates in CI (lifecycle spec §5.1): run one use case's declared gate.

A use case declares its gate in its repository, next to its prompt, as a TOML file
the owning team reviews like code (spec §5.1 item 3)::

    app = "dagster-ml"                  # the Opik project
    use_case = "feedback-summary"
    mode = "advisory"                   # until the owner trusts the thresholds
    suite = "feedback_summary_eval:suite"
    judge_model = "bedrock:us.anthropic.claude-sonnet-5"

    [prompts]                           # the version production runs, per prompt
    feedback-summary = "v4"

    [dataset]
    name = "feedback-summary-golden"
    version = "v2"                      # the pinned dataset version
    file = "feedback_summary_golden.json"   # optional: seeded before each run

    [[thresholds]]
    metric = "placeholders"
    minimum = 1.0

    [[thresholds]]
    metric = "answer_relevance_metric"
    tolerance = 0.05                    # at most 0.05 below the baseline
    direction = "higher"

    [pairwise]                          # optional: judge against the baseline
    criteria = "Prefer the summary that keeps every point the feedback makes."

    [[thresholds]]
    metric = "win_rate_vs_baseline"     # a tie counts as half a win
    minimum = 0.45

``suite`` names a callable, ``module:attribute``, that takes a ``GateContext`` and
returns a ``Suite``: the task under test and its metrics. The module is imported
with the declaration's directory first on ``sys.path``, so it can sit beside the
declaration or be part of the app's installed package.

``run_gate`` fetches the pinned prompt versions, checks the dataset and its pinned
version exist (seeding the declared file first), runs ``evaluate`` tagged with the
gate and the change under review, and checks the scores against the thresholds,
with each ``tolerance`` measured from the baseline. With ``[pairwise]`` declared,
the judge also compares each answer with the baseline's answer to the same item
(``judge_pairwise``), and ``win_rate_vs_baseline`` is the share it preferred,
counting a tie as half. Anything that keeps the gate
from running raises ``GateConfigurationError`` with what to do about it, in either
mode: a gate that cannot run must not pass.

The baseline is the newest experiment recorded with ``record_baseline`` (CI does
that on the default branch) for this gate, on the pinned dataset version, linked to
every prompt's ``production``-labeled version. The spec's "last experiment for the
production version" would otherwise include runs from pull requests that never
merged. With no baseline, only the fixed bounds apply, the pairwise win rate is not
compared, and the next default-branch run records one.

Needs the ``pydantic-ai`` extra, the provider extras for the task's and judge's
models, and opik installed whole (see ``ol_llm.evaluation``).
"""

import dataclasses
import json
import logging
import pkgutil
import sys
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import httpx
import opik
import pydantic
from opik import url_helpers
from opik.api_objects.dataset.dataset_item import DatasetItem
from opik.evaluation.metrics import BaseMetric
from opik.exceptions import DatasetVersionNotFound
from opik.rest_api.core import ApiError
from opik.rest_api.types import ExperimentPublic

from ol_llm import tracing
from ol_llm.evaluation import (
    LLM_ERROR_FIELD,
    PAIRWISE_METRIC,
    GateMode,
    GateReport,
    JudgeModel,
    MetricOutcome,
    PairwiseComparison,
    Threshold,
    evaluate,
    judge_model,
    judge_pairwise,
    load_items,
    score_report,
    seed_dataset,
)
from ol_llm.prompts import ManagedPrompt, fetch_prompt

if TYPE_CHECKING:
    from opik.evaluation.evaluation_result import EvaluationResult

log = logging.getLogger(__name__)

BASELINE_TAG = "gate-baseline"
"""Added to a gate's experiment once a ``record_baseline`` run finishes."""

PRODUCTION_LABEL = "production"
"""The prompt label whose version the baseline must have run (spec §2.3)."""

GATE_ERROR_STATUS = 2
"""What ``ol-llm-eval gate`` exits with when the gate cannot run."""

_PAGE_SIZE = 100

VersionName = Annotated[str, pydantic.StringConstraints(pattern=r"^v[1-9][0-9]*$")]
"""An Opik prompt or dataset version, e.g. ``v3``."""


class GateConfigurationError(Exception):
    """The gate cannot run as declared; the message says what to change."""


class DatasetPin(pydantic.BaseModel):
    """The golden dataset a gate runs on (spec §5.1 item 6).

    :param name: the dataset, in the app's Opik project.
    :param version: the pinned version; changing it is a reviewed change.
    :param file: golden items to seed before each run, relative to the
        declaration. Seeding is idempotent, so this adds only new items.
    :param group: the named list to read from ``file``.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    version: VersionName
    file: Path | None = None
    group: str | None = None


class PairwiseDeclaration(pydantic.BaseModel):
    """A gate's pairwise comparison with its baseline; see ``judge_pairwise``.

    :param criteria: what makes one answer better than another for this use
        case, as the judge is to apply it.
    :param answer_fields: the task output fields the judge compares.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    criteria: Annotated[
        str, pydantic.StringConstraints(strip_whitespace=True, min_length=1)
    ]
    answer_fields: Annotated[list[str], pydantic.Field(min_length=1)] = ["output"]


class GateDeclaration(pydantic.BaseModel):
    """One use case's evaluation gate, as its repository declares it.

    :param app: the app's Opik project, which holds its prompts and datasets.
    :param use_case: the use case the gate covers.
    :param mode: ``advisory`` reports a miss; ``blocking`` fails the check.
    :param suite: ``module:attribute`` of the callable that builds the ``Suite``.
    :param judge_model: the judge's model id, when a metric needs a judge. It
        should not be the model under test (spec §5.1 item 7).
    :param prompts: prompt name to the version under test: on a pull request
        that changes it, the new version; otherwise production's.
    :param dataset: the dataset and its pinned version.
    :param thresholds: what passing means, one per metric.
    :param pairwise: judge each answer against the baseline's; needs
        ``judge_model`` and a threshold on ``win_rate_vs_baseline``.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    app: str
    use_case: str
    mode: GateMode = "advisory"
    suite: Annotated[str, pydantic.StringConstraints(pattern=r"^[\w.]+:[\w.]+$")]
    judge_model: str | None = None
    prompts: Annotated[dict[str, VersionName], pydantic.Field(min_length=1)]
    dataset: DatasetPin
    thresholds: Annotated[list[Threshold], pydantic.Field(min_length=1)]
    pairwise: PairwiseDeclaration | None = None

    @pydantic.field_validator("thresholds")
    @classmethod
    def _one_threshold_per_metric(cls, thresholds: list[Threshold]) -> list[Threshold]:
        metrics = [t.metric for t in thresholds]
        if duplicated := sorted({m for m in metrics if metrics.count(m) > 1}):
            msg = f"Each metric takes one threshold; {duplicated} have more"
            raise ValueError(msg)
        return thresholds

    @pydantic.model_validator(mode="after")
    def _pairwise_is_judged_and_bounded(self) -> "GateDeclaration":
        bound = next((t for t in self.thresholds if t.metric == PAIRWISE_METRIC), None)
        if self.pairwise is None:
            if bound is not None:
                msg = f"A threshold on {PAIRWISE_METRIC} needs [pairwise] declared"
                raise ValueError(msg)
            return self
        if self.judge_model is None:
            msg = "[pairwise] needs judge_model set"
            raise ValueError(msg)
        if bound is None:
            msg = (
                f"[pairwise] needs a threshold on {PAIRWISE_METRIC}, e.g. "
                "minimum = 0.45"
            )
            raise ValueError(msg)
        if bound.tolerance is not None:
            # The baseline's own win rate was against an older baseline.
            msg = (
                f"{PAIRWISE_METRIC} is already relative to the baseline, so it takes "
                "a minimum or maximum, not a tolerance"
            )
            raise ValueError(msg)
        return self

    @property
    def scored_thresholds(self) -> list[Threshold]:
        """The thresholds on the experiment's own scores, without the pairwise one."""
        return [t for t in self.thresholds if t.metric != PAIRWISE_METRIC]

    @classmethod
    def load(cls, path: Path) -> "GateDeclaration":
        """Read a declaration from a TOML file.

        :param path: the file.
        :returns: the declaration, with ``dataset.file`` resolved against the
            file's directory.
        :rtype: GateDeclaration
        :raises pydantic.ValidationError: if the file does not declare a gate.
        :raises tomllib.TOMLDecodeError: if the file is not TOML.
        """
        declaration = cls.model_validate(
            tomllib.loads(path.read_text(encoding="utf-8"))
        )
        if declaration.dataset.file is None:
            return declaration
        pin = declaration.dataset.model_copy(
            update={"file": path.parent / declaration.dataset.file}
        )
        return declaration.model_copy(update={"dataset": pin})


@dataclass(frozen=True, kw_only=True)
class GateContext:
    """What a suite builder gets: the prompts under test and the judge.

    :param app: the app's Opik project.
    :param use_case: the use case under test.
    :param prompts: each declared prompt at the version under test.
    :param judge: the declared judge, built through ``judge_model``.
    """

    app: str
    use_case: str
    prompts: Mapping[str, ManagedPrompt[Any]]
    judge: JudgeModel | None = None

    def require_judge(self) -> JudgeModel:
        """Return the judge, for a suite whose metrics need one.

        :returns: the judge.
        :rtype: JudgeModel
        :raises GateConfigurationError: if the gate declares no ``judge_model``.
        """
        if self.judge is None:
            msg = (
                f"The {self.use_case} suite needs a judge; set judge_model in the gate"
            )
            raise GateConfigurationError(msg)
        return self.judge


@dataclass(frozen=True, kw_only=True)
class Suite:
    """The task under test and how its output is scored; see ``evaluate``.

    :param task: called with each dataset item's fields; returns the output.
    :param metrics: the metrics, e.g. ``ValidatorMetric`` and judge metrics on
        ``GateContext.require_judge()``.
    :param scoring_key_mapping: metric input name to item or output field.
    :param experiment_config: what describes the run, e.g. the model under test.
    :param task_threads: items run at once.
    """

    task: Callable[[dict[str, Any]], dict[str, Any]]
    metrics: Sequence[BaseMetric]
    scoring_key_mapping: Mapping[str, str] | None = None
    experiment_config: Mapping[str, Any] | None = None
    task_threads: int = 16


@dataclass(frozen=True, kw_only=True)
class Baseline:
    """The experiment a gate's tolerances are measured from.

    :param experiment_id: the experiment.
    :param name: its name.
    :param url: the experiment in Opik.
    :param prompt_versions: prompt name to the production version it ran.
    :param means: each metric's mean, as Opik averaged it.
    """

    experiment_id: str
    name: str | None
    url: str | None
    prompt_versions: Mapping[str, str]
    means: Mapping[str, float]


@dataclass(frozen=True, kw_only=True)
class GateOutcome:
    """A gate's result: the report, and what it ran against.

    :param declaration: the gate.
    :param report: the scores against the thresholds.
    :param experiment_id: the experiment the gate ran.
    :param dataset_version: the dataset version it ran on.
    :param prompt_versions: prompt name to the version it ran.
    :param baseline: the baseline, when one was found.
    :param baseline_note: why there is none, when there is none.
    :param recorded_baseline: whether this run is now a baseline.
    :param not_run: why the experiment evaluated nothing, when it did not: every
        task raised, a thresholded metric scored no item, or the pairwise judge
        judged no pair. The gate then cannot vouch for the change in either mode.
    :param pairwise: the comparison with the baseline, when the gate declares
        one and there was a baseline.
    """

    declaration: GateDeclaration
    report: GateReport
    experiment_id: str
    dataset_version: str
    prompt_versions: Mapping[str, str]
    baseline: Baseline | None
    baseline_note: str | None = None
    recorded_baseline: bool = False
    not_run: str | None = None
    pairwise: PairwiseComparison | None = None

    @property
    def failed(self) -> bool:
        """Whether the check fails: a blocking gate that missed a threshold."""
        return self.declaration.mode == "blocking" and not self.report.passed

    @property
    def exit_status(self) -> int:
        """What ``ol-llm-eval gate`` exits with for this outcome."""
        if self.not_run is not None:
            return GATE_ERROR_STATUS
        return 1 if self.failed else 0

    def describe(self) -> str:
        """Return the report as text, with what it ran against."""
        lines = [self.report.describe(), *self._context_lines()]
        if self.not_run is not None:
            lines.append(f"The gate could not run: {self.not_run}")
        return "\n".join(lines)

    def markdown(self) -> str:
        """Return the report as Markdown, for the pull request and job summary."""
        declaration, report = self.declaration, self.report
        missed = sum(o.failed for o in report.outcomes)
        uncompared = sum(o.status == "not compared" for o in report.outcomes)
        if self.not_run is not None:
            verdict = "could not run"
        elif missed:
            verdict = f"{missed} of {len(report.outcomes)} thresholds not met"
        elif uncompared:
            verdict = f"no threshold missed, {uncompared} not compared"
        else:
            verdict = "all thresholds met"
        experiment = (
            f"[{report.experiment}]({report.url})" if report.url else report.experiment
        )
        lines = [
            (
                f"### Evaluation gate `{declaration.use_case}`: {verdict} "
                f"({declaration.mode})"
            ),
            "",
            (
                f"Experiment {experiment} ran {_versions(self.prompt_versions)} on "
                f"dataset `{declaration.dataset.name}` {self.dataset_version}"
                + (
                    f", judged by `{declaration.judge_model}`."
                    if declaration.judge_model
                    else "."
                )
            ),
            "",
            "| Metric | Mean | Needs | Baseline | Scored | Result |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        if self.not_run is not None:
            lines[2:2] = [
                (
                    f"It evaluated nothing: {self.not_run} A gate that could not "
                    "run fails the check in either mode."
                ),
                "",
            ]
        for outcome in report.outcomes:
            mean = "none" if outcome.mean is None else f"{outcome.mean:.2f}"
            baseline = "" if outcome.baseline is None else f"{outcome.baseline:.2f}"
            total = outcome.scored + outcome.unscored
            result = outcome.status
            lines.append(
                f"| `{outcome.threshold.metric}` | {mean} | "
                f"{outcome.threshold.describe()} | {baseline} | "
                f"{outcome.scored} of {total} | {result} |"
            )
        lines.extend(["", *self._context_lines(markdown=True)])
        if uncompared:
            lines.append(
                "A threshold with only a tolerance, or on the pairwise win rate, is "
                "not compared until there is a baseline."
            )
        if missed and self.not_run is None and declaration.mode == "advisory":
            lines.append(
                "Advisory: this gate reports and does not fail the check. The use "
                'case\'s owner makes it blocking by setting `mode = "blocking"` in '
                "the gate."
            )
        return "\n".join(lines) + "\n"

    def _context_lines(self, *, markdown: bool = False) -> list[str]:
        lines = []
        if self.baseline is not None:
            name = self.baseline.name or self.baseline.experiment_id
            linked = (
                f"[{name}]({self.baseline.url})"
                if markdown and self.baseline.url
                else name
            )
            lines.append(
                f"Baseline: {linked}, which ran production "
                f"{_versions(self.baseline.prompt_versions)}."
            )
        elif self.baseline_note:
            lines.append(f"No baseline: {self.baseline_note}")
        if self.pairwise is not None:
            lines.append(
                f"Pairwise against the baseline: {self.pairwise.describe()} A tie "
                "counts as half a win."
            )
        if self.recorded_baseline:
            lines.append("This run is recorded as the baseline for later changes.")
        return lines


def _versions(versions: Mapping[str, str]) -> str:
    return ", ".join(f"`{name}` {version}" for name, version in versions.items())


def load_suite(spec: str, search_path: Path) -> Callable[[GateContext], Suite]:
    """Import a suite builder named ``module:attribute``.

    :param spec: the builder.
    :param search_path: put first on ``sys.path`` for the import, so a module
        beside the declaration is found.
    :returns: the builder.
    :rtype: Callable[[GateContext], Suite]
    :raises GateConfigurationError: if it cannot be imported or is not callable.
    """
    directory = str(search_path.resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        builder = pkgutil.resolve_name(spec)
    except (ImportError, AttributeError) as exc:
        msg = (
            f"Could not import the suite {spec!r} ({exc}). It is imported with "
            f"{directory} first on sys.path, in the environment the gate runs in."
        )
        raise GateConfigurationError(msg) from exc
    if not callable(builder):
        msg = f"The suite {spec!r} is not callable"
        raise GateConfigurationError(msg)
    return builder  # type: ignore[no-any-return]


def run_gate(
    declaration: GateDeclaration,
    *,
    suite: Callable[[GateContext], Suite],
    change: str,
    record_baseline: bool = False,
) -> GateOutcome:
    """Run a use case's gate: one experiment, checked against its thresholds.

    :param declaration: the gate.
    :param suite: builds the task and metrics; see ``load_suite``.
    :param change: what is under review, e.g. ``pr-123``; the experiment is
        tagged ``change:<change>``.
    :param record_baseline: make this run a baseline once it finishes, as CI does
        on the default branch, unless it evaluated nothing.
    :returns: the outcome. A blocking miss does not raise; see
        ``GateOutcome.failed``.
    :rtype: GateOutcome
    :raises GateConfigurationError: if tracing is off, the dataset, its pinned
        version, or a pinned prompt version does not exist, or the pairwise
        comparison cannot pair the baseline's answers with this run's.
    """
    if not tracing.enabled():
        msg = (
            "Tracing is off (OPIK_TRACK_DISABLE), so there is no Opik to run the "
            "gate against"
        )
        raise GateConfigurationError(msg)
    _check_dataset(declaration)
    prompts = {
        name: _pinned_prompt(declaration, name, version)
        for name, version in declaration.prompts.items()
    }
    judge = None
    if declaration.judge_model is not None:
        try:
            judge = judge_model(declaration.app, declaration.judge_model)
        except ValueError as exc:
            raise GateConfigurationError(str(exc)) from exc
    built = suite(
        GateContext(
            app=declaration.app,
            use_case=declaration.use_case,
            prompts=prompts,
            judge=judge,
        )
    )
    config = {
        **(built.experiment_config or {}),
        "gate": declaration.use_case,
        "gate_mode": declaration.mode,
        "change": change,
        "judge_model": declaration.judge_model,
    }
    result = evaluate(
        built.task,
        dataset=declaration.dataset.name,
        project_name=declaration.app,
        metrics=built.metrics,
        prompts=list(prompts.values()),
        dataset_version=declaration.dataset.version,
        experiment_name=f"{declaration.use_case} {change}",
        experiment_config=config,
        scoring_key_mapping=built.scoring_key_mapping,
        task_threads=built.task_threads,
        experiment_tags=[_gate_tag(declaration), f"change:{change}"],
    )
    baseline, note = find_baseline(
        declaration, dataset_id=result.dataset_id, exclude=result.experiment_id
    )
    pairwise = None
    if declaration.pairwise is not None and baseline is not None and judge is not None:
        try:
            pairwise = judge_pairwise(
                result,
                baseline.experiment_id,
                judge=judge,
                criteria=declaration.pairwise.criteria,
                project_name=declaration.app,
                answer_fields=declaration.pairwise.answer_fields,
                threads=built.task_threads,
            )
        except ValueError as exc:
            msg = f"Could not compare with the baseline: {exc}"
            raise GateConfigurationError(msg) from exc
    # A run that evaluated nothing must not become the baseline either.
    not_run = _not_run(
        result,
        score_report(result, declaration.scored_thresholds, mode=declaration.mode),
        pairwise,
    )
    recorded = record_baseline and not_run is None
    if recorded:
        tracing.opik_client().rest_client.experiments.update_experiment(
            result.experiment_id, tags_to_add=[BASELINE_TAG]
        )
    report = _with_pairwise(
        score_report(
            result,
            declaration.thresholds,
            mode=declaration.mode,
            baselines=None if baseline is None else baseline.means,
        ),
        pairwise,
    )
    return GateOutcome(
        declaration=declaration,
        report=report,
        experiment_id=result.experiment_id,
        dataset_version=declaration.dataset.version,
        prompt_versions={name: p.version or "" for name, p in prompts.items()},
        baseline=baseline,
        baseline_note=note,
        recorded_baseline=recorded,
        not_run=not_run,
        pairwise=pairwise,
    )


def _with_pairwise(
    report: GateReport, pairwise: PairwiseComparison | None
) -> GateReport:
    """Put the pairwise win rate in place of the experiment's own scores for it."""

    def outcome(scored: MetricOutcome) -> MetricOutcome:
        if scored.threshold.metric != PAIRWISE_METRIC:
            return scored
        if pairwise is None:
            return MetricOutcome(
                threshold=scored.threshold,
                mean=None,
                scored=0,
                unscored=0,
                compared=False,
                detail="no baseline",
            )
        return pairwise.outcome(scored.threshold)

    return dataclasses.replace(
        report, outcomes=tuple(outcome(o) for o in report.outcomes)
    )


def _not_run(
    result: "EvaluationResult",
    report: GateReport,
    pairwise: PairwiseComparison | None,
) -> str | None:
    """Say why an experiment evaluated nothing, or return ``None`` if it did."""
    items = result.test_results
    if not items:
        return "the dataset version has no items."
    errors = Counter(
        str(output[LLM_ERROR_FIELD])
        for item in items
        if LLM_ERROR_FIELD in (output := item.test_case.task_output or {})
    )
    if errors.total() == len(items):
        # As in a batch, every row failing the same way is a configuration
        # problem (credentials, model access, a gateway route), not the prompt.
        counts = ", ".join(f"{name} x{n}" for name, n in errors.most_common())
        return (
            f"the task raised an LLMError on every item ({counts}); the job log "
            "has each one."
        )
    if unscored := [o.threshold.metric for o in report.outcomes if not o.scored]:
        return (
            f"no item has a score for {unscored}: the metric failed on every item "
            "(e.g. its judge raised each time), or the suite produces no score of "
            "that name."
        )
    if pairwise is not None and pairwise.unjudged and not pairwise.verdicts:
        return f"the pairwise judge judged no pair. {pairwise.describe()}"
    return None


def _gate_tag(declaration: GateDeclaration) -> str:
    return f"gate:{declaration.use_case}"


def _check_dataset(declaration: GateDeclaration) -> None:
    """Seed the declared file, then check the pinned version holds its items."""
    pin, app = declaration.dataset, declaration.app
    items = None
    if pin.file is None:
        existing = _get_dataset(pin.name, app)
    else:
        try:
            items = load_items(pin.file, group=pin.group)
            existing = seed_dataset(pin.name, items, project_name=app)
        except (OSError, ValueError) as exc:
            msg = f"Could not seed dataset {pin.name!r} from {pin.file}: {exc}"
            raise GateConfigurationError(msg) from exc
    if existing is None:
        msg = (
            f"Opik project {app!r} has no dataset {pin.name!r}. Create it with "
            f"`ol-llm-eval seed <golden.json> --dataset {pin.name} --project {app}`, "
            "or set dataset.file in the gate to seed it on every run, then pin the "
            "version it reports."
        )
        raise GateConfigurationError(msg)
    latest = existing.get_current_version_name()
    try:
        pinned = existing.get_version_view(pin.version)
    except DatasetVersionNotFound as exc:
        msg = (
            f"Dataset {pin.name!r} has no version {pin.version}; its latest is "
            f"{latest}. Pin a version that exists."
        )
        raise GateConfigurationError(msg) from exc
    if items is None:
        return
    # Checked on every run, not only when seeding adds a version: a retry seeds
    # nothing new, and would otherwise evaluate a stale pin without the new items.
    held = {DatasetItem(**item).content_hash() for item in pinned.get_items()}
    if missing := {DatasetItem(**item).content_hash() for item in items} - held:
        msg = (
            f"Dataset {pin.name!r} {pin.version} lacks {len(missing)} of the items in "
            f"{pin.file}, and seeding put them in {latest}. Set dataset.version = "
            f'"{latest}" in the gate to evaluate them.'
        )
        raise GateConfigurationError(msg)


def _get_dataset(name: str, project_name: str) -> opik.Dataset | None:
    try:
        return tracing.opik_client().get_dataset(name, project_name=project_name)
    except ApiError as exc:
        if exc.status_code == httpx.codes.NOT_FOUND:
            return None
        raise


def _pinned_prompt(
    declaration: GateDeclaration, name: str, version: str
) -> ManagedPrompt[Any]:
    prompt = fetch_prompt(name, project_name=declaration.app, version=version)
    if prompt is None:
        msg = (
            f"Opik project {declaration.app!r} has no prompt {name!r} at {version}. "
            "Create the version in Opik, or point the gate at one that exists."
        )
        raise GateConfigurationError(msg)
    return prompt


def find_baseline(
    declaration: GateDeclaration, *, dataset_id: str, exclude: str | None = None
) -> tuple[Baseline | None, str | None]:
    """Find the experiment a gate's tolerances are measured from.

    The newest experiment for this gate tagged ``gate-baseline``, on the pinned
    dataset version, linked to every declared prompt's production-labeled
    version.

    :param declaration: the gate.
    :param dataset_id: the dataset's id.
    :param exclude: an experiment to pass over, e.g. the one just run.
    :returns: the baseline and ``None``, or ``None`` and why there is none.
    :rtype: tuple[Baseline | None, str | None]
    """
    production: dict[str, ManagedPrompt[Any]] = {}
    for name in declaration.prompts:
        prompt = fetch_prompt(
            name, project_name=declaration.app, environment=PRODUCTION_LABEL
        )
        if prompt is None or prompt.opik_prompt is None:
            return None, f"no version of `{name}` carries the production label."
        production[name] = prompt
    wanted = {
        p.opik_prompt.__internal_api__version_id__
        for p in production.values()
        if p.opik_prompt is not None
    }
    versions = _versions(
        {name: prompt.version or "" for name, prompt in production.items()}
    )
    for experiment in _baseline_experiments(dataset_id, declaration):
        if (
            experiment.id is not None
            and experiment.id != exclude
            and wanted
            <= {v.prompt_version_id for v in experiment.prompt_versions or []}
            and experiment.dataset_version_summary is not None
            and experiment.dataset_version_summary.version_name
            == declaration.dataset.version
        ):
            return (
                Baseline(
                    experiment_id=experiment.id,
                    name=experiment.name,
                    url=_experiment_url(dataset_id, experiment.id),
                    prompt_versions={n: p.version or "" for n, p in production.items()},
                    means={s.name: s.value for s in experiment.feedback_scores or []},
                ),
                None,
            )
    return None, (
        f"no default-branch run of this gate on dataset "
        f"{declaration.dataset.version} with production {versions} yet. The "
        "next one records it."
    )


def _baseline_experiments(
    dataset_id: str, declaration: GateDeclaration
) -> Iterator[ExperimentPublic]:
    """Yield the gate's baseline experiments on a dataset, newest first.

    Opik filters on both tags (``=`` on a list is membership) and returns
    experiments by id, descending, when no sorting is given; ids are UUIDv7, so
    that is newest first. Checked against the Opik backend at 2.2.81 and 2.2.103
    (``ExperimentDAO.FIND``, ``FilterQueryBuilder``).
    """
    tags = [_gate_tag(declaration), BASELINE_TAG]
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
            # Checked here too, so a backend that ignored the filter cannot hand
            # back a pull request's run as the baseline.
            if set(tags) <= set(experiment.tags or []):
                yield experiment
        if len(content) < _PAGE_SIZE:
            return
        page += 1


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


def gate_command(
    path: Path, *, change: str, record_baseline: bool, report: Path | None
) -> int:
    """Run the gate a file declares, for ``ol-llm-eval gate``.

    :param path: the declaration.
    :param change: what is under review; see ``run_gate``.
    :param record_baseline: see ``run_gate``.
    :param report: where to write the Markdown report, if anywhere. A gate that
        cannot run or crashes writes that, so the pull request shows it.
    :returns: 0 on a pass or an advisory miss, 1 on a blocking miss, and
        ``GATE_ERROR_STATUS`` when the gate cannot run or evaluated nothing.
    :rtype: int
    """
    try:
        declaration = GateDeclaration.load(path)
    except (OSError, tomllib.TOMLDecodeError, pydantic.ValidationError) as exc:
        return _cannot_run(path, exc, report)
    try:
        outcome = run_gate(
            declaration,
            suite=load_suite(declaration.suite, path.parent),
            change=change,
            record_baseline=record_baseline,
        )
    except GateConfigurationError as exc:
        return _cannot_run(path, exc, report)
    except Exception as exc:
        # Only the class: a provider's or Opik's message can quote the request,
        # and the report is posted on the pull request. The log has the rest.
        _write(
            report,
            f"### Evaluation gate `{path}` crashed\n\n"
            f"It raised `{type(exc).__name__}`; the job log has the traceback.\n",
        )
        raise
    sys.stdout.write(f"{outcome.describe()}\n")
    _write(report, outcome.markdown())
    return outcome.exit_status


def _cannot_run(path: Path, exc: Exception, report: Path | None) -> int:
    sys.stderr.write(f"The gate in {path} could not run: {exc}\n")
    _write(
        report,
        f"### Evaluation gate `{path}` could not run\n\n"
        f"```text\n{exc}\n```\n\n"
        "A gate that cannot run fails the check in either mode.\n",
    )
    return GATE_ERROR_STATUS


def _write(report: Path | None, text: str) -> None:
    if report is not None:
        report.write_text(text, encoding="utf-8")
