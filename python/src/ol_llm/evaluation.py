"""Evaluation (lifecycle spec §4.2 and §5): datasets, experiments, gates.

A use case whose output a learner or instructor sees keeps a golden dataset and runs
an experiment on every prompt or model change. This module is the path for that,
on the experiment backend ``ol_llm.experiments`` configures (Opik by default):

- ``load_items`` and ``seed_dataset`` put a golden set from a local file into a
  dataset. Seeding again is a no-op for items already there, so a seed job can
  run on every change to the file. Each seed that adds items makes a new dataset
  version, which a gate pins.
- ``evaluate`` runs an experiment linked to the versions of the managed prompts
  the task uses. An ``LLMError`` from the task or a judge fails that item's
  scores instead of aborting the run.
- Metrics are ol-llm's (``Metric``, scoring an item to a ``Score``).
  ``ValidatorMetric`` turns a deterministic check (an SRT structure check, a
  placeholder check) into one. ``hallucination``, ``answer_relevance``,
  ``context_precision`` and ``context_recall`` are LLM-judge metrics on
  ``judge_model``, which calls through this library, so the judge is
  catalog-checked, routed, retried, and traced like any other call, under the
  ``eval-judge`` use case.
- ``judge_pairwise`` has the judge compare a candidate experiment's answers with
  a reference experiment's, item by item, and logs each verdict on the
  candidate's items, so its win rate can be thresholded like any other metric.
- ``check_thresholds`` compares an experiment's scores to the declared
  thresholds. Gates are advisory until a use case's owner makes them blocking
  (spec §5.1 item 4), so by default it reports and does not raise.

``ol-llm-eval seed`` and ``ol-llm-eval check`` do the seeding and the threshold
check from a CI job. Needs the ``pydantic-ai`` extra, and the provider extra for
the judge's model. With the Opik backend, run experiments where the Opik SDK is
installed whole (see ``ol_llm.experiments.opik_backend``).
"""

import argparse
import concurrent.futures
import contextvars
import functools
import inspect
import json
import logging
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pydantic

from ol_llm import tracing
from ol_llm.errors import LLMError
from ol_llm.experiments import (
    ExperimentItem,
    ExperimentResult,
    ExperimentRun,
    Metric,
    Score,
    Task,
    TraceScore,
    answer_relevance,
    context_precision,
    context_recall,
    experiment_backend,
    hallucination,
)
from ol_llm.judge import JUDGE_USE_CASE, JudgeModel, judge_model
from ol_llm.prompts import ManagedPrompt
from ol_llm.runs import complete_sync
from ol_llm.settings import get_settings

__all__ = [
    "JUDGE_USE_CASE",
    "LLM_ERROR_FIELD",
    "PAIRWISE_METRIC",
    "PAIRWISE_SCORES",
    "Direction",
    "EvaluationGateError",
    "ExperimentItem",
    "ExperimentResult",
    "GateMode",
    "GateReport",
    "JudgeModel",
    "Metric",
    "MetricOutcome",
    "OutcomeStatus",
    "PairVerdict",
    "PairwiseComparison",
    "PairwiseOutcome",
    "Score",
    "SkippedItem",
    "Threshold",
    "ValidatorMetric",
    "Verdict",
    "answer_relevance",
    "check_experiment",
    "check_thresholds",
    "context_precision",
    "context_recall",
    "evaluate",
    "hallucination",
    "judge_model",
    "judge_pairwise",
    "load_items",
    "main",
    "score_report",
    "seed_dataset",
]

log = logging.getLogger(__name__)

GateMode = Literal["advisory", "blocking"]
"""Advisory reports and does not fail; blocking fails the check (spec §5.1)."""

Direction = Literal["higher", "lower"]
"""Which way a metric's mean is better."""

LLM_ERROR_FIELD = "ol_llm_error"
"""The output field ``evaluate`` records, as the error's class name, for an item
whose task raised an ``LLMError``. It shows on the item's trace."""

_FLOAT_SLACK = 1e-9

Verdict = bool | Sequence[str]
"""What a validator returns: whether the output passed, or the problems it found,
where none means it passed."""


class ValidatorMetric(Metric):
    """A deterministic check as a metric, scored 1.0 on a pass and 0.0 not.

    The check takes the inputs it names from the dataset item and the task output
    (after ``scoring_key_mapping``) and returns a ``Verdict``: a bool, or the
    problems it found, which become the score's reason. A runtime validator is
    reused as it is, e.g. a placeholder check
    ``lambda source, output: placeholders(source) == placeholders(output)``.

    An exception raised by the check is a failed score, which
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
        super().__init__(name, track=track)
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

    def score(self, **inputs: Any) -> Score:  # noqa: ANN401
        """Run the check on one item.

        :param inputs: the item's fields and the task's output.
        :returns: 1.0 when the check passed, 0.0 with the problems when not.
        :rtype: Score
        :raises ValueError: if an input the check needs is missing.
        :raises TypeError: if the check returns a string, which reads as neither a
            pass nor a list of problems.
        """
        if (error := inputs.get(LLM_ERROR_FIELD)) is not None:
            return Score(name=self.name, value=0.0, reason=f"the task raised {error}")
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
        return Score(
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


def seed_dataset(
    name: str,
    items: Iterable[Mapping[str, Any]],
    *,
    app: str,
    description: str | None = None,
) -> str | None:
    """Add golden items to a dataset, creating it if needed.

    Idempotent: an item the dataset already holds is skipped, so seeding the same
    file again adds no items and no version; changing or adding items adds just
    those, as a new version. Items removed from the file stay in the dataset: it
    is curated in the backend (spec §5.1 item 6), and deleting there is a
    deliberate change.

    :param name: the dataset, e.g. ``<use_case>-golden``.
    :param items: the items; an item's keys are its fields.
    :param app: the app, whose namespace holds the dataset.
    :param description: set on a dataset this creates.
    :returns: the dataset's latest version, which a gate pins; ``None`` when it
        has none.
    :rtype: str | None
    :raises ValueError: if an item has a field the backend keeps for itself, e.g.
        ``id`` or ``description`` in Opik.
    """
    return experiment_backend().seed_dataset(
        name, [dict(item) for item in items], app=app, description=description
    )


def _guard_task(task: Task) -> Task:
    """Return the task with an ``LLMError`` turned into an output naming it.

    A backend aborts the whole run on any task exception, so one content filter
    or exhausted retry would leave a gate with nothing to report.
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


class _MetricOnLLMError(Metric):
    """A metric that reports a failed score for an item an ``LLMError`` cut short.

    Wraps every metric ``evaluate`` runs except ``ValidatorMetric``, which scores
    such an item 0.0 itself. A judge metric cannot do that: 0.0 is the passing
    end of some, e.g. ``hallucination``. So the item gets a failed score, which
    is left out of the mean, and ``check_thresholds`` counts as unscored. A
    judge's own ``LLMError`` is reported the same way, which also keeps Opik's
    rate-limit check, and its litellm import, off that path.

    :param metric: the metric.
    """

    def __init__(self, metric: Metric) -> None:
        """Take the metric's name and tracking."""
        super().__init__(metric.name, track=metric.track)
        self.metric = metric

    def score(self, **inputs: Any) -> Score | Sequence[Score]:  # noqa: ANN401
        """Score the item with the metric, or fail it after an ``LLMError``."""
        if (error := inputs.get(LLM_ERROR_FIELD)) is not None:
            return self._failed(f"the task raised {error}")
        try:
            return self.metric.score(**inputs)
        except LLMError as exc:
            return self._failed(f"the judge raised {type(exc).__name__}")

    def _failed(self, reason: str) -> Score:
        return Score(name=self.name, value=0.0, reason=reason, failed=True)


def _guard_metric(metric: Metric) -> Metric:
    if isinstance(metric, ValidatorMetric):
        return metric
    return _MetricOnLLMError(metric)


def evaluate(  # noqa: PLR0913
    task: Task,
    *,
    dataset: str,
    app: str,
    metrics: Sequence[Metric],
    prompts: Sequence[ManagedPrompt[Any]],
    dataset_version: str | None = None,
    experiment_name: str | None = None,
    experiment_config: Mapping[str, Any] | None = None,
    scoring_key_mapping: Mapping[str, str] | None = None,
    task_threads: int = 16,
    experiment_tags: Sequence[str] = (),
) -> ExperimentResult:
    """Run an experiment on a dataset, linked to the prompt versions under test.

    The backend records which prompt versions and which dataset version the
    experiment ran against. The experiment is tagged with the app and
    environment, and its config records the environment. Buffered traces and
    spans are sent before it returns, so a CI job can exit straight after.

    An ``LLMError`` from the task does not abort the run: the item's output
    becomes ``{"ol_llm_error": "<error class>"}``, a ``ValidatorMetric`` scores
    it 0.0, and every other metric reports a failed score, which
    ``check_thresholds`` counts as unscored. A judge metric's ``LLMError`` is a
    failed score too. Any other exception from the task still aborts the run.

    :param task: called with each dataset item's fields; returns the output the
        metrics score.
    :param dataset: the dataset's name.
    :param app: the app, whose namespace holds the dataset.
    :param metrics: the metrics, e.g. ``ValidatorMetric`` and judge metrics on
        ``judge_model``.
    :param prompts: every managed prompt the task uses, fetched from the prompt
        registry. For a prompt change, the version under review; for a model
        change, the production version.
    :param dataset_version: the dataset version to run against, e.g. ``v3``; the
        latest when omitted. A gate pins it (spec §5.1 item 6).
    :param experiment_name: the experiment's name; the backend names it when
        omitted.
    :param experiment_config: what describes the run, e.g. the model under test.
    :param scoring_key_mapping: metric input name to item or output field, e.g.
        ``{"input": "question"}``.
    :param task_threads: items run at once; 1 runs them in this thread.
    :param experiment_tags: tags beside the app and environment ones.
    :returns: the experiment, with its items and scores.
    :rtype: ExperimentResult
    :raises ValueError: if ``prompts`` is empty or holds an embedded default or
        a version the experiment backend cannot link.
    """
    if not prompts:
        msg = "Pass every managed prompt the task uses, so the experiment links them"
        raise ValueError(msg)
    backend = experiment_backend()
    if unlinked := [p.name for p in prompts if backend.linked_version(p) is None]:
        msg = (
            f"Prompts {unlinked} are not versions the experiment backend can link "
            "(embedded defaults, or from another prompt registry); an evaluation "
            "links the versions it ran"
        )
        raise ValueError(msg)
    environment = get_settings().environment
    result = backend.run_experiment(
        ExperimentRun(
            task=_guard_task(task),
            metrics=[_guard_metric(metric) for metric in metrics],
            app=app,
            dataset=dataset,
            dataset_version=dataset_version,
            prompts=prompts,
            name=experiment_name,
            config={"environment": environment, **(experiment_config or {})},
            tags=[f"app:{app}", f"env:{environment}", *experiment_tags],
            scoring_key_mapping=dict(scoring_key_mapping or {}),
            threads=task_threads,
        )
    )
    tracing.flush()
    return result


_THRESHOLD = re.compile(r"\s*(?P<metric>[^<>=\s]+)\s*(?P<op>>=|<=)\s*(?P<bound>\S+)\s*")


@dataclass(frozen=True, kw_only=True)
class Threshold:
    """A bound on one metric's mean over an experiment.

    A deterministic validator that must pass on every item (spec §5.1 item 5) is
    ``minimum=1.0``. A judge metric where higher is worse, such as
    ``hallucination``, takes a ``maximum``.

    ``tolerance`` bounds the mean against a baseline, the last default-branch
    experiment for the production prompt version (spec §5.1 item 5): the mean may
    fall at most that far below it, or for a metric where lower is better, rise at
    most that far above it. With no baseline the tolerance is not compared, and
    only the fixed bounds apply.

    :param metric: the score's name.
    :param minimum: the lowest mean that passes.
    :param maximum: the highest mean that passes.
    :param tolerance: how far the mean may move the wrong way from the baseline.
    :param direction: which way is better, for ``tolerance``. Required with a
        tolerance and no bound; otherwise ``lower`` for a threshold with only a
        ``maximum`` and ``higher`` for any other.
    :raises ValueError: if no bound or tolerance is set, a bound or the tolerance
        is not a finite number, the minimum is above the maximum, the tolerance is
        negative, or a tolerance with no bound has no direction.
    """

    metric: str
    minimum: float | None = None
    maximum: float | None = None
    tolerance: float | None = None
    direction: Direction | None = None

    def __post_init__(self) -> None:
        """Reject a threshold that bounds nothing or nothing could pass."""
        unbounded = self.minimum is None and self.maximum is None
        if unbounded and self.tolerance is None:
            msg = (
                f"Threshold for {self.metric!r} needs a minimum, a maximum, or a "
                "tolerance"
            )
            raise ValueError(msg)
        limits = (self.minimum, self.maximum, self.tolerance)
        if not all(math.isfinite(limit) for limit in limits if limit is not None):
            # A NaN bound compares false both ways, so a mean would never miss it.
            msg = (
                f"Threshold for {self.metric!r} has a bound or tolerance that is not "
                "a finite number"
            )
            raise ValueError(msg)
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            msg = f"Threshold for {self.metric!r} has its minimum above its maximum"
            raise ValueError(msg)
        if self.tolerance is not None and self.tolerance < 0:
            msg = f"Threshold for {self.metric!r} has a negative tolerance"
            raise ValueError(msg)
        if unbounded and self.direction is None:
            # Guessing would pass a regression on a metric where lower is better.
            msg = (
                f"Threshold for {self.metric!r} has a tolerance and no bound, so it "
                'needs direction = "higher" or "lower" (which way is better)'
            )
            raise ValueError(msg)

    @property
    def better(self) -> Direction:
        """Which way is better: ``direction``, or what the bounds imply."""
        if self.direction is not None:
            return self.direction
        if self.maximum is not None and self.minimum is None:
            return "lower"
        return "higher"

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
        if self.tolerance is not None:
            way = "below" if self.better == "higher" else "above"
            parts.append(f"at most {self.tolerance:.2f} {way} baseline")
        return " and ".join(parts)


OutcomeStatus = Literal["pass", "fail", "not compared"]
"""How a metric did: ``not compared`` is a tolerance-only threshold, or a pairwise
win rate, with no baseline, which neither passes nor fails."""


@dataclass(frozen=True, kw_only=True)
class MetricOutcome:
    """How one metric did against its threshold.

    :param threshold: the threshold.
    :param mean: the mean of the scored items, ``None`` when none was scored.
    :param scored: items with a score.
    :param unscored: items without one, because scoring failed or never ran.
    :param baseline: the baseline experiment's mean, when there is one and it
        scored the metric.
    :param compared: false for a metric that only exists against a baseline,
        such as a pairwise win rate, when there was none to compare with.
    :param detail: more about how the items scored, for the report line.
    """

    threshold: Threshold
    mean: float | None
    scored: int
    unscored: int
    baseline: float | None = None
    compared: bool = True
    detail: str | None = None

    @property
    def status(self) -> OutcomeStatus:
        """Whether the metric passed, failed, or could not be compared.

        ``not compared`` for a metric that was not ``compared`` or a
        tolerance-only threshold with no baseline; ``fail`` when an item is
        unscored, the mean is not a number, or it is outside a bound or the
        tolerance; else ``pass``.
        """
        if not self.compared:
            return "not compared"
        threshold, mean = self.threshold, self.mean
        if mean is None or math.isnan(mean) or self.unscored:
            return "fail"
        low, high = threshold.minimum, threshold.maximum
        if (low is not None and mean < low) or (high is not None and mean > high):
            return "fail"
        if threshold.tolerance is None:
            return "pass"
        if self.baseline is None:
            return "not compared" if low is None and high is None else "pass"
        # Within float error of the edge counts: 0.8 - 0.1 is 0.7000000000000001.
        allowed = threshold.tolerance + _FLOAT_SLACK
        within = (
            mean >= self.baseline - allowed
            if threshold.better == "higher"
            else mean <= self.baseline + allowed
        )
        return "pass" if within else "fail"

    @property
    def passed(self) -> bool:
        """Whether the metric met its threshold, bounds and baseline both."""
        return self.status == "pass"

    @property
    def failed(self) -> bool:
        """Whether the metric missed its threshold."""
        return self.status == "fail"

    def describe(self) -> str:
        """Return one report line."""
        status = self.status.upper()
        mean = "no scores" if self.mean is None else f"mean {self.mean:.2f}"
        counts = f"{self.scored} scored"
        if self.unscored:
            counts += f", {self.unscored} unscored"
        if self.threshold.tolerance is not None:
            counts += (
                ", no baseline"
                if self.baseline is None
                else f", baseline {self.baseline:.2f}"
            )
        if self.detail:
            counts += f", {self.detail}"
        return (
            f"{status} {self.threshold.metric}: {mean}, needs "
            f"{self.threshold.describe()} ({counts})"
        )


@dataclass(frozen=True, kw_only=True)
class GateReport:
    """An experiment's scores against a use case's thresholds.

    :param experiment: the experiment's name, or its id when it has none.
    :param url: the experiment in the backend's UI, when known.
    :param mode: whether a miss fails the check.
    :param outcomes: one per threshold, in the order given.
    """

    experiment: str
    url: str | None
    mode: GateMode
    outcomes: tuple[MetricOutcome, ...]

    @property
    def passed(self) -> bool:
        """Whether no threshold was missed; a ``not compared`` one is no miss."""
        return not any(outcome.failed for outcome in self.outcomes)

    def describe(self) -> str:
        """Return the report as text, one line per threshold."""
        heading = f"Experiment {self.experiment}" + (
            f" ({self.url})" if self.url else ""
        )
        missed = sum(outcome.failed for outcome in self.outcomes)
        uncompared = sum(o.status == "not compared" for o in self.outcomes)
        if missed and self.mode == "advisory":
            verdict = f"{missed} threshold(s) not met; advisory, so not failing."
        elif missed:
            verdict = f"{missed} threshold(s) not met."
        elif uncompared:
            verdict = (
                f"No threshold missed; {uncompared} not compared, for want of a "
                "baseline."
            )
        else:
            verdict = "All thresholds met."
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
    result: ExperimentResult,
    thresholds: Sequence[Threshold],
    *,
    mode: GateMode = "advisory",
    baselines: Mapping[str, float] | None = None,
) -> GateReport:
    """Compare an experiment's scores to its thresholds, and enforce the mode.

    ``score_report``, then a blocking miss raises and an advisory one is logged.

    :param result: what ``evaluate`` returned.
    :param thresholds: the use case's thresholds.
    :param mode: ``advisory`` logs a miss as a warning; ``blocking`` raises.
    :param baselines: the baseline experiment's mean per metric, which a
        threshold's ``tolerance`` is measured from.
    :returns: the report.
    :rtype: GateReport
    :raises EvaluationGateError: in blocking mode, if a threshold was not met.
    :raises ValueError: if an item has two scores of the same name.
    """
    return _enforce(score_report(result, thresholds, mode=mode, baselines=baselines))


def score_report(
    result: ExperimentResult,
    thresholds: Sequence[Threshold],
    *,
    mode: GateMode = "advisory",
    baselines: Mapping[str, float] | None = None,
) -> GateReport:
    """Compare an experiment's scores to its thresholds.

    A metric fails its threshold when any item has no score for it, because its
    scoring raised or the metric returned no score of that name, since a gate
    cannot vouch for items it did not score.

    :param result: what ``evaluate`` returned.
    :param thresholds: the use case's thresholds.
    :param mode: recorded on the report; nothing is raised or logged.
    :param baselines: the baseline experiment's mean per metric, which a
        threshold's ``tolerance`` is measured from.
    :returns: the report.
    :rtype: GateReport
    :raises ValueError: if an item has two scores of the same name.
    """
    return _report(
        experiment=result.name or result.experiment_id,
        url=result.url,
        items=_scores(result),
        thresholds=thresholds,
        mode=mode,
        baselines=baselines,
    )


def check_experiment(
    experiment_id: str,
    thresholds: Sequence[Threshold],
    *,
    mode: GateMode = "advisory",
) -> GateReport:
    """Compare a finished experiment's scores to its thresholds.

    For a CI step separate from the one that ran the experiment. Opik keeps only
    the scores that succeeded, so there an item whose scoring failed counts as
    unscored, as in ``check_thresholds``. A backend reads at most 10,000 items,
    so a larger experiment is refused rather than judged on part of it.

    :param experiment_id: the experiment's id, from ``ExperimentResult``.
    :param thresholds: the use case's thresholds.
    :param mode: ``advisory`` logs a miss as a warning; ``blocking`` raises.
    :returns: the report.
    :rtype: GateReport
    :raises EvaluationGateError: in blocking mode, if a threshold was not met.
    :raises ValueError: if the experiment has more than 10,000 items, or an item
        has two scores of the same name.
    """
    return _enforce(_experiment_report(experiment_id, thresholds, mode=mode))


def _experiment_report(
    experiment_id: str, thresholds: Sequence[Threshold], *, mode: GateMode
) -> GateReport:
    result = experiment_backend().read_experiment(experiment_id)
    return _report(
        experiment=result.name or experiment_id,
        url=None,
        items=_scores(result),
        thresholds=thresholds,
        mode=mode,
    )


def _scores(result: ExperimentResult) -> list[dict[str, float | None]]:
    """Return each item's scores by name, ``None`` for one that failed."""
    return [
        _item_scores(
            (score.name, None if score.failed else score.value) for score in item.scores
        )
        for item in result.items
    ]


def _item_scores(scores: Iterable[tuple[str, float | None]]) -> dict[str, float | None]:
    """Return one item's scores by name, ``None`` for one that failed.

    :raises ValueError: if two scores share a name, since then the item's score
        for that metric is ambiguous.
    """
    by_name: dict[str, float | None] = {}
    for name, value in scores:
        if name in by_name:
            msg = f"An item has two scores named {name!r}"
            raise ValueError(msg)
        by_name[name] = value
    return by_name


def _report(  # noqa: PLR0913
    *,
    experiment: str,
    url: str | None,
    items: Sequence[Mapping[str, float | None]],
    thresholds: Sequence[Threshold],
    mode: GateMode,
    baselines: Mapping[str, float] | None = None,
) -> GateReport:
    outcomes = []
    for threshold in thresholds:
        values = [
            value for item in items if (value := item.get(threshold.metric)) is not None
        ]
        outcomes.append(
            MetricOutcome(
                threshold=threshold,
                mean=sum(values) / len(values) if values else None,
                scored=len(values),
                unscored=len(items) - len(values),
                baseline=(baselines or {}).get(threshold.metric),
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


PAIRWISE_METRIC = "win_rate_vs_baseline"
"""The score ``judge_pairwise`` logs on each candidate item, and the metric a
threshold bounds the win rate with."""

PairwiseOutcome = Literal["win", "loss", "tie"]
"""How the candidate's answer did against the reference's on one item."""

PAIRWISE_SCORES: Mapping[PairwiseOutcome, float] = {"win": 1.0, "tie": 0.5, "loss": 0.0}
"""Each outcome's score. A tie is half a win, so the mean is the win rate."""

_PAIRWISE_INSTRUCTIONS = """\
You compare two answers to the same request and say which better meets the \
criteria below. The request is a dataset item, as JSON. Each answer is the output \
of a different version of the same system, also as JSON.

Judge only by the criteria. Which answer comes first is no reason to prefer it, \
and neither is length, unless the criteria ask for brevity. When neither answer \
meets the criteria better than the other, the verdict is "tie".

Give your reason in a few sentences, then the better answer: "A", "B", or "tie".

Criteria:
{criteria}"""


class _PairwiseChoice(pydantic.BaseModel):
    reason: str = pydantic.Field(description="Why, by the criteria.")
    better: Literal["A", "B", "tie"] = pydantic.Field(
        description='The answer that better meets the criteria, or "tie".'
    )


@dataclass(frozen=True, kw_only=True)
class PairVerdict:
    """The judge's verdict on one item's pair of answers.

    :param dataset_item_id: the dataset item both experiments answered.
    :param trace_id: the candidate's trace for the item, which holds the score.
    :param outcome: how the candidate's answer did.
    :param reason: the judge's reason in each answer order.
    :param orders_agreed: whether both orders gave the same verdict; when they
        did not, the outcome is a tie.
    """

    dataset_item_id: str
    trace_id: str
    outcome: PairwiseOutcome
    reason: str
    orders_agreed: bool


@dataclass(frozen=True, kw_only=True)
class SkippedItem:
    """An item ``judge_pairwise`` did not judge, and why.

    :param dataset_item_id: the dataset item.
    :param reason: e.g. ``only in the reference``.
    """

    dataset_item_id: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class PairwiseComparison:
    """A candidate experiment judged item by item against a reference experiment.

    The win rate counts a tie as half a win: it is the mean of each verdict's
    score (``PAIRWISE_SCORES``), and 0.5 means the candidate did as well as the
    reference.

    :param metric: the score's name.
    :param candidate: the candidate experiment's id.
    :param reference: the reference experiment's id.
    :param verdicts: one per judged pair, in the candidate's item order.
    :param unpaired: items with no pair to judge: in one experiment only, or
        with no answer from the reference (its task raised, or its output lacks
        an answer field). They do not count for or against the candidate, and
        when no item is paired the win rate is not compared.
    :param unjudged: pairs that could not be judged: the candidate's task or the
        judge raised an ``LLMError``. They count as unscored, which fails a
        threshold as for any other metric.
    """

    metric: str
    candidate: str
    reference: str
    verdicts: tuple[PairVerdict, ...]
    unpaired: tuple[SkippedItem, ...] = ()
    unjudged: tuple[SkippedItem, ...] = ()

    def count(self, outcome: PairwiseOutcome) -> int:
        """Return how many verdicts had an outcome."""
        return sum(v.outcome == outcome for v in self.verdicts)

    @property
    def disagreements(self) -> int:
        """How many ties came from the two answer orders disagreeing."""
        return sum(not v.orders_agreed for v in self.verdicts)

    @property
    def win_rate(self) -> float | None:
        """Wins plus half the ties, over the judged pairs; ``None`` with none."""
        if not self.verdicts:
            return None
        scores = [PAIRWISE_SCORES[v.outcome] for v in self.verdicts]
        return sum(scores) / len(scores)

    def outcome(self, threshold: Threshold) -> MetricOutcome:
        """Return the win rate against a threshold on ``metric``.

        ``not compared`` when no item had a pair, as with no reference at all.

        :param threshold: the threshold.
        :returns: the outcome, with the counts as its detail.
        :rtype: MetricOutcome
        """
        if not self.verdicts and not self.unjudged:
            return MetricOutcome(
                threshold=threshold,
                mean=None,
                scored=0,
                unscored=0,
                compared=False,
                detail=f"no item paired, {len(self.unpaired)} unpaired",
            )
        return MetricOutcome(
            threshold=threshold,
            mean=self.win_rate,
            scored=len(self.verdicts),
            unscored=len(self.unjudged),
            detail=(
                f"{self.count('win')} won, {self.count('tie')} tied, "
                f"{self.count('loss')} lost"
            ),
        )

    def describe(self) -> str:
        """Return the counts, and why any item was not judged, as one paragraph."""
        parts = [
            (
                f"Judged {len(self.verdicts)} pair(s) in both answer orders: "
                f"{self.count('win')} won, {self.count('tie')} tied "
                f"({self.disagreements} because the orders disagreed), "
                f"{self.count('loss')} lost."
            )
        ]
        for label, skipped in (
            ("not paired", self.unpaired),
            ("not judged", self.unjudged),
        ):
            if skipped:
                reasons = Counter(s.reason for s in skipped).most_common()
                counts = ", ".join(f"{reason} x{n}" for reason, n in reasons)
                parts.append(f"{len(skipped)} {label} ({counts}).")
        return " ".join(parts)


def judge_pairwise(  # noqa: PLR0913
    candidate: ExperimentResult | str,
    reference: str,
    *,
    judge: JudgeModel,
    criteria: str,
    app: str,
    answer_fields: Sequence[str] = ("output",),
    metric: str = PAIRWISE_METRIC,
    threads: int = 16,
) -> PairwiseComparison:
    """Judge a candidate experiment's answers against a reference experiment's.

    Pairs the two experiments' items by dataset item id; both must have run on
    the same dataset version. The judge sees the item and the two answers, and
    is asked twice, once with each answer first: when the two orders agree that
    is the verdict, and when they do not the pair is a tie, so a judge that
    favors a position cannot decide an item. Each verdict is logged as the
    ``metric`` score on the candidate's item, 1 for a win, 0.5 for a tie and 0
    for a loss, with the judge's reasons, so the backend shows it beside the
    experiment's other scores.

    The judge's calls go through ``complete_sync`` under the judge's use case
    (``eval-judge`` from ``judge_model``), so each is traced and costed there.

    :param candidate: the experiment under test: what ``evaluate`` returned, or
        its id.
    :param reference: the experiment to compare with, e.g. the gate's baseline.
    :param judge: the judge, from ``judge_model``.
    :param criteria: what makes one answer better than the other, for this use
        case.
    :param app: the app, whose namespace holds the experiments.
    :param answer_fields: the task output fields the judge compares; it sees
        the whole dataset item.
    :param metric: the score's name.
    :param threads: pairs judged at once.
    :returns: the verdicts, and the items that were not judged.
    :rtype: PairwiseComparison
    :raises ValueError: if the experiments did not run on the same dataset
        version, either has two items for one dataset item, or a candidate's
        answer lacks one of ``answer_fields``. A reference answer that lacks one
        leaves its item unpaired, so a gate whose answer fields changed can
        record a new baseline.
    """
    backend = experiment_backend()
    candidate_id = candidate if isinstance(candidate, str) else candidate.experiment_id
    _check_same_dataset_version(candidate_id, reference)
    if isinstance(candidate, str):
        candidate = backend.read_experiment(candidate)
    candidates = _keyed(candidate.items)
    references = _keyed(backend.read_experiment(reference).items)
    unpaired = [
        SkippedItem(dataset_item_id=item_id, reason="only in the reference")
        for item_id in references.keys() - candidates.keys()
    ]
    unjudged = []
    pairs = []
    for item_id, answered in candidates.items():
        # Checked before pairing, so answer fields that match neither experiment
        # raise rather than leave every item unpaired and record a new baseline.
        mine = (
            None
            if answered.output.get(LLM_ERROR_FIELD)
            else _answer(answered, answer_fields)
        )
        if (other := references.get(item_id)) is None:
            unpaired.append(
                SkippedItem(dataset_item_id=item_id, reason="only in the candidate")
            )
        elif error := other.output.get(LLM_ERROR_FIELD):
            unpaired.append(
                SkippedItem(
                    dataset_item_id=item_id,
                    reason=f"the reference's task raised {error}",
                )
            )
        elif missing := [f for f in answer_fields if f not in other.output]:
            # The reference predates the fields; a new reference will have them.
            unpaired.append(
                SkippedItem(
                    dataset_item_id=item_id,
                    reason=f"the reference's answer has no {missing}",
                )
            )
        elif mine is None:
            unjudged.append(
                SkippedItem(
                    dataset_item_id=item_id,
                    reason=(
                        "the candidate's task raised "
                        f"{answered.output[LLM_ERROR_FIELD]}"
                    ),
                )
            )
        else:
            pairs.append((answered, mine, {f: other.output[f] for f in answer_fields}))
    instructions = _PAIRWISE_INSTRUCTIONS.format(criteria=criteria)
    verdicts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        # Each call runs in a copy of this context, so model overrides and the
        # trace context set by the caller reach the judge's calls.
        futures = [
            pool.submit(
                contextvars.copy_context().run,
                _judge_pair,
                judge,
                instructions,
                answered,
                mine,
                theirs,
            )
            for answered, mine, theirs in pairs
        ]
        for (answered, _, _), future in zip(pairs, futures, strict=True):
            try:
                verdicts.append(future.result())
            except LLMError as exc:
                log.warning(
                    "The pairwise judge raised %s on item %s",
                    type(exc).__name__,
                    answered.dataset_item_id,
                )
                unjudged.append(
                    SkippedItem(
                        dataset_item_id=answered.dataset_item_id,
                        reason=f"the judge raised {type(exc).__name__}",
                    )
                )
    if verdicts:
        backend.log_scores(
            [
                TraceScore(
                    trace_id=verdict.trace_id,
                    name=metric,
                    value=PAIRWISE_SCORES[verdict.outcome],
                    reason=verdict.reason,
                    category=verdict.outcome,
                )
                for verdict in verdicts
            ],
            app=app,
        )
    tracing.flush()
    return PairwiseComparison(
        metric=metric,
        candidate=candidate_id,
        reference=reference,
        verdicts=tuple(verdicts),
        unpaired=tuple(unpaired),
        unjudged=tuple(unjudged),
    )


def _check_same_dataset_version(candidate: str, reference: str) -> None:
    backend = experiment_backend()
    versions = {backend.experiment_dataset_version(e) for e in (candidate, reference)}
    if len(versions) > 1 or None in versions:
        msg = (
            f"Experiments {candidate} and {reference} did not run on the same "
            "dataset version, so their items cannot be paired"
        )
        raise ValueError(msg)


def _keyed(answers: Iterable[ExperimentItem]) -> dict[str, ExperimentItem]:
    """Key answers by dataset item id.

    :raises ValueError: if two answer one item, as with ``trial_count`` above 1,
        since then which to pair is ambiguous.
    """
    keyed: dict[str, ExperimentItem] = {}
    for answered in answers:
        if answered.dataset_item_id in keyed:
            msg = (
                f"Dataset item {answered.dataset_item_id} has two answers in one "
                "experiment; a pairwise comparison takes one trial per item"
            )
            raise ValueError(msg)
        keyed[answered.dataset_item_id] = answered
    return keyed


def _answer(answered: ExperimentItem, fields: Sequence[str]) -> dict[str, Any]:
    if missing := [f for f in fields if f not in answered.output]:
        msg = (
            f"The candidate's answer to dataset item {answered.dataset_item_id} has no "
            f"{missing}; it has {sorted(answered.output)}. Name the fields to "
            "compare with answer_fields."
        )
        raise ValueError(msg)
    return {f: answered.output[f] for f in fields}


def _judge_pair(
    judge: JudgeModel,
    instructions: str,
    answered: ExperimentItem,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> PairVerdict:
    candidate_first = _choose(judge, instructions, answered.item, candidate, reference)
    reference_first = _choose(judge, instructions, answered.item, reference, candidate)
    first = {"A": "win", "B": "loss", "tie": "tie"}[candidate_first.better]
    second = {"A": "loss", "B": "win", "tie": "tie"}[reference_first.better]
    agreed = first == second
    reason = (
        f"Candidate as A: {candidate_first.reason}\n"
        f"Candidate as B: {reference_first.reason}"
    )
    if not agreed:
        reason = f"The two answer orders disagreed, so this is a tie.\n{reason}"
    return PairVerdict(
        dataset_item_id=answered.dataset_item_id,
        trace_id=answered.trace_id,
        outcome=cast("PairwiseOutcome", first if agreed else "tie"),
        reason=reason,
        orders_agreed=agreed,
    )


def _choose(
    judge: JudgeModel,
    instructions: str,
    item: Mapping[str, Any],
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> _PairwiseChoice:
    request = "\n\n".join(
        f"{label}:\n{json.dumps(value, ensure_ascii=False, indent=2, default=str)}"
        for label, value in (("Request", item), ("Answer A", a), ("Answer B", b))
    )
    return complete_sync(
        judge.use_case, request, output_type=_PairwiseChoice, instructions=instructions
    ).output


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``ol-llm-eval`` command.

    ``seed`` adds a file's golden items to a dataset and prints its version.
    ``check`` prints an experiment's scores against thresholds and exits 1 on a
    miss only with ``--mode blocking``. ``gate`` runs a declared gate (see
    ``ol_llm.gate``): it exits 1 on a blocking miss, and 2 when the gate cannot
    run, in either mode.

    :param argv: the arguments; ``sys.argv[1:]`` when omitted.
    :returns: the exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        prog="ol-llm-eval", description="Golden datasets, experiments and gates."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser("seed", help="add a file's golden items to a dataset")
    seed.add_argument("file", type=Path, help="a JSON list of items, or named lists")
    seed.add_argument("--dataset", required=True, help="the dataset's name")
    seed.add_argument("--app", required=True, help="the app, which owns the dataset")
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
    gate = commands.add_parser("gate", help="run a use case's declared gate")
    gate.add_argument("declaration", type=Path, help="the gate's TOML file")
    gate.add_argument(
        "--change",
        default="local",
        help="what is under review, e.g. pr-123; tags the experiment",
    )
    gate.add_argument(
        "--record-baseline",
        action="store_true",
        help="make this run the baseline for later changes (the default branch)",
    )
    gate.add_argument("--report", type=Path, help="write the report as Markdown here")
    args = parser.parse_args(argv)
    if args.command == "gate":
        # Imported here: ol_llm.gate imports this module.
        from ol_llm.gate import gate_command  # noqa: PLC0415

        return gate_command(
            args.declaration,
            change=args.change,
            record_baseline=args.record_baseline,
            report=args.report,
        )
    if args.command == "seed":
        version = seed_dataset(
            args.dataset, load_items(args.file, group=args.group), app=args.app
        )
        sys.stdout.write(f"{args.dataset} {version or 'unversioned'}\n")
        return 0
    report = _experiment_report(
        args.experiment_id, args.threshold, mode=cast("GateMode", args.mode)
    )
    sys.stdout.write(f"{report.describe()}\n")
    return 1 if report.mode == "blocking" and not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
