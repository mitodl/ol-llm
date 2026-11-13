"""What an experiment backend does for ol-llm (library spec §8.3 seam 4).

A backend holds an app's golden datasets as numbered versions, runs experiments on
them, and keeps each experiment's items, scores and tags. ol-llm's evaluation logic
(thresholds, baselines, the pairwise comparison, the gate) sits above this
interface in ``ol_llm.evaluation`` and ``ol_llm.gate``, so every backend gets it.

Metrics are ol-llm's: a ``Metric`` has a name and scores one item from its fields
and the task's output, returning ``Score`` values. A backend runs them as its own
metric type.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ol_llm.prompts import ManagedPrompt

MAX_EXPERIMENT_ITEMS = 10_000
"""The most items ``read_experiment`` returns; a larger experiment is refused."""


class DatasetNotFoundError(LookupError):
    """The app has no dataset of that name."""


class DatasetVersionNotFoundError(LookupError):
    """The dataset has no such version.

    :param latest: the dataset's latest version, for the message.
    """

    def __init__(self, message: str, *, latest: str | None) -> None:
        """Keep the latest version beside the message."""
        super().__init__(message)
        self.latest = latest


@dataclass(frozen=True, kw_only=True)
class Score:
    """One metric's score for one item.

    :param name: the score's name, which a threshold names.
    :param value: the score.
    :param reason: why, when the metric says.
    :param failed: whether scoring failed; a failed score is left out of the
        mean and counts as unscored.
    """

    name: str
    value: float
    reason: str | None = None
    failed: bool = False


class Metric(ABC):
    """A named scoring function.

    ``score`` takes the item's fields and the task output's fields by keyword,
    after the run's ``scoring_key_mapping``, and should accept and ignore any it
    does not use.

    :param name: the score's name.
    :param track: whether a backend that traces scoring records this metric's
        scoring as a span of the item's trace.
    """

    def __init__(self, name: str, *, track: bool = True) -> None:
        """Hold the name and whether scoring is traced."""
        self.name = name
        self.track = track

    @abstractmethod
    def score(self, **inputs: Any) -> Score | Sequence[Score]:  # noqa: ANN401
        """Score one item.

        :param inputs: the item's fields and the task output's.
        :returns: one score, or several for a metric that scores more than one
            thing.
        :rtype: Score | Sequence[Score]
        """


Task = Callable[[dict[str, Any]], dict[str, Any]]
"""Called with a dataset item's fields; returns the output the metrics score."""


@dataclass(frozen=True, kw_only=True)
class ExperimentRun:
    """An experiment to run on a dataset version.

    :param task: the task under test.
    :param metrics: how its output is scored.
    :param app: the app, whose namespace holds the dataset.
    :param dataset: the dataset's name.
    :param dataset_version: the version to run on; the latest when ``None``.
    :param prompts: every prompt the task uses, each a version the backend links
        (``ExperimentBackend.linked_version`` is not ``None``).
    :param name: the experiment's name; the backend names it when ``None``.
    :param config: what describes the run, e.g. the model under test.
    :param tags: the experiment's tags.
    :param scoring_key_mapping: metric input name to item or output field.
    :param threads: items run at once.
    """

    task: Task
    metrics: Sequence[Metric]
    app: str
    dataset: str
    dataset_version: str | None = None
    prompts: Sequence[ManagedPrompt[Any]]
    name: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    tags: Sequence[str] = ()
    scoring_key_mapping: Mapping[str, str] = field(default_factory=dict)
    threads: int = 16


@dataclass(frozen=True, kw_only=True)
class ExperimentItem:
    """One dataset item as an experiment answered and scored it.

    :param dataset_item_id: the dataset item.
    :param trace_id: the item's trace, which holds its scores.
    :param item: the dataset item's fields.
    :param output: the task's output.
    :param scores: its scores, failed ones included.
    """

    dataset_item_id: str
    trace_id: str
    item: Mapping[str, Any]
    output: Mapping[str, Any]
    scores: tuple[Score, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ExperimentResult:
    """An experiment and its items.

    :param experiment_id: the experiment.
    :param name: its name.
    :param url: where to see it, when the backend has a UI.
    :param items: one per dataset item answered.
    """

    experiment_id: str
    name: str | None
    url: str | None
    items: tuple[ExperimentItem, ...]


@dataclass(frozen=True, kw_only=True)
class ExperimentSummary:
    """An experiment as a search returns it, without its items.

    :param experiment_id: the experiment.
    :param name: its name.
    :param url: where to see it, when the backend has a UI.
    :param tags: its tags.
    :param prompt_versions: the linked ids (``linked_version``) of the prompt
        versions it ran.
    :param dataset_version: the dataset version it ran on.
    :param means: each metric's mean over its scored items.
    """

    experiment_id: str
    name: str | None
    url: str | None
    tags: frozenset[str]
    prompt_versions: frozenset[str]
    dataset_version: str | None
    means: Mapping[str, float]


@dataclass(frozen=True, kw_only=True)
class TraceScore:
    """A score to record on an item's trace after its experiment ran.

    :param trace_id: the trace.
    :param name: the score's name.
    :param value: the score.
    :param reason: why.
    :param category: a label for the value, e.g. ``win``.
    """

    trace_id: str
    name: str
    value: float
    reason: str | None = None
    category: str | None = None


class ExperimentBackend(ABC):
    """A store of golden datasets and the experiments run on them, per app."""

    def disabled_reason(self) -> str | None:
        """Say why the backend cannot run experiments, or return ``None``.

        :returns: the reason, e.g. that tracing is off.
        :rtype: str | None
        """
        return None

    @abstractmethod
    def seed_dataset(
        self,
        name: str,
        items: Sequence[Mapping[str, Any]],
        *,
        app: str,
        description: str | None = None,
    ) -> str | None:
        """Add golden items to a dataset, creating it if needed.

        Idempotent: an item the dataset already holds is not added again, and
        adding items makes a new version.

        :param name: the dataset.
        :param items: the items; an item's keys are its fields.
        :param app: the app, whose namespace holds the dataset.
        :param description: set on a dataset this creates.
        :returns: the dataset's latest version, ``None`` when it has none.
        :rtype: str | None
        :raises ValueError: if an item has a field the backend keeps for itself.
        """

    @abstractmethod
    def latest_dataset_version(self, name: str, *, app: str) -> str | None:
        """Return a dataset's latest version.

        :param name: the dataset.
        :param app: the app, whose namespace holds the dataset.
        :returns: the version, ``None`` when it has none.
        :rtype: str | None
        :raises DatasetNotFoundError: if there is no such dataset.
        """

    @abstractmethod
    def missing_items(
        self,
        name: str,
        items: Sequence[Mapping[str, Any]],
        *,
        app: str,
        version: str,
    ) -> int:
        """Count the items a dataset version does not hold.

        :param name: the dataset.
        :param items: the items to look for, as ``seed_dataset`` takes them.
        :param app: the app, whose namespace holds the dataset.
        :param version: the version.
        :returns: how many of ``items`` the version lacks.
        :rtype: int
        :raises DatasetNotFoundError: if there is no such dataset.
        :raises DatasetVersionNotFoundError: if it has no such version.
        """

    @abstractmethod
    def linked_version(self, prompt: ManagedPrompt[Any]) -> str | None:
        """Return the id by which an experiment links a prompt version.

        :param prompt: a fetched prompt.
        :returns: the id, or ``None`` for an embedded default or a version this
            backend cannot link.
        :rtype: str | None
        """

    @abstractmethod
    def run_experiment(self, run: ExperimentRun) -> ExperimentResult:
        """Run the task on each item of a dataset version and score it.

        A metric that raises is a failed score for that item. An exception from
        the task aborts the run.

        :param run: what to run.
        :returns: the experiment.
        :rtype: ExperimentResult
        :raises DatasetNotFoundError: if there is no such dataset.
        """

    @abstractmethod
    def read_experiment(self, experiment_id: str) -> ExperimentResult:
        """Read a finished experiment's items and scores.

        A backend that keeps only successful scores returns those.

        :param experiment_id: the experiment.
        :returns: the experiment.
        :rtype: ExperimentResult
        :raises ValueError: if it has more than ``MAX_EXPERIMENT_ITEMS`` items,
            rather than return part of them.
        """

    @abstractmethod
    def experiment_dataset_version(self, experiment_id: str) -> str | None:
        """Return an id for the dataset version an experiment ran on.

        Two experiments ran on the same dataset version when the ids are equal.

        :param experiment_id: the experiment.
        :returns: the id, or ``None`` when the backend does not know it.
        :rtype: str | None
        """

    @abstractmethod
    def log_scores(self, scores: Sequence[TraceScore], *, app: str) -> None:
        """Record scores on items' traces.

        :param scores: the scores.
        :param app: the app, whose namespace holds the traces.
        """

    @abstractmethod
    def tag_experiment(self, experiment_id: str, tag: str) -> None:
        """Add a tag to an experiment.

        :param experiment_id: the experiment.
        :param tag: the tag.
        """

    @abstractmethod
    def find_experiments(
        self, *, app: str, dataset: str, tags: Sequence[str]
    ) -> Iterator[ExperimentSummary]:
        """Yield a dataset's experiments that carry every tag, newest first.

        :param app: the app, whose namespace holds the dataset.
        :param dataset: the dataset's name.
        :param tags: the tags.
        :returns: the experiments.
        :rtype: Iterator[ExperimentSummary]
        """
