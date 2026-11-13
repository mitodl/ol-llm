"""An experiment backend in this process's memory, for tests and local runs.

Datasets, experiments and scores live as long as the backend object. A prompt
version links by its name and version, so any registry's fetched version links and
an embedded default does not.
"""

import itertools
import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

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
    MissingMetricInputError,
    Score,
    TraceScore,
)
from ol_llm.prompts import ManagedPrompt


def _content_key(item: Mapping[str, Any]) -> str:
    return json.dumps(item, sort_keys=True, default=str)


@dataclass
class _Dataset:
    id: str
    versions: list[dict[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def latest(self) -> str | None:
        return f"v{len(self.versions)}" if self.versions else None

    def version(self, name: str | None) -> dict[str, dict[str, Any]]:
        if name is None:
            return self.versions[-1] if self.versions else {}
        number = name.removeprefix("v")
        if not (number.isdigit() and 0 < int(number) <= len(self.versions)):
            msg = f"no version {name}"
            raise DatasetVersionNotFoundError(msg, latest=self.latest)
        return self.versions[int(number) - 1]


@dataclass
class _Experiment:
    result: ExperimentResult
    app: str
    dataset: _Dataset
    tags: set[str]
    prompt_versions: frozenset[str]
    dataset_version: str | None


class MemoryExperimentBackend(ExperimentBackend):
    """Datasets and experiments in memory; nothing leaves the process."""

    def __init__(self) -> None:
        """Start empty."""
        self._lock = threading.Lock()
        self.datasets: dict[tuple[str, str], _Dataset] = {}
        self.experiments: dict[str, _Experiment] = {}
        self.trace_scores: list[tuple[str, TraceScore]] = []
        self._order = itertools.count()
        self._sequence: dict[str, int] = {}

    def seed_dataset(
        self,
        name: str,
        items: Sequence[Mapping[str, Any]],
        *,
        app: str,
        description: str | None = None,
    ) -> str | None:
        """Add the items the dataset lacks, as a new version when there are any."""
        del description
        with self._lock:
            dataset = self.datasets.setdefault(
                (app, name), _Dataset(id=f"{app}/{name}")
            )
            current = dict(dataset.versions[-1]) if dataset.versions else {}
            new = {_content_key(item): dict(item) for item in items}
            if new.keys() - current.keys():
                dataset.versions.append(current | new)
            return dataset.latest

    def latest_dataset_version(self, name: str, *, app: str) -> str | None:
        """Return the latest version's name."""
        return self._dataset(name, app).latest

    def missing_items(
        self,
        name: str,
        items: Sequence[Mapping[str, Any]],
        *,
        app: str,
        version: str,
    ) -> int:
        """Count the items the version does not hold, by content."""
        held = self._dataset(name, app).version(version)
        return len({_content_key(item) for item in items} - held.keys())

    def linked_version(self, prompt: ManagedPrompt[Any]) -> str | None:
        """Link a fetched version as ``<name>@<version>``."""
        if prompt.source != "registry" or prompt.version is None:
            return None
        return f"{prompt.name}@{prompt.version}"

    def run_experiment(self, run: ExperimentRun) -> ExperimentResult:
        """Run the task on each item in turn, then each metric on its output."""
        dataset = self._dataset(run.dataset, run.app)
        version = run.dataset_version or dataset.latest
        rows = dataset.version(version)
        experiment_id = str(uuid.uuid4())
        items = []
        for key, row in rows.items():
            output = run.task(dict(row))
            inputs = {**row, **output}
            # As Opik does, a mapping whose field is missing is skipped, so a
            # metric that needs the input reports it missing.
            for target, source in run.scoring_key_mapping.items():
                if source in inputs:
                    inputs[target] = inputs[source]
            items.append(
                ExperimentItem(
                    dataset_item_id=key,
                    trace_id=str(uuid.uuid4()),
                    item=row,
                    output=output,
                    scores=tuple(
                        score
                        for metric in run.metrics
                        for score in _scored(metric, inputs)
                    ),
                )
            )
        result = ExperimentResult(
            experiment_id=experiment_id,
            name=run.name or experiment_id,
            url=None,
            items=tuple(items),
        )
        linked = (self.linked_version(prompt) for prompt in run.prompts)
        with self._lock:
            self._sequence[experiment_id] = next(self._order)
            self.experiments[experiment_id] = _Experiment(
                result=result,
                app=run.app,
                dataset=dataset,
                tags=set(run.tags),
                prompt_versions=frozenset(v for v in linked if v is not None),
                dataset_version=version,
            )
        return result

    def read_experiment(self, experiment_id: str) -> ExperimentResult:
        """Return the experiment with its successful scores, as Opik keeps them."""
        result = self.experiments[experiment_id].result
        if len(result.items) > MAX_EXPERIMENT_ITEMS:
            msg = (
                f"Experiment {experiment_id} has more than {MAX_EXPERIMENT_ITEMS} items"
            )
            raise ValueError(msg)
        kept = self._trace_scores()
        return replace(
            result,
            items=tuple(
                replace(
                    item,
                    scores=tuple(s for s in item.scores if not s.failed)
                    + kept.get(item.trace_id, ()),
                )
                for item in result.items
            ),
        )

    def experiment_dataset_version(self, experiment_id: str) -> str | None:
        """Return the dataset and version it ran on."""
        experiment = self.experiments[experiment_id]
        if experiment.dataset_version is None:
            return None
        return f"{experiment.dataset.id}@{experiment.dataset_version}"

    def log_scores(self, scores: Sequence[TraceScore], *, app: str) -> None:
        """Keep the scores; ``read_experiment`` and ``find_experiments`` see them."""
        with self._lock:
            self.trace_scores.extend((app, score) for score in scores)

    def tag_experiment(self, experiment_id: str, tag: str) -> None:
        """Add the tag."""
        with self._lock:
            self.experiments[experiment_id].tags.add(tag)

    def find_experiments(
        self, *, app: str, dataset: str, tags: Sequence[str]
    ) -> Iterator[ExperimentSummary]:
        """Yield matching experiments, the latest run first."""
        found = self._dataset(dataset, app)
        with self._lock:
            matching = sorted(
                (
                    (experiment_id, experiment)
                    for experiment_id, experiment in self.experiments.items()
                    if experiment.dataset is found and set(tags) <= experiment.tags
                ),
                key=lambda pair: self._sequence[pair[0]],
                reverse=True,
            )
        for experiment_id, experiment in matching:
            yield ExperimentSummary(
                experiment_id=experiment_id,
                name=experiment.result.name,
                url=None,
                tags=frozenset(experiment.tags),
                prompt_versions=experiment.prompt_versions,
                dataset_version=experiment.dataset_version,
                means=_means(self.read_experiment(experiment_id)),
            )

    def _dataset(self, name: str, app: str) -> _Dataset:
        with self._lock:
            if (dataset := self.datasets.get((app, name))) is None:
                msg = f"App {app!r} has no dataset {name!r}"
                raise DatasetNotFoundError(msg)
            return dataset

    def _trace_scores(self) -> dict[str, tuple[Score, ...]]:
        kept: dict[str, tuple[Score, ...]] = {}
        with self._lock:
            for _, score in self.trace_scores:
                kept[score.trace_id] = (
                    *kept.get(score.trace_id, ()),
                    Score(name=score.name, value=score.value, reason=score.reason),
                )
        return kept


def _scored(metric: Metric, inputs: Mapping[str, Any]) -> list[Score]:
    try:
        result = metric.score(**inputs)
        scores = [result] if isinstance(result, Score) else list(result)
        if not all(isinstance(score, Score) for score in scores):
            msg = f"{metric.name} returned {result!r}, not scores"
            raise TypeError(msg)  # noqa: TRY301 - a failed score, as on Opik
    except MissingMetricInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - a metric's error is its failed score
        return [
            Score(name=metric.name, value=0.0, reason=type(exc).__name__, failed=True)
        ]
    return scores


def _means(result: ExperimentResult) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for item in result.items:
        for score in item.scores:
            values.setdefault(score.name, []).append(score.value)
    return {name: sum(found) / len(found) for name, found in values.items()}
