"""The experiment backend, chosen by name like memory backends (library spec §8.3).

``OL_LLM_EXPERIMENT_BACKEND`` names it: ``opik`` (the default) or ``memory``, which
keeps datasets and experiments in the process for tests. Another package adds a
backend with an entry point in the ``ol_llm.experiment_backends`` group, naming a
factory that takes the environment mapping and returns an ``ExperimentBackend``,
or in code with ``register_experiment_backend``. A host that builds its backend in
code passes it to ``configure_experiment_backend``.

The judge metrics (``hallucination`` and the others) are Opik's metric classes run
in this process on the judge model, whichever backend runs the experiment.
"""

import os
import threading
from collections.abc import Callable, Mapping

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
from ol_llm.experiments.memory import MemoryExperimentBackend
from ol_llm.experiments.opik_backend import (
    OpikExperimentBackend,
    OpikJudgeMetric,
    answer_relevance,
    context_precision,
    context_recall,
    hallucination,
)
from ol_llm.loops import register_at_fork
from ol_llm.plugins import NamedFactories

__all__ = [
    "ENTRY_POINT_GROUP",
    "MAX_EXPERIMENT_ITEMS",
    "BackendFactory",
    "DatasetNotFoundError",
    "DatasetVersionNotFoundError",
    "ExperimentBackend",
    "ExperimentItem",
    "ExperimentResult",
    "ExperimentRun",
    "ExperimentSummary",
    "MemoryExperimentBackend",
    "Metric",
    "OpikExperimentBackend",
    "OpikJudgeMetric",
    "Score",
    "Task",
    "TraceScore",
    "answer_relevance",
    "configure_experiment_backend",
    "context_precision",
    "context_recall",
    "experiment_backend",
    "hallucination",
    "register_experiment_backend",
]

ENTRY_POINT_GROUP = "ol_llm.experiment_backends"
_ENV_NAME = "OL_LLM_EXPERIMENT_BACKEND"
DEFAULT_BACKEND = "opik"

BackendFactory = Callable[[Mapping[str, str]], ExperimentBackend]
"""Builds a backend from the process environment."""

_backends: NamedFactories[BackendFactory] = NamedFactories(
    group=ENTRY_POINT_GROUP,
    kind="experiment backend",
    builtins={
        "opik": lambda _env: OpikExperimentBackend(),
        "memory": lambda _env: MemoryExperimentBackend(),
    },
)
_lock = threading.Lock()
_backend: ExperimentBackend | None = None


def _after_fork_in_child() -> None:
    # A parent thread may have held it at the fork, and won't release it here.
    global _lock  # noqa: PLW0603
    _lock = threading.Lock()


register_at_fork(after_in_child=_after_fork_in_child)


def register_experiment_backend(name: str, factory: BackendFactory) -> None:
    """Make a backend available by name, in place of any registered before.

    :param name: what ``OL_LLM_EXPERIMENT_BACKEND`` names it as.
    :param factory: builds the backend from the environment.
    """
    _backends.register(name, factory)


def configure_experiment_backend(backend: ExperimentBackend | None) -> None:
    """Set the backend for this process, in place of reading the environment.

    :param backend: the backend; ``None`` reads the environment again on next
        use.
    """
    global _backend  # noqa: PLW0603
    with _lock:
        _backend = backend


def experiment_backend() -> ExperimentBackend:
    """Return the configured backend, building it from the environment on first use.

    :returns: the backend.
    :rtype: ExperimentBackend
    :raises ValueError: if ``OL_LLM_EXPERIMENT_BACKEND`` names no backend.
    """
    global _backend  # noqa: PLW0603
    with _lock:
        if _backend is not None:
            return _backend
    env = dict(os.environ)
    built = _backends.get(env.get(_ENV_NAME) or DEFAULT_BACKEND)(env)
    with _lock:
        if _backend is None:
            _backend = built
        return _backend
