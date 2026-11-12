"""The prompt registry, chosen by name like memory backends (library spec §8.3).

``OL_LLM_PROMPT_REGISTRY`` names it: ``opik`` (the default), or ``file`` with
``OL_LLM_PROMPT_REGISTRY_PATH`` set to an app's exports directory. The file registry
serves each export under ``production``, or under the comma-separated labels in
``OL_LLM_PROMPT_REGISTRY_LABELS``; a test or a local run that fetches by the
deployment environment adds that environment there. Another package
adds a registry with an entry point in the ``ol_llm.prompt_registries`` group,
naming a factory that takes the environment mapping and returns a
``PromptRegistry``, or in code with ``register_prompt_registry``. A host that
builds its registry in code passes it to ``configure_prompt_registry``.
"""

import os
import threading
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from pathlib import Path

from ol_llm.loops import register_at_fork
from ol_llm.prompt_registry.files import PRODUCTION, FilePromptRegistry
from ol_llm.prompt_registry.interface import (
    PromptRegistry,
    RegistryUnavailableError,
    RegistryVersion,
)
from ol_llm.prompt_registry.opik_registry import OpikPromptRegistry

__all__ = [
    "ENTRY_POINT_GROUP",
    "FilePromptRegistry",
    "OpikPromptRegistry",
    "PromptRegistry",
    "RegistryFactory",
    "RegistryUnavailableError",
    "RegistryVersion",
    "configure_prompt_registry",
    "prompt_registry",
    "register_prompt_registry",
]

ENTRY_POINT_GROUP = "ol_llm.prompt_registries"
_ENV_NAME = "OL_LLM_PROMPT_REGISTRY"
_ENV_PATH = "OL_LLM_PROMPT_REGISTRY_PATH"
_ENV_LABELS = "OL_LLM_PROMPT_REGISTRY_LABELS"
DEFAULT_REGISTRY = "opik"

RegistryFactory = Callable[[Mapping[str, str]], PromptRegistry]
"""Builds a registry from the process environment."""


def _file(env: Mapping[str, str]) -> PromptRegistry:
    if not (path := env.get(_ENV_PATH)):
        msg = f"the file prompt registry needs {_ENV_PATH}, the exports directory"
        raise ValueError(msg)
    labels = env.get(_ENV_LABELS) or PRODUCTION
    return FilePromptRegistry(
        Path(path), labels=[label.strip() for label in labels.split(",")]
    )


_lock = threading.Lock()
_factories: dict[str, RegistryFactory] = {
    "opik": lambda _env: OpikPromptRegistry(),
    "file": _file,
}
_registry: PromptRegistry | None = None


def _after_fork_in_child() -> None:
    # A parent thread may have held it at the fork, and won't release it here.
    global _lock  # noqa: PLW0603
    _lock = threading.Lock()


register_at_fork(after_in_child=_after_fork_in_child)


def register_prompt_registry(name: str, factory: RegistryFactory) -> None:
    """Make a registry available by name, in place of any registered before.

    :param name: what ``OL_LLM_PROMPT_REGISTRY`` names it as.
    :param factory: builds the registry from the environment.
    """
    with _lock:
        _factories[name] = factory


def configure_prompt_registry(registry: PromptRegistry | None) -> None:
    """Set the registry for this process, in place of reading the environment.

    :param registry: the registry; ``None`` reads the environment again on next
        use.
    """
    global _registry  # noqa: PLW0603
    with _lock:
        _registry = registry


def prompt_registry() -> PromptRegistry:
    """Return the configured registry, building it from the environment on first use.

    :returns: the registry.
    :rtype: PromptRegistry
    :raises ValueError: if ``OL_LLM_PROMPT_REGISTRY`` names no registry, or the
        named one is missing its configuration.
    """
    global _registry  # noqa: PLW0603
    with _lock:
        if _registry is not None:
            return _registry
    env = dict(os.environ)
    built = _factory(env.get(_ENV_NAME) or DEFAULT_REGISTRY)(env)
    with _lock:
        if _registry is None:
            _registry = built
        return _registry


def _factory(name: str) -> RegistryFactory:
    with _lock:
        if (factory := _factories.get(name)) is not None:
            return factory
    found = entry_points(group=ENTRY_POINT_GROUP, name=name)
    if not found:
        with _lock:
            known = sorted(_factories)
        msg = f"No prompt registry named {name!r}; registered: {known}"
        raise ValueError(msg)
    [entry_point] = found
    loaded: RegistryFactory = entry_point.load()
    register_prompt_registry(name, loaded)
    return loaded
