"""Factories chosen by name: the library's own, or another package's entry point.

Memory backends and prompt registries are both picked this way, so a deployment
names one in configuration and a package adds one without code changes in apps.
"""

import threading
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Generic, TypeVar

from ol_llm.loops import register_at_fork

FactoryT = TypeVar("FactoryT")


class NamedFactories(Generic[FactoryT]):
    """The factories registered under names, loading entry points on first use.

    :param group: the entry-point group other packages register under.
    :param kind: what a factory builds, for the error naming an unknown name,
        e.g. ``memory backend``.
    :param builtins: the library's own factories by name.
    """

    def __init__(
        self, *, group: str, kind: str, builtins: Mapping[str, FactoryT]
    ) -> None:
        """Hold ``builtins`` and resolve other names from ``group``."""
        self.group = group
        self.kind = kind
        self._factories = dict(builtins)
        self._lock = threading.Lock()
        register_at_fork(after_in_child=self._after_fork_in_child)

    def _after_fork_in_child(self) -> None:
        # A parent thread may have held it at the fork, and won't release it here.
        self._lock = threading.Lock()

    def register(self, name: str, factory: FactoryT) -> None:
        """Make a factory available by name, in place of any registered before.

        :param name: the name configuration uses.
        :param factory: the factory.
        """
        with self._lock:
            self._factories[name] = factory

    def get(self, name: str) -> FactoryT:
        """Return the factory registered under ``name``, loading an entry point.

        :param name: the name.
        :returns: the factory.
        :rtype: FactoryT
        :raises ValueError: if nothing has that name.
        """
        with self._lock:
            if (factory := self._factories.get(name)) is not None:
                return factory
        found = entry_points(group=self.group, name=name)
        if not found:
            with self._lock:
                known = sorted(self._factories)
            msg = f"No {self.kind} named {name!r}; registered: {known}"
            raise ValueError(msg)
        [entry_point] = found
        loaded: FactoryT = entry_point.load()
        self.register(name, loaded)
        return loaded
