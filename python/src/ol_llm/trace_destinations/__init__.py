"""Trace destinations, chosen by name like memory backends (library spec §8.3).

``OL_LLM_TRACE_DESTINATIONS`` names them, comma-separated: ``opik`` (the default),
``otlp`` for any OTLP/HTTP endpoint the standard ``OTEL_EXPORTER_OTLP_*`` variables
name (e.g. Tempo), and ``memory`` for tests. Every span goes to each destination
named. Another package adds a destination with an entry point in the
``ol_llm.trace_destinations`` group, naming a factory that takes
``DestinationSettings`` and returns a ``TraceDestination``, or in code with
``register_trace_destination``.
"""

from collections.abc import Callable, Mapping, Sequence

from ol_llm.plugins import NamedFactories
from ol_llm.trace_destinations.interface import (
    CallTrace,
    DestinationSettings,
    TraceDestination,
    error_frames,
)
from ol_llm.trace_destinations.memory import MemoryDestination
from ol_llm.trace_destinations.opik_destination import OpikDestination
from ol_llm.trace_destinations.otlp import OtlpDestination

__all__ = [
    "DEFAULT_DESTINATIONS",
    "ENTRY_POINT_GROUP",
    "CallTrace",
    "DestinationFactory",
    "DestinationSettings",
    "MemoryDestination",
    "OpikDestination",
    "OtlpDestination",
    "TraceDestination",
    "build_destinations",
    "error_frames",
    "register_trace_destination",
    "tracing_disabled",
]

ENTRY_POINT_GROUP = "ol_llm.trace_destinations"
DEFAULT_DESTINATIONS = ("opik",)

DestinationFactory = Callable[[DestinationSettings], TraceDestination]
"""Builds a destination from the environment and the Keycloak token cache."""

_destinations: NamedFactories[DestinationFactory] = NamedFactories(
    group=ENTRY_POINT_GROUP,
    kind="trace destination",
    builtins={
        "opik": OpikDestination,
        "otlp": OtlpDestination,
        "memory": MemoryDestination,
    },
)


def register_trace_destination(name: str, factory: DestinationFactory) -> None:
    """Make a destination available by name, in place of any registered before.

    :param name: what ``OL_LLM_TRACE_DESTINATIONS`` names it as.
    :param factory: builds the destination.
    """
    _destinations.register(name, factory)


def tracing_disabled(names: Sequence[str], environ: Mapping[str, str]) -> bool:
    """Return whether a named destination's backend has tracing turned off.

    E.g. ``OPIK_TRACK_DISABLE=true`` with ``opik`` named.

    :param names: the destinations' names.
    :param environ: the process environment.
    :returns: whether any of them says tracing is off.
    :rtype: bool
    :raises ValueError: if a name names no destination.
    """
    for name in names:
        factory = _destinations.get(name)
        if (
            isinstance(factory, type)
            and issubclass(factory, TraceDestination)
            and factory.disabled(environ)
        ):
            return True
    return False


def build_destinations(
    names: Sequence[str], settings: DestinationSettings
) -> tuple[TraceDestination, ...]:
    """Build the named destinations, in order.

    :param names: the destinations' names.
    :param settings: what each is built from.
    :returns: the destinations.
    :rtype: tuple[TraceDestination, ...]
    :raises ValueError: if a name names no destination.
    """
    return tuple(_destinations.get(name)(settings) for name in names)
