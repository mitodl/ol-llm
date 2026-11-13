"""What a trace destination does for ol-llm (library spec §8.3 seam 3).

ol-llm's trace API records every call as OpenTelemetry spans: a call span for each
entry-point call (``run``, ``traced_chat``, ``traced_embedding``), with the §4
attribution as ``ol.*`` attributes, and PydanticAI's, embedding, and memory spans
under it. A destination receives those spans through the span processor it puts
on each app's tracer provider, and maps them to its backend.

A destination may also take part in each call while it runs (``call``), for what
a backend needs that a span cannot carry, e.g. Opik's native trace environment, or
callbacks for LangChain runs (``chat_callbacks``).
"""

import traceback
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor
from opentelemetry.trace import Span, Status

from ol_llm.attribution import DegradedReason, TraceFields
from ol_llm.keycloak import KeycloakTokens
from ol_llm.policy import UseCase

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables.graph import Graph
    from opentelemetry.util.types import AttributeValue


def parse_bool(value: str, name: str) -> bool:
    """Read a boolean setting.

    :param value: the setting, e.g. ``true`` or ``0``.
    :param name: the variable, for the error.
    :returns: the value.
    :rtype: bool
    :raises ValueError: if it is not one of ``true``, ``1``, ``yes``, ``false``,
        ``0``, ``no``.
    """
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    msg = f"{name} must be true or false, not {value!r}"
    raise ValueError(msg)


def copy_span(
    span: ReadableSpan,
    *,
    attributes: Mapping[str, "AttributeValue"],
    events: Sequence[Event] | None = None,
    status: Status | None = None,
) -> ReadableSpan:
    """Return a copy of an ended span with other attributes, events, or status.

    How a destination maps spans without changing what the others get.

    :param span: the span.
    :param attributes: the copy's attributes.
    :param events: the copy's events; the span's when omitted.
    :param status: the copy's status; the span's when omitted.
    :returns: the copy.
    :rtype: ReadableSpan
    """
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=span.events if events is None else events,
        links=span.links,
        kind=span.kind,
        status=span.status if status is None else status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def error_frames(error: BaseException) -> str:
    """Return where each frame of an error's traceback is, and nothing else.

    What a trace keeps of a traceback without content: the formatted one ends with
    the message, repeats those of chained exceptions, and quotes source lines.

    :param error: the error.
    :returns: one ``File "...", line N, in name`` line per frame.
    :rtype: str
    """
    return "".join(
        f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n'
        for frame in traceback.extract_tb(error.__traceback__)
    )


@dataclass(kw_only=True)
class CallTrace:
    """One entry-point call while it runs, as the destinations see it.

    The trace API fills in the outcome before the call span ends.

    :param name: the call's name, the use case's.
    :param fields: the attribution the call opened with.
    :param span: the call span, current while the call runs.
    :param content: whether the trace carries input, output, and error messages.
    :param input: the call's input, ``None`` without content.
    :param output: what the call returned, once it has.
    :param error: what the call raised, if it did.
    :param degraded_reasons: why its result is degraded, in order.
    """

    name: str
    fields: TraceFields
    span: Span
    content: bool
    input: Mapping[str, Any] | None
    output: Any = None
    error: BaseException | None = None
    degraded_reasons: list[DegradedReason] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class DestinationSettings:
    """What a destination is built from.

    :param environ: the process environment.
    :param tokens: the Keycloak token cache, when ``OPIK_KEYCLOAK_*`` is set.
    """

    environ: Mapping[str, str]
    tokens: KeycloakTokens | None


class TraceDestination(ABC):
    """Where ol-llm's spans go, and how they map to that backend."""

    @classmethod
    def disabled(cls, environ: Mapping[str, str]) -> bool:  # noqa: ARG003
        """Return whether the environment turns tracing off for this backend.

        Checked before the destination is built, as the backend's SDK reads its
        setting; any destination named that says so turns tracing off.

        :param environ: the process environment.
        :returns: whether tracing is off.
        :rtype: bool
        """
        return False

    @abstractmethod
    def span_processor(self, app: str) -> SpanProcessor:
        """Return the processor that sends one app's spans to this destination.

        Called once per app, when its tracer provider is built.

        :param app: the app; a span outside any call belongs to it.
        :returns: the processor.
        :rtype: SpanProcessor
        """

    def active(self) -> bool:
        """Return whether calls are traced right now; a runtime switch, if any.

        :returns: whether to record spans.
        :rtype: bool
        """
        return True

    @contextmanager
    def call(self, call: CallTrace) -> Generator[None]:  # noqa: ARG002
        """Hold whatever the destination needs open while a call runs.

        Entered after the call span starts, and left before it ends, with the
        call's outcome filled in.

        :param call: the call.
        """
        yield

    @contextmanager
    def chat_callbacks(
        self,
        call: CallTrace,  # noqa: ARG002
        *,
        use_case: UseCase,  # noqa: ARG002
        metadata: Mapping[str, Any],  # noqa: ARG002
        graph: "Graph | None",  # noqa: ARG002
    ) -> Generator[Sequence["BaseCallbackHandler"]]:
        """Yield LangChain callbacks that record a chat turn's runs here.

        Needs the ``langchain`` extra. None by default: a LangChain run is not an
        OpenTelemetry span.

        :param call: the ``traced_chat`` call, entered.
        :param use_case: the call site; its model names the chat spans.
        :param metadata: call-site fields for the runs' spans.
        :param graph: the LangGraph graph, for a backend that shows it.
        """
        yield ()

    def flush(self, timeout_millis: int) -> bool:  # noqa: ARG002
        """Send what the destination buffers outside its span processors.

        :param timeout_millis: how long to wait.
        :returns: whether it finished in time.
        :rtype: bool
        """
        return True

    def shutdown(self) -> None:  # noqa: B027 - optional, a no-op by default
        """Release what the destination holds outside its span processors."""
