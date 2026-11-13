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

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span

from ol_llm.attribution import DegradedReason, TraceFields
from ol_llm.keycloak import KeycloakTokens
from ol_llm.policy import UseCase

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables.graph import Graph


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

    def shutdown(self) -> None:  # noqa: B027 - optional, a no-op by default
        """Release what the destination holds outside its span processors."""
