"""The plain OTLP trace destination, and the exporter both OTLP destinations use.

``otlp`` sends ol-llm's spans, unmapped, to the endpoint the standard
``OTEL_EXPORTER_OTLP_*`` variables name, e.g. a Tempo or an OpenTelemetry
collector. It keeps the ``ol.*`` and ``gen_ai.*`` attributes and
``trace_attributes.CONTENT``, and without ``OL_LLM_TRACE_OTLP_CONTENT=true`` it
drops what carries learner content or identifies a user: inputs, outputs, messages,
note text and queries, user ids, and error messages. The data-handling policy for
learner content is not decided, so a second destination gets none by default.
"""

import threading
import weakref
from collections.abc import Callable, Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status

from ol_llm import trace_attributes as attrs
from ol_llm.loops import register_at_fork
from ol_llm.trace_destinations.interface import (
    DestinationSettings,
    TraceDestination,
    copy_span,
    parse_bool,
)

CONTENT_ENV = "OL_LLM_TRACE_OTLP_CONTENT"


class ForkSafeExporter(SpanExporter):
    """An exporter that a forked child builds again on its next export.

    OTLP exporters hold a ``requests.Session`` whose pooled sockets the parent
    still uses. OpenTelemetry restarts its batch thread in the child, but not the
    exporter.

    :param make: builds the exporter.
    """

    def __init__(self, make: Callable[[], SpanExporter]) -> None:
        """Build the exporter on first use."""
        self._make = make
        self._lock = threading.Lock()
        self._exporter: SpanExporter | None = None
        _fork_safe.add(self)

    def _after_fork(self) -> None:
        self._lock = threading.Lock()
        self._exporter = None

    def _current(self) -> SpanExporter:
        with self._lock:
            if self._exporter is None:
                self._exporter = self._make()
            return self._exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Send the spans with the current exporter."""
        return self._current().export(spans)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Flush the current exporter, if one was built."""
        with self._lock:
            exporter = self._exporter
        return exporter is None or exporter.force_flush(timeout_millis)

    def shutdown(self) -> None:
        """Shut down the current exporter, if one was built."""
        with self._lock:
            exporter, self._exporter = self._exporter, None
        if exporter is not None:
            exporter.shutdown()


def _kept(name: str, *, content: bool) -> bool:
    if name in attrs.CONTENT:
        return content
    if not name.startswith((attrs.OL_PREFIX, "gen_ai.")):
        return False
    return content or (
        name != attrs.USER_ID and not name.startswith(attrs.GEN_AI_CONTENT_PREFIXES)
    )


def _filtered(span: ReadableSpan, *, content: bool) -> ReadableSpan:
    """Return a copy of ``span`` with only what the destination may receive."""
    attributes = {
        name: value
        for name, value in (span.attributes or {}).items()
        if _kept(name, content=content)
    }
    if content:
        return copy_span(span, attributes=attributes)
    events = [
        Event(
            attrs.EXCEPTION_EVENT,
            {
                name: value
                for name, value in (event.attributes or {}).items()
                if name in {attrs.EXCEPTION_TYPE, attrs.EXCEPTION_ESCAPED}
            },
            timestamp=event.timestamp,
        )
        for event in span.events
        if event.name == attrs.EXCEPTION_EVENT
    ]
    return copy_span(
        span,
        attributes=attributes,
        events=events,
        status=Status(span.status.status_code),
    )


class _Filtered(SpanProcessor):
    """Hands each ended span to ``export`` with only what it may receive."""

    def __init__(self, export: SpanProcessor, *, content: bool) -> None:
        self._export = export
        self._content = content

    def on_end(self, span: ReadableSpan) -> None:
        self._export.on_end(_filtered(span, content=self._content))

    def shutdown(self) -> None:
        self._export.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._export.force_flush(timeout_millis)


class OtlpDestination(TraceDestination):
    """ol-llm's spans, unmapped, over OTLP/HTTP to a configured endpoint.

    :param settings: ``OL_LLM_TRACE_OTLP_CONTENT=true`` in its environment lets
        content and user ids through. The exporter reads
        ``OTEL_EXPORTER_OTLP_*`` itself.
    :raises ValueError: if ``OL_LLM_TRACE_OTLP_CONTENT`` is not a boolean.
    """

    def __init__(self, settings: DestinationSettings) -> None:
        """Read whether the destination may receive content."""
        self.content = parse_bool(
            settings.environ.get(CONTENT_ENV, "false"), CONTENT_ENV
        )

    def span_processor(self, app: str) -> SpanProcessor:  # noqa: ARG002
        """Batch the app's spans, filtered, to the configured endpoint."""
        return _Filtered(
            BatchSpanProcessor(ForkSafeExporter(OTLPSpanExporter)),
            content=self.content,
        )


_fork_safe: "weakref.WeakSet[ForkSafeExporter]" = weakref.WeakSet()


def _after_fork_in_child() -> None:
    for exporter in list(_fork_safe):
        exporter._after_fork()  # noqa: SLF001


register_at_fork(after_in_child=_after_fork_in_child)
