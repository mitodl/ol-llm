"""The plain OTLP trace destination, and the exporter both OTLP destinations use.

``otlp`` sends ol-llm's spans as they are, ``gen_ai.*`` and ``ol.*`` attributes and
all, to the endpoint the standard ``OTEL_EXPORTER_OTLP_*`` variables name, e.g. a
Tempo or an OpenTelemetry collector.
"""

import threading
import weakref
from collections.abc import Callable, Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from ol_llm.loops import register_at_fork
from ol_llm.trace_destinations.interface import DestinationSettings, TraceDestination


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


class OtlpDestination(TraceDestination):
    """Spans as ol-llm records them, over OTLP/HTTP to a configured endpoint.

    :param settings: unused; the exporter reads ``OTEL_EXPORTER_OTLP_*`` itself.
    """

    def __init__(self, settings: DestinationSettings) -> None:
        """Read nothing; the exporter reads its own configuration."""

    def span_processor(self, app: str) -> SpanProcessor:  # noqa: ARG002
        """Batch the app's spans to the configured endpoint."""
        return BatchSpanProcessor(ForkSafeExporter(OTLPSpanExporter))


_fork_safe: "weakref.WeakSet[ForkSafeExporter]" = weakref.WeakSet()


def _after_fork_in_child() -> None:
    for exporter in list(_fork_safe):
        exporter._after_fork()  # noqa: SLF001


register_at_fork(after_in_child=_after_fork_in_child)
