"""A trace destination that keeps spans in memory, for tests.

``OL_LLM_TRACE_DESTINATIONS=memory`` names it. It keeps the spans as ol-llm records
them, before any backend's mapping, so a test sees the ``gen_ai.*`` and ``ol.*``
attributes another backend would get. Read them with ``tracing.destinations()``.
"""

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from ol_llm.trace_destinations.interface import DestinationSettings, TraceDestination


class MemoryDestination(TraceDestination):
    """Every span of every app, in memory, as it ended.

    :param settings: unused.
    """

    def __init__(self, settings: DestinationSettings | None = None) -> None:  # noqa: ARG002
        """Start with no spans."""
        self.exporter = InMemorySpanExporter()

    def span_processor(self, app: str) -> SpanProcessor:  # noqa: ARG002
        """Keep the app's spans as they end."""
        return SimpleSpanProcessor(self.exporter)

    def spans(self) -> tuple[ReadableSpan, ...]:
        """Return the spans kept so far, oldest first.

        :returns: the spans.
        :rtype: tuple[ReadableSpan, ...]
        """
        return self.exporter.get_finished_spans()
