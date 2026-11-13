"""The Opik trace destination, and every use of the Opik SDK that tracing needs.

Spans go to Opik's OTLP endpoint (``/v1/private/otel/v1/traces``), mapped to the
attributes its ``OpenTelemetryMapper`` reads: each span's project as
``opik.project_name``, the ids of the Opik trace and span it belongs to as
``opik.trace_id``, ``opik.span_id`` and ``opik.parent_span_id``, ``ol.tags`` as
``opik.tags``, and every other ``ol.*`` attribute as ``opik.metadata.*``, since
Opik files an attribute it has no rule for as the span's input. Other destinations
get the spans unmapped: the mapping is applied to a copy as each span ends.

What OTLP cannot carry goes through the SDK, here and nowhere else (verified
against the Opik 2.2.143 backend source; see the library spec §4.1 and §8.3):

- The call span becomes Opik's own trace and root span, through the SDK's tracker,
  so they carry Opik's native ``environment`` (no OTLP rule sets it), and a
  caller's ``@opik.track`` function, a ``@track`` function inside a run, and the
  LangChain tracer nest as they always have. The call span is not also sent over
  OTLP. A call inside a caller's ``@opik.track`` is a span of the caller's trace,
  in the caller's project; its tags, thread, prompt links, and degraded marking go
  on the caller's trace.
- Every other span is stamped with the Opik ids of its parent: an ol-llm span's,
  else the SDK context's current span (a caller's ``@opik.track``).
- LangChain runs are recorded by Opik's LangChain tracer (``opik_langchain``).

Configuration is the environment the SDK reads, so the SDK and the exporter agree
on the instance: ``OPIK_URL_OVERRIDE``, ``OPIK_WORKSPACE``, ``OPIK_TRACK_DISABLE``.
A forked child drops the SDK's client and connection bundle without closing them,
and builds new ones on its next SDK call.
"""

import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import opik
import requests
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import INVALID_SPAN, get_current_span
from opik import context_storage, exceptions, id_helpers, opik_context
from opik.api_objects import connection_resources
from opik.api_objects.opik_client import reset_global_client
from opik.config import OpikConfig
from opik.decorator import error_info_collector
from opik.decorator.arguments_helpers import TrackOptions
from opik.decorator.tracker import OpikTrackDecorator
from opik.hooks import HttpxClientHook, add_httpx_client_hook
from opik.types import ErrorInfoDict

from ol_llm import trace_attributes as attrs
from ol_llm.attribution import degraded_metadata
from ol_llm.keycloak import HttpxBearerAuth, KeycloakTokens, RequestsBearerAuth
from ol_llm.loops import register_at_fork
from ol_llm.policy import UseCase
from ol_llm.trace_destinations.interface import (
    CallTrace,
    DestinationSettings,
    TraceDestination,
    error_frames,
)
from ol_llm.trace_destinations.otlp import ForkSafeExporter

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables.graph import Graph
    from opentelemetry.util.types import AttributeValue

OTLP_TRACES_PATH = "/v1/private/otel/v1/traces"
"""Opik's OTLP/HTTP traces endpoint, relative to the SDK's API base URL."""

WORKSPACE_HEADER = "Comet-Workspace"
PROJECT_HEADER = "projectName"

PROJECT_NAME = "opik.project_name"
"""Opik files a span under the project this names (``OpenTelemetryService``)."""
TRACE_ID = "opik.trace_id"
SPAN_ID = "opik.span_id"
PARENT_SPAN_ID = "opik.parent_span_id"
TAGS = "opik.tags"
METADATA_PREFIX = "opik.metadata."

_RENAMED: Mapping[str, str] = {
    attrs.TAGS: TAGS,
    attrs.INPUT: "input",
    attrs.OUTPUT: "output",
    # Opik has no rule for these, so they would land in the span's input.
    attrs.EMBEDDINGS_DIMENSION_COUNT: f"{METADATA_PREFIX}embedding_dimensions",
    # PydanticAI's embedding span names the count this way; ol-llm's, ol.*.
    "inputs_count": f"{METADATA_PREFIX}inputs_count",
}

OpikClient = opik.Opik
"""The SDK's client, as the other seams' Opik adapters use it."""

# Drives the call's Opik trace with the halves of opik.track's own wrapper.
_TRACKER = OpikTrackDecorator()


def track_disabled() -> bool:
    """Return whether ``OPIK_TRACK_DISABLE`` (or the SDK's config file) turns it off.

    :returns: the SDK's setting.
    :rtype: bool
    """
    return OpikConfig().track_disable


def api_key_set() -> bool:
    """Return whether the SDK has a static ``OPIK_API_KEY``.

    :returns: whether one is set.
    :rtype: bool
    """
    return bool(OpikConfig().api_key)


def global_client() -> opik.Opik:
    """Return the SDK's global client.

    :returns: the client.
    :rtype: opik.Opik
    """
    return opik.get_global_client()


def flush_sdk() -> None:
    """Send everything the SDK has buffered: call traces, scores, experiments."""
    opik.flush_tracker()


# Registered on the SDK at most once: the SDK reads its hooks only when it builds
# its client, so a later configuration swaps the tokens this auth uses.
_sdk_auth: HttpxBearerAuth | None = None


def install_sdk_auth(tokens: KeycloakTokens | None) -> None:
    """Send the SDK's requests to Opik with ``tokens``, or with no credentials.

    :param tokens: the token cache; ``None`` sends none.
    """
    global _sdk_auth  # noqa: PLW0603
    host = httpx.URL(OpikConfig().url_override).host
    if _sdk_auth is not None:
        _sdk_auth.tokens = tokens
        _sdk_auth.host = host
    elif tokens is not None:
        _sdk_auth = HttpxBearerAuth(tokens, host=host)
        add_httpx_client_hook(
            HttpxClientHook(
                client_modifier=None, client_init_arguments={"auth": _sdk_auth}
            )
        )


def clear_sdk_auth() -> None:
    """Stop sending credentials with the SDK's requests. For tests."""
    if _sdk_auth is not None:
        _sdk_auth.tokens = None


def error_info(error: BaseException, *, content: bool) -> ErrorInfoDict:
    """Return what an Opik trace or span records about an error.

    Without content the message is left out, and the traceback keeps only where
    each frame is: the formatted one ends with the message, repeats those of
    chained exceptions, and quotes source lines.

    :param error: the error.
    :param content: whether the trace carries content.
    :returns: the error's type and traceback, and its message with content.
    :rtype: ErrorInfoDict
    """
    if content:
        return error_info_collector.collect(error)
    return {
        "exception_type": type(error).__name__,
        "traceback": error_frames(error),
    }


@dataclass(frozen=True, kw_only=True)
class _OpikIds:
    """Where a span goes in Opik.

    :param trace_id: the Opik trace, ``None`` for a span outside any.
    :param span_id: the span's Opik id.
    :param parent_span_id: its parent's Opik id, ``None`` at the top of a trace.
    :param project: the trace's project; ``None`` files it under the app's.
    :param bridged: the call span, which the SDK sends as Opik's own span.
    """

    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    project: str | None = None
    bridged: bool = False


_lock = threading.Lock()
# By OpenTelemetry span id, from each span's start to its end. Shared by every
# provider, so a span finds its parent's ids whichever app's provider started it.
_ids: dict[int, _OpikIds] = {}


def _resolve(parent_context: Context | None, default_project: str) -> _OpikIds:
    """Return the Opik ids of a span starting under ``parent_context``."""
    parent = get_current_span(parent_context)
    if parent is not INVALID_SPAN:
        with _lock:
            found = _ids.get(parent.get_span_context().span_id)
        if found is not None:
            if found.trace_id is None:
                return _OpikIds(project=found.project)
            return _OpikIds(
                trace_id=found.trace_id,
                span_id=id_helpers.generate_id(),
                parent_span_id=found.span_id,
                project=found.project,
            )
    # Under no ol-llm span: attach to the SDK's current span, e.g. a caller's
    # @opik.track. The SDK raises when its context has no span.
    try:
        headers = opik_context.get_distributed_trace_headers()
    except exceptions.OpikException:
        return _OpikIds()
    # Under a span from distributed headers there is no trace data, and the span
    # has the project.
    top = context_storage.top_span_data()
    trace = context_storage.get_trace_data()
    return _OpikIds(
        trace_id=headers["opik_trace_id"],
        span_id=id_helpers.generate_id(),
        parent_span_id=headers["opik_parent_span_id"],
        project=(trace.project_name if trace else None)
        or (top.project_name if top else None)
        or default_project,
    )


def _mapped(span: ReadableSpan, ids: _OpikIds, app: str) -> ReadableSpan:
    """Return a copy of ``span`` with the attributes Opik's OTLP mapper reads."""
    mapped: dict[str, AttributeValue] = {}
    for name, value in (span.attributes or {}).items():
        if (renamed := _RENAMED.get(name)) is not None:
            mapped[renamed] = value
        elif name.startswith(attrs.OL_PREFIX):
            mapped[METADATA_PREFIX + name.removeprefix(attrs.OL_PREFIX)] = value
        else:
            mapped[name] = value
    mapped[PROJECT_NAME] = ids.project or app
    if ids.trace_id is not None and ids.span_id is not None:
        mapped[TRACE_ID] = ids.trace_id
        mapped[SPAN_ID] = ids.span_id
        if ids.parent_span_id is not None:
            mapped[PARENT_SPAN_ID] = ids.parent_span_id
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=mapped,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class OpikSpans(SpanProcessor):
    """Sends an app's spans to Opik, each mapped and placed in its Opik trace.

    :param app: the project for a span outside any Opik trace.
    :param default_project: where Opik files a trace that names no project.
    :param export: the processor that sends the mapped spans.
    """

    def __init__(self, app: str, default_project: str, export: SpanProcessor) -> None:
        """Map spans for ``app`` and hand them to ``export``."""
        self._app = app
        self._default_project = default_project
        self._export = export

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Record where the span goes, from its parent or the SDK's context."""
        ids = _resolve(parent_context, self._default_project)
        with _lock:
            _ids[span.get_span_context().span_id] = ids

    def on_end(self, span: ReadableSpan) -> None:
        """Send a mapped copy, unless the SDK sends the span itself."""
        if span.context is None:
            return
        with _lock:
            ids = _ids.pop(span.context.span_id, _OpikIds())
        if not ids.bridged:
            self._export.on_end(_mapped(span, ids, self._app))

    def shutdown(self) -> None:
        """Shut down the export."""
        self._export.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Send what the export has buffered."""
        return self._export.force_flush(timeout_millis)


def _call() -> None:
    """Stand in for a traced function: the call's input is set from ``CallTrace``."""


class OpikDestination(TraceDestination):
    """Opik: spans over OTLP, the call as Opik's trace through the SDK.

    :param settings: the Keycloak token cache for the exporter.
    """

    def __init__(self, settings: DestinationSettings) -> None:
        """Read the instance from the SDK's configuration."""
        config = OpikConfig()
        self.url = config.url_override
        self.workspace = config.workspace
        self.default_project = config.project_name
        self.tokens = settings.tokens

    def span_processor(self, app: str) -> SpanProcessor:
        """Batch the app's spans, mapped, to Opik's OTLP endpoint."""
        return OpikSpans(
            app, self.default_project, BatchSpanProcessor(_exporter(self, app))
        )

    def active(self) -> bool:
        """Return whether the SDK's runtime switch has tracing on."""
        return opik.is_tracing_active()

    @contextmanager
    def call(self, call: CallTrace) -> Generator[None]:
        """Hold the call's Opik trace and root span current while it runs.

        Opened by the two halves ``opik.track``'s wrapper runs around a call,
        rather than by ``opik.track``, whose wrapper records an exception's message
        and source lines whatever the trace's content setting says. The exact opik
        pin keeps these private methods stable.
        """
        fields = call.fields
        options = TrackOptions(
            name=call.name,
            type="general",
            tags=list(fields.tags),
            metadata=dict(fields.metadata),
            capture_input=False,
            ignore_arguments=None,
            capture_output=call.content,
            generations_aggregator=None,
            flush=False,
            project_name=fields.project_name,
            create_duplicate_root_span=True,
            source=None,
            environment=fields.environment,
        )
        before = context_storage.get_trace_data()
        started = _TRACKER._before_call(  # noqa: SLF001
            func=_call, track_options=options, args=(), kwargs={}
        )
        try:
            # Under a span from distributed headers the SDK opens no trace, so
            # the context then has a span and no trace data.
            opened = before is None and context_storage.get_trace_data() is not None
            self._mark(call, opened=opened, started=started)
            yield
        finally:
            self._mark_degraded(call)
            error = call.error
            _TRACKER._after_call(  # noqa: SLF001
                output=None if error is not None else call.output,
                error_info=error_info(error, content=call.content)
                if isinstance(error, Exception)
                else None,
                capture_output=call.content,
                should_process_span_data=started,
            )

    def _mark(self, call: CallTrace, *, opened: bool, started: bool) -> None:
        """Set the input, put the tags and thread on the trace, and bridge the span.

        For a call that opened its trace the tags are already there; for one in a
        caller's trace, whose own metadata lands on the call's span, this adds
        them there. A caller's trace that already has a thread keeps it. Under a
        span from distributed headers there is no trace in this process to mark,
        and the call's span is a child of that span, in its project.
        """
        if call.input is not None:
            if started:
                opik_context.update_current_span(input=dict(call.input))
            if opened:
                opik_context.update_current_trace(input=dict(call.input))
        trace = context_storage.get_trace_data()
        if trace is not None:
            fields = call.fields
            opik_context.update_current_trace(
                tags=list(fields.tags),
                metadata={
                    key: fields.metadata[key]
                    for key in ("prompt_name", "prompt_version")
                    if key in fields.metadata
                },
                thread_id=None if trace.thread_id else fields.thread_id,
            )
        top = context_storage.top_span_data() if started else None
        trace_id = trace.id if trace is not None else top.trace_id if top else None
        if trace_id is None:
            # The SDK recorded nothing, so the call span goes over OTLP.
            return
        project = (top.project_name if top else None) or (
            trace.project_name if trace is not None else None
        )
        with _lock:
            _ids[call.span.get_span_context().span_id] = _OpikIds(
                trace_id=trace_id,
                span_id=top.id if top else None,
                project=project or self.default_project,
                bridged=True,
            )

    @staticmethod
    def _mark_degraded(call: CallTrace) -> None:
        """Mark the call's trace degraded, keeping what another call marked."""
        if not call.degraded_reasons:
            return
        if (trace := opik_context.get_current_trace_data()) is None:
            return
        earlier = (trace.metadata or {}).get("degraded_reason")
        marked = earlier.split(",") if earlier else []
        opik_context.update_current_trace(
            metadata=dict(degraded_metadata([*marked, *call.degraded_reasons]))
        )

    @contextmanager
    def chat_callbacks(
        self,
        call: CallTrace,
        *,
        use_case: UseCase,
        metadata: Mapping[str, Any],
        graph: "Graph | None",
    ) -> Generator[Sequence["BaseCallbackHandler"]]:
        """Yield Opik's LangChain tracer, attached to the call's root span."""
        # Imported here: it needs the langchain extra.
        from ol_llm.trace_destinations.opik_langchain import (  # noqa: PLC0415
            chat_tracer,
        )

        with chat_tracer(call, use_case=use_case, metadata=metadata, graph=graph) as (
            callbacks
        ):
            yield callbacks


def _exporter(destination: OpikDestination, app: str) -> SpanExporter:
    """Build the exporter to Opik's OTLP endpoint, authenticated with Keycloak.

    The ``projectName`` header names the app's project, where an Opik that reads
    no ``opik.project_name`` attribute (before 2.2.143 was not checked) files every
    span the exporter sends.
    """

    def otlp() -> SpanExporter:
        session = requests.Session()
        if destination.tokens is not None:
            session.auth = RequestsBearerAuth(destination.tokens)
        return OTLPSpanExporter(
            endpoint=destination.url.rstrip("/") + OTLP_TRACES_PATH,
            headers={WORKSPACE_HEADER: destination.workspace, PROJECT_HEADER: app},
            session=session,
        )

    return ForkSafeExporter(otlp)


def _after_fork_in_child() -> None:
    global _lock  # noqa: PLW0603
    _lock = threading.Lock()
    # The SDK caches one connection bundle (HTTP pool, sender and replay threads,
    # a SQLite replay store) per instance, and the global client leases it. Both
    # are dropped without closing: closing would join dead threads under locks
    # they may hold and close the parent's SQLite connection. The next SDK call
    # builds a new client and bundle. Private to the exactly pinned opik; the
    # forked-child test catches a change.
    manager = connection_resources.MANAGER
    manager._lock = threading.Lock()  # noqa: SLF001
    manager._entries = {}  # noqa: SLF001
    reset_global_client(end_client=False)


register_at_fork(after_in_child=_after_fork_in_child)
