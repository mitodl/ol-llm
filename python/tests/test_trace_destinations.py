import contextlib
import gzip
import json
import os
import threading
from collections.abc import Generator, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import opik
import pytest
from conftest import opik_context_is_empty
from ol_llm import (
    Attribution,
    LLMRateLimited,
    PostProcessed,
    UseCase,
    run_sync,
    traced_embedding,
    tracing,
)
from ol_llm.models import build_agent, override_models
from ol_llm.trace_destinations import (
    DestinationSettings,
    MemoryDestination,
    opik_destination,
    register_trace_destination,
)
from ol_llm.trace_destinations.opik_destination import (
    OTLP_TRACES_PATH,
    PARENT_SPAN_ID,
    PROJECT_NAME,
    TRACE_ID,
)
from ol_llm.tracing import KeycloakClient, TracingConfig, configure_tracing
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from opik.api_objects.opik_client import reset_global_client
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

SUMMARY = UseCase(
    app="mit-learn", name="summary", profile="request", model="openai:gpt-4o-mini"
)
EMBEDDING = UseCase(
    app="mit-learn",
    name="search",
    profile="request",
    model="openai:text-embedding-3-small",
)
ATTRIBUTION = Attribution(user_id="user-1", thread_id="thread-1", run_id="run-1")
REAL_EXPORTER = opik_destination._exporter  # noqa: SLF001


def replying(text: str) -> FunctionModel:
    def reply(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(reply)


def failing(status: int) -> FunctionModel:
    def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status, "gpt-4o-mini", body="provider said no")

    return FunctionModel(fail)


def agent_for(model: FunctionModel) -> Agent[None, str]:
    with override_models(model):
        return build_agent(SUMMARY)


def configured(**config: Any) -> MemoryDestination:  # noqa: ANN401
    """Trace to Opik and memory, and return the memory destination."""
    tracing.reset()
    configure_tracing(TracingConfig(destinations=("opik", "memory"), **config))
    [memory] = [d for d in tracing.destinations() if isinstance(d, MemoryDestination)]
    return memory


@pytest.fixture
def memory() -> MemoryDestination:
    return configured()


def by_name(spans: Iterator[ReadableSpan] | tuple[ReadableSpan, ...]) -> dict:
    return {span.name: span for span in spans}


def test_another_destination_gets_the_call_span_with_the_attribution(
    memory: MemoryDestination,
) -> None:
    run_sync(agent_for(replying("a summary")), "summarize", attribution=ATTRIBUTION)

    spans = memory.spans()
    call = by_name(spans)["summary"]
    attributes = call.attributes or {}
    assert {
        key: attributes[f"ol.{key}"]
        for key in ("app", "use_case", "environment", "user_id", "run_id")
    } == {
        "app": "mit-learn",
        "use_case": "summary",
        "environment": "ci",
        "user_id": "user-1",
        "run_id": "run-1",
    }
    assert attributes["ol.policy_max_retries"] >= 0
    assert set(attributes["ol.tags"]) == {"app:mit-learn", "use_case:summary", "env:ci"}
    assert attributes["gen_ai.conversation.id"] == "thread-1"
    assert json.loads(attributes["ol.input"]) == {"user_prompt": "summarize"}
    assert json.loads(attributes["ol.output"]) == "a summary"
    assert call.parent is None
    model_spans = [span for span in spans if span is not call]
    assert model_spans
    assert all(span.context.trace_id == call.context.trace_id for span in spans)
    # Unmapped: Opik's names reach only the Opik destination.
    assert not [
        name for span in spans for name in span.attributes or {} if "opik" in name
    ]


def test_opik_gets_the_call_as_its_trace_and_the_model_spans_inside_it(
    memory: MemoryDestination,  # noqa: ARG001
    exported: InMemorySpanExporter,
) -> None:
    with opik.record_traces_locally() as storage:
        run_sync(agent_for(replying("ok")), "hi", attribution=ATTRIBUTION)
        [trace] = storage.trace_trees
        [root] = storage.span_trees
    tracing.flush()

    assert trace.name == "summary"
    assert trace.thread_id == "thread-1"
    assert root.name == "summary"
    spans = exported.get_finished_spans()
    # The call span is Opik's own trace and root span, sent by the SDK.
    assert "summary" not in {span.name for span in spans}
    assert {(s.attributes or {})[TRACE_ID] for s in spans} == {trace.id}
    assert {(s.attributes or {})[PROJECT_NAME] for s in spans} == {"mit-learn"}
    top = [s for s in spans if s.parent is None or s.parent.span_id not in ids(spans)]
    assert {(s.attributes or {})[PARENT_SPAN_ID] for s in top} == {root.id}


@pytest.mark.parametrize("content", [True, False])
def test_a_run_under_a_span_from_distributed_headers_attaches_to_it(
    content: bool,  # noqa: FBT001
    exported: InMemorySpanExporter,
) -> None:
    configured(include_content=content)

    @opik.track(project_name="mit-learn")
    def upstream() -> Mapping[str, Any]:
        return opik.opik_context.get_distributed_trace_headers()

    @opik.track(project_name="ol-data-platform")
    def downstream() -> str:
        # The SDK opens a span under the headers' span, and no trace.
        assert opik.opik_context.get_current_trace_data() is None
        return run_sync(agent_for(replying("ok")), "hi").output

    with opik.record_traces_locally() as storage:
        headers = upstream()
        assert downstream(opik_distributed_trace_headers=headers) == "ok"  # type: ignore[call-arg]
        tracing.flush()
        span_names = [span.name for tree in storage.span_trees for span in walk(tree)]
    assert opik_context_is_empty()

    assert span_names.count("summary") == 1
    spans = exported.get_finished_spans()
    assert spans
    # The call span is the SDK's, not also sent over OTLP.
    assert "summary" not in {span.name for span in spans}
    assert {(s.attributes or {})[TRACE_ID] for s in spans} == {headers["opik_trace_id"]}
    assert {(s.attributes or {})[PROJECT_NAME] for s in spans} == {"ol-data-platform"}


def walk(span: Any) -> Iterator[Any]:  # noqa: ANN401
    yield span
    for child in span.spans:
        yield from walk(child)


def ids(spans: tuple[ReadableSpan, ...]) -> set[int]:
    return {span.context.span_id for span in spans if span.context}


def test_a_degraded_result_is_marked_on_the_call_span(
    memory: MemoryDestination,
) -> None:
    def partial(output: str) -> PostProcessed[str]:
        return PostProcessed(output=output, degraded_reasons=("partial_output",))

    with opik.record_traces_locally() as storage:
        run_sync(agent_for(replying("ok")), "hi", postprocess=partial)
        [trace] = storage.trace_trees

    attributes = by_name(memory.spans())["summary"].attributes or {}
    assert attributes["ol.degraded"] is True
    assert attributes["ol.degraded_reason"] == "partial_output"
    assert (trace.metadata or {})["degraded_reason"] == "partial_output"


def test_without_content_the_call_span_has_no_input_or_output() -> None:
    memory = configured(include_content=False)

    run_sync(agent_for(replying("secret answer")), "secret question")

    call = by_name(memory.spans())["summary"]
    assert not {"ol.input", "ol.output"} & (call.attributes or {}).keys()
    assert "secret" not in repr(memory.spans())


def test_a_failed_call_records_its_error_on_the_call_span(
    memory: MemoryDestination,
) -> None:
    with pytest.raises(LLMRateLimited):
        run_sync(agent_for(failing(429)), "hi")

    call = by_name(memory.spans())["summary"]
    assert call.status.status_code == StatusCode.ERROR
    [event] = call.events
    assert event.name == "exception"
    assert "LLMRateLimited" in str((event.attributes or {})["exception.type"])


def test_tracing_only_to_memory_sends_nothing_to_opik(
    exported: InMemorySpanExporter,
) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(destinations=("memory",)))
    [memory] = tracing.destinations()
    assert isinstance(memory, MemoryDestination)

    with opik.record_traces_locally() as storage:
        run_sync(agent_for(replying("ok")), "hi")
        traces = storage.trace_trees
    tracing.flush()

    assert traces == []
    assert exported.get_finished_spans() == ()
    assert "summary" in by_name(memory.spans())


def test_an_embedding_span_goes_to_every_destination(
    memory: MemoryDestination, exported: InMemorySpanExporter
) -> None:
    with traced_embedding(EMBEDDING, ["a"]) as request:
        request.record([[0.1, 0.2]], input_tokens=3)
    tracing.flush()

    plain = by_name(memory.spans())["embeddings text-embedding-3-small"]
    assert (plain.attributes or {})["ol.inputs_count"] == 1
    assert (plain.attributes or {})["gen_ai.embeddings.dimension.count"] == 2
    [mapped] = exported.get_finished_spans()
    assert (mapped.attributes or {})["opik.metadata.inputs_count"] == 1
    assert (mapped.attributes or {})["opik.metadata.embedding_dimensions"] == 2


def test_opik_files_ol_attributes_as_metadata_and_keeps_the_rest(
    exported: InMemorySpanExporter,
) -> None:
    provider = tracing.tracer_provider("mit-learn")
    assert provider is not None
    attributes: Mapping[str, Any] = {
        "ol.app": "mit-learn",
        "ol.tags": ["app:mit-learn"],
        "gen_ai.request.model": "gpt-4o-mini",
        "memory.query": "content",
        "inputs_count": 2,
        "input_type": "query",
        "embedding_settings": '{"dimensions": 256}',
    }
    with provider.get_tracer("test").start_as_current_span("s", attributes=attributes):
        pass
    tracing.flush()

    [span] = exported.get_finished_spans()
    assert dict(span.attributes or {}) == {
        "opik.metadata.app": "mit-learn",
        "opik.tags": ("app:mit-learn",),
        "gen_ai.request.model": "gpt-4o-mini",
        "memory.query": "content",
        "opik.metadata.inputs_count": 2,
        "opik.metadata.input_type": "query",
        "opik.metadata.embedding_settings": '{"dimensions": 256}',
        PROJECT_NAME: "mit-learn",
    }


def test_an_unregistered_destination_is_refused() -> None:
    tracing.reset()

    with pytest.raises(ValueError, match="No trace destination named 'nowhere'"):
        configure_tracing(TracingConfig(destinations=("nowhere",)))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, ("opik",)), ("opik, memory", ("opik", "memory")), ("otlp", ("otlp",))],
)
def test_from_env_reads_the_destinations(
    value: str | None,
    expected: tuple[str, ...],
) -> None:
    env = {} if value is None else {"OL_LLM_TRACE_DESTINATIONS": value}

    assert TracingConfig.from_env(env).destinations == expected


def test_from_env_refuses_an_empty_destination_list() -> None:
    with pytest.raises(ValueError, match="names no trace destination"):
        TracingConfig.from_env({"OL_LLM_TRACE_DESTINATIONS": " , "})


def test_a_registered_destination_is_built_with_the_keycloak_tokens() -> None:
    built: list[DestinationSettings] = []
    destination = MemoryDestination()

    def factory(settings: DestinationSettings) -> MemoryDestination:
        built.append(settings)
        return destination

    register_trace_destination("recorded", factory)
    tracing.reset()
    configure_tracing(
        TracingConfig(
            destinations=("recorded",),
            keycloak=KeycloakClient(
                token_url="https://sso.example/token", client_id="c", client_secret="s"
            ),
        )
    )
    run_sync(agent_for(replying("ok")), "hi")

    [settings] = built
    assert settings.tokens is tracing._state().tokens  # noqa: SLF001
    assert "summary" in by_name(destination.spans())


class RecordingOpik(BaseHTTPRequestHandler):
    """Records every request body by path, as Opik would receive it."""

    protocol_version = "HTTP/1.1"
    received: list[tuple[str, bytes]]

    def _answer(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        self.received.append((self.path, body))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    do_GET = do_POST = do_PATCH = do_PUT = _answer  # noqa: N815

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def recording_opik(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[list[tuple[str, bytes]]]:
    received: list[tuple[str, bytes]] = []
    handler = type("Handler", (RecordingOpik,), {"received": received})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(opik_destination, "_exporter", REAL_EXPORTER)
    previous = os.environ["OPIK_URL_OVERRIDE"]
    os.environ["OPIK_URL_OVERRIDE"] = f"http://127.0.0.1:{server.server_address[1]}/api"
    reset_global_client()
    yield received
    os.environ["OPIK_URL_OVERRIDE"] = previous
    tracing.reset()
    reset_global_client()
    server.shutdown()
    server.server_close()


def objects_with(value: object, key: str) -> Iterator[dict[str, Any]]:
    """Yield every object in a JSON document that has ``key``."""
    if isinstance(value, dict):
        if key in value:
            yield value
        for item in value.values():
            yield from objects_with(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from objects_with(item, key)


def test_opik_receives_the_trace_by_rest_and_the_spans_by_otlp(
    recording_opik: list[tuple[str, bytes]],
) -> None:
    tracing.reset()
    configure_tracing(TracingConfig())
    run_sync(agent_for(replying("ok")), "hi", attribution=ATTRIBUTION)
    assert tracing.flush(10_000)

    traces = [
        trace
        for path, body in recording_opik
        if "/v1/private/traces" in path and body
        for trace in objects_with(json.loads(body), "environment")
        if trace.get("name") == "summary" and "thread_id" in trace
    ]
    [trace] = traces
    assert trace["environment"] == "ci"
    assert trace["thread_id"] == "thread-1"
    assert trace["project_name"] == "mit-learn"
    assert set(trace["tags"]) >= {"app:mit-learn", "use_case:summary", "env:ci"}
    assert trace["metadata"]["user_id"] == "user-1"

    otlp = [body for path, body in recording_opik if path.endswith(OTLP_TRACES_PATH)]
    assert otlp
    spans = []
    for body in otlp:
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        spans.extend(
            span
            for resource in request.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        )
    assert spans
    for span in spans:
        attributes = {a.key: a.value.string_value for a in span.attributes}
        assert attributes[TRACE_ID] == trace["id"]
        assert attributes[PROJECT_NAME] == "mit-learn"
        assert "ol_llm.opik_project" not in attributes


def otlp_spans(
    received: list[tuple[str, bytes]],
) -> dict[str, Any]:
    """Return the spans an OTLP collector received at ``/v1/traces``, by name."""
    spans = {}
    for path, body in received:
        if path == "/v1/traces":
            request = ExportTraceServiceRequest.FromString(body)
            spans.update(
                {
                    span.name: span
                    for resource in request.resource_spans
                    for scope in resource.scope_spans
                    for span in scope.spans
                }
            )
    return spans


def attributes_of(span: Any) -> dict[str, Any]:  # noqa: ANN401
    return {a.key: a.value for a in span.attributes}


def traced_to_otlp(
    monkeypatch: pytest.MonkeyPatch, model: FunctionModel, *, content: str | None
) -> None:
    endpoint = os.environ["OPIK_URL_OVERRIDE"].removesuffix("/api") + "/v1/traces"
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
    if content is not None:
        monkeypatch.setenv("OL_LLM_TRACE_OTLP_CONTENT", content)
    tracing.reset()
    configure_tracing(TracingConfig(destinations=("otlp",)))
    with contextlib.suppress(LLMRateLimited):
        run_sync(agent_for(model), "the learner's question", attribution=ATTRIBUTION)
    assert tracing.flush(10_000)


def test_the_otlp_destination_sends_no_content_or_user_id_by_default(
    recording_opik: list[tuple[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traced_to_otlp(monkeypatch, replying("the answer"), content=None)

    spans = otlp_spans(recording_opik)
    call = attributes_of(spans["summary"])
    assert call["ol.use_case"].string_value == "summary"
    assert call["ol.run_id"].string_value == "run-1"
    assert call["gen_ai.conversation.id"].string_value == "thread-1"
    assert not {"ol.input", "ol.output", "ol.user_id"} & call.keys()
    names = {name for span in spans.values() for name in attributes_of(span)}
    assert "gen_ai.operation.name" in names
    assert not {"gen_ai.input.messages", "gen_ai.output.messages"} & names
    assert not [name for name in names if "opik" in name]
    assert not [name for name in names if not name.startswith(("ol.", "gen_ai."))]
    assert "the learner's question" not in repr(spans)
    assert "the answer" not in repr(spans)
    assert "user-1" not in repr(spans)


def test_the_otlp_destination_keeps_only_an_errors_type_by_default(
    recording_opik: list[tuple[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traced_to_otlp(monkeypatch, failing(429), content=None)

    call = otlp_spans(recording_opik)["summary"]
    assert not call.status.message
    [event] = call.events
    assert event.name == "exception"
    assert {a.key for a in event.attributes} <= {"exception.type", "exception.escaped"}
    assert "LLMRateLimited" in str(event.attributes)
    assert "provider said no" not in repr(otlp_spans(recording_opik))


def test_the_otlp_destination_sends_content_when_opted_in(
    recording_opik: list[tuple[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traced_to_otlp(monkeypatch, replying("the answer"), content="true")

    spans = otlp_spans(recording_opik)
    call = attributes_of(spans["summary"])
    assert json.loads(call["ol.input"].string_value) == {
        "user_prompt": "the learner's question"
    }
    assert json.loads(call["ol.output"].string_value) == "the answer"
    assert call["ol.user_id"].string_value == "user-1"


def test_the_otlp_content_setting_must_be_a_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OL_LLM_TRACE_OTLP_CONTENT", "sometimes")
    tracing.reset()

    with pytest.raises(ValueError, match="OL_LLM_TRACE_OTLP_CONTENT"):
        configure_tracing(TracingConfig(destinations=("otlp",)))
