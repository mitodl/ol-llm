import asyncio
from decimal import Decimal

import ol_llm.catalog as catalog_module
import opik
import pytest
from ol_llm import Attribution, UseCase, traced_embedding, tracing
from ol_llm.catalog import Catalog, CatalogEntry, load_catalog
from ol_llm.testing import tracing_disabled
from ol_llm.tracing import PROJECT_ATTRIBUTE, TracingConfig, configure_tracing
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

MODEL = "openai:text-embedding-3-small"

SEARCH = UseCase(app="mit-learn", name="search-query", profile="request", model=MODEL)
ATTRIBUTION = Attribution(user_id="user-1", thread_id="thread-1", run_id="run-1")
VECTORS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def embed_two(use_case: UseCase = SEARCH, *, input_tokens: int = 7) -> None:
    with traced_embedding(use_case, ["a", "b"], attribution=ATTRIBUTION) as request:
        request.record(VECTORS, input_tokens=input_tokens)


def test_the_root_trace_matches_embeds() -> None:
    with opik.record_traces_locally() as storage:
        embed_two()
        [trace] = storage.trace_trees

    assert trace.name == "search-query"
    assert trace.project_name == "mit-learn"
    assert trace.thread_id == "thread-1"
    assert sorted(trace.tags or []) == [
        "app:mit-learn",
        "env:ci",
        "use_case:search-query",
    ]
    assert trace.metadata is not None
    assert {k: trace.metadata[k] for k in ("use_case", "user_id", "run_id")} == {
        "use_case": "search-query",
        "user_id": "user-1",
        "run_id": "run-1",
    }
    assert trace.input == {"texts": ["a", "b"]}
    assert trace.output == {
        "model": MODEL,
        "dimensions": 3,
        "embeddings": 2,
        "input_tokens": 7,
    }


def test_the_span_lands_in_the_root_trace_with_the_reported_usage(
    exported: InMemorySpanExporter,
) -> None:
    with opik.record_traces_locally() as storage:
        embed_two()
        [trace] = storage.trace_trees
    tracing.flush()

    [span] = exported.get_finished_spans()
    attributes = span.attributes or {}
    assert span.name == "embeddings text-embedding-3-small"
    assert attributes["opik.trace_id"] == trace.id
    assert attributes[PROJECT_ATTRIBUTE] == "mit-learn"
    assert attributes["gen_ai.operation.name"] == "embeddings"
    assert attributes["gen_ai.provider.name"] == "openai"
    assert attributes["gen_ai.request.model"] == "text-embedding-3-small"
    assert attributes["gen_ai.response.model"] == "text-embedding-3-small"
    assert attributes["gen_ai.usage.input_tokens"] == 7
    assert attributes["gen_ai.embeddings.dimension.count"] == 3
    assert attributes["inputs"] == '["a", "b"]'
    assert "embeddings" not in attributes


def test_a_cache_hit_records_no_usage(exported: InMemorySpanExporter) -> None:
    with opik.record_traces_locally() as storage:
        embed_two(input_tokens=0)
        [trace] = storage.trace_trees
    tracing.flush()

    [span] = exported.get_finished_spans()
    assert "gen_ai.usage.input_tokens" not in (span.attributes or {})
    assert (trace.output or {})["input_tokens"] == 0


def test_without_content_neither_the_texts_nor_the_output_are_recorded(
    exported: InMemorySpanExporter,
) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(include_content=False))

    with opik.record_traces_locally() as storage:
        embed_two()
        [trace] = storage.trace_trees
    tracing.flush()

    assert not trace.input
    assert not trace.output
    [span] = exported.get_finished_spans()
    attributes = span.attributes or {}
    assert "inputs" not in attributes
    assert attributes["gen_ai.usage.input_tokens"] == 7


def test_the_callers_exception_passes_through_and_is_recorded(
    exported: InMemorySpanExporter,
) -> None:
    class ClientError(Exception):
        pass

    error = ClientError("rate limited")
    with opik.record_traces_locally() as storage:
        with pytest.raises(ClientError) as raised, traced_embedding(SEARCH, ["a"]):
            raise error
        [trace] = storage.trace_trees
    tracing.flush()

    assert raised.value is error
    assert trace.error_info is not None
    assert trace.error_info["exception_type"] == "ClientError"
    [span] = exported.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR


def test_a_block_that_records_nothing_raises() -> None:
    with (
        pytest.raises(ValueError, match="without calling record"),
        traced_embedding(SEARCH, ["a"]),
    ):
        pass


def test_no_texts_is_refused() -> None:
    with (
        pytest.raises(ValueError, match="at least one text"),
        traced_embedding(SEARCH, []),
    ):
        pass


def test_a_request_inside_a_tracked_function_is_a_span_of_its_trace(
    exported: InMemorySpanExporter,
) -> None:
    @opik.track(project_name="mit-learn-search")
    def handler() -> None:
        embed_two()

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees
        [root] = storage.span_trees
    tracing.flush()

    assert trace.name == "handler"
    assert "use_case:search-query" in (trace.tags or [])
    assert [span.name for span in root.spans] == ["search-query"]
    [span] = exported.get_finished_spans()
    assert (span.attributes or {})[PROJECT_ATTRIBUTE] == "mit-learn-search"


def test_it_works_around_an_awaited_call(exported: InMemorySpanExporter) -> None:
    async def request() -> None:
        with traced_embedding(SEARCH, ["a"]) as traced:
            await asyncio.sleep(0)
            traced.record([[1.0]], input_tokens=1)

    with opik.record_traces_locally() as storage:
        asyncio.run(request())
        [trace] = storage.trace_trees
    tracing.flush()

    [span] = exported.get_finished_spans()
    assert (span.attributes or {})["opik.trace_id"] == trace.id


def test_with_tracing_off_nothing_is_recorded(exported: InMemorySpanExporter) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False))

    with opik.record_traces_locally() as storage:
        embed_two()
        traces = storage.trace_trees

    assert traces == []
    assert exported.get_finished_spans() == ()


def test_with_tracing_disabled_at_runtime_nothing_is_recorded(
    exported: InMemorySpanExporter,
) -> None:
    with opik.record_traces_locally() as storage, tracing_disabled():
        embed_two()
        traces = storage.trace_trees
    tracing.flush()

    assert traces == []
    assert exported.get_finished_spans() == ()


def test_a_chat_model_is_refused() -> None:
    chat = UseCase(app="a", name="b", profile="batch", model="openai:gpt-4o-mini")

    with pytest.raises(ValueError, match="kind 'chat'"):
        embed_two(chat)


def test_fallback_models_are_refused() -> None:
    with_fallback = UseCase(
        app="a",
        name="b",
        profile="batch",
        model=MODEL,
        fallback_models=("azure:text-embedding-3-small",),
    )

    with pytest.raises(ValueError, match="not comparable"):
        embed_two(with_fallback)


def test_a_retired_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    retired = CatalogEntry(
        id="openai:text-embedding-ada-001",
        kind="embedding",
        dimensions=1024,
        status="retired",
        input_usd_per_mtok=Decimal("0.1"),
        output_usd_per_mtok=Decimal(0),
    )
    catalog = Catalog({**load_catalog().entries, retired.id: retired})
    monkeypatch.setattr(catalog_module, "load_catalog", lambda: catalog)

    with pytest.raises(ValueError, match="retired"):
        embed_two(UseCase(app="a", name="b", profile="batch", model=retired.id))


def test_an_unlisted_model_is_traced_tagged_and_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unlisted = UseCase(
        app="a", name="b", profile="batch", model="openai:text-embedding-ada-002"
    )

    with opik.record_traces_locally() as storage:
        embed_two(unlisted)
        [trace] = storage.trace_trees

    assert "model_status:uncataloged" in (trace.tags or [])
    assert "the model catalog does not list" in caplog.text
