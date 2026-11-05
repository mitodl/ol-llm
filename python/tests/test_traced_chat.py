import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Generator, Iterator
from datetime import date
from decimal import Decimal
from typing import Any

import ol_llm.catalog as catalog_module
import ol_llm.langchain as langchain_module
import opik
import pytest
from conftest import opik_context_is_empty
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from ol_llm import (
    Attribution,
    UseCase,
    check_chat_model,
    run_sync,
    traced_chat,
    tracing,
)
from ol_llm.catalog import Catalog, CatalogEntry, load_catalog
from ol_llm.langchain import CUT_OFF_METADATA_KEY, GRAPH_METADATA_KEY
from ol_llm.models import build_agent, override_models
from ol_llm.testing import tracing_disabled
from ol_llm.tracing import TracingConfig, configure_tracing
from opik.message_processing.emulation.models import SpanModel, TraceModel
from pydantic_ai.models.test import TestModel

TUTOR = UseCase(
    app="learn-ai", name="tutor", profile="request", model="openai:gpt-4o-mini"
)
ATTRIBUTION = Attribution(user_id="user-1", thread_id="thread-1")
USAGE = UsageMetadata(
    input_tokens=12,
    output_tokens=5,
    total_tokens=17,
    input_token_details={"cache_read": 4},
    output_token_details={"reasoning": 2},
)
OPENAI_USAGE = {
    "prompt_tokens": 12,
    "completion_tokens": 5,
    "total_tokens": 17,
    "original_usage.prompt_tokens": 12,
    "original_usage.completion_tokens": 5,
    "original_usage.total_tokens": 17,
    "original_usage.prompt_tokens_details.cached_tokens": 4,
    "original_usage.completion_tokens_details.reasoning_tokens": 2,
}


class ProviderError(Exception):
    pass


class Replies(BaseChatModel):
    """A chat model that answers ``reply`` and reports ``usage``, as ChatLiteLLM does.

    Its serialized form names no provider, so Opik's own extractors find no usage.
    """

    reply: str = "pong"
    usage: UsageMetadata | None = USAGE
    error: str | None = None

    @property
    def _llm_type(self) -> str:
        return "replies"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> ChatResult:
        if self.error is not None:
            raise ProviderError(self.error)
        message = AIMessage(content=self.reply, usage_metadata=self.usage)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Iterator[ChatGenerationChunk]:
        for word in self.reply.split():
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=self.usage)
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop, **kwargs):
            await asyncio.sleep(0)
            yield chunk


class Choices(Replies):
    """A chat model that answers with two choices, each carrying the response's usage.

    As langchain-openai and ChatLiteLLM do for a request with ``n=2``.
    """

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=reply, usage_metadata=self.usage)
                )
                for reply in ("pong", "pang")
            ]
        )


class Stalls(Replies):
    """A chat model that streams its first word, then waits until it is cancelled."""

    async def _astream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content=self.reply))
        await asyncio.Event().wait()


@pytest.fixture(autouse=True)
def fresh_catalog_checks() -> Generator[None]:
    """Check each test's use cases against the catalog it patches in."""
    check_chat_model.cache_clear()
    yield
    check_chat_model.cache_clear()


def agent_graph(model: BaseChatModel) -> RunnableLambda[str, str]:
    """A one-node agent: the node calls the model, as a LangGraph node does."""

    def agent(message: str, config: RunnableConfig) -> str:
        return str(model.invoke(message, config).content)

    return RunnableLambda(agent, name="agent")


def chat(
    model: BaseChatModel | None = None,
    use_case: UseCase = TUTOR,
    **kwargs: Any,  # noqa: ANN401
) -> str:
    with traced_chat(use_case, "ping", attribution=ATTRIBUTION, **kwargs) as traced:
        reply = agent_graph(model or Replies()).invoke(
            "ping", {"callbacks": [*traced.callbacks]}
        )
        traced.record(reply)
    return reply


def streaming_graph(model: BaseChatModel) -> Runnable[str, BaseMessage]:
    """A graph whose root run streams the model's reply, as ``astream`` on a graph."""
    return RunnableLambda(lambda message: message, name="prepare") | model


def spans(trace: TraceModel) -> list[SpanModel]:
    """Return every span of the trace, depth first."""
    found: list[SpanModel] = []

    def walk(span: SpanModel) -> None:
        found.append(span)
        for child in span.spans:
            walk(child)

    for span in trace.spans:
        walk(span)
    return found


def llm_span(trace: TraceModel) -> SpanModel:
    [span] = [span for span in spans(trace) if span.type == "llm"]
    return span


def test_the_root_trace_matches_runs() -> None:
    with (
        override_models(TestModel(custom_output_text="pong")),
        opik.record_traces_locally() as storage,
    ):
        run_sync(build_agent(TUTOR), "ping", attribution=ATTRIBUTION)
        chat()
        ran, traced = storage.trace_trees

    def fields(trace: TraceModel) -> tuple[object, ...]:
        names = ("name", "project_name", "thread_id", "tags", "metadata", "input")
        return tuple(getattr(trace, name) for name in names)

    assert fields(traced) == fields(ran)
    assert traced.environment == ran.environment == "ci"
    assert traced.output == ran.output == {"output": "pong"}


def test_the_runs_are_spans_of_the_root_with_the_catalogs_model_and_usage() -> None:
    with opik.record_traces_locally() as storage:
        chat()
        [trace] = storage.trace_trees

    [root] = trace.spans
    assert root.name == "tutor"
    [agent] = root.spans
    assert agent.name == "agent"
    [model] = agent.spans
    assert model.type == "llm"
    assert (model.provider, model.model) == ("openai", "gpt-4o-mini")
    assert model.usage == OPENAI_USAGE
    assert {span.project_name for span in spans(trace)} == {"learn-ai"}
    assert {span.environment for span in spans(trace)} == {"ci"}


def test_a_streamed_reply_reports_its_usage() -> None:
    async def stream() -> str:
        with traced_chat(TUTOR, "ping") as traced:
            config: RunnableConfig = {"callbacks": [*traced.callbacks]}
            words = [str(c.content) async for c in Replies().astream("ping", config)]
        return "".join(words)

    with opik.record_traces_locally() as storage:
        assert asyncio.run(stream()) == "pong"
        [trace] = storage.trace_trees

    assert llm_span(trace).usage == OPENAI_USAGE


def test_a_reply_without_usage_is_priced_by_nothing() -> None:
    with opik.record_traces_locally() as storage:
        chat(Replies(usage=None))
        [trace] = storage.trace_trees

    span = llm_span(trace)
    assert (span.provider, span.model) == ("openai", "gpt-4o-mini")
    assert not span.usage


def test_the_span_names_the_model_as_opiks_price_table_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = CatalogEntry(
        id="azure:gpt-4o-deployment",
        kind="chat",
        opik_provider="azure",
        opik_model="gpt-4o",
        input_usd_per_mtok=Decimal("2.5"),
        output_usd_per_mtok=Decimal(10),
    )
    catalog = Catalog({**load_catalog().entries, entry.id: entry})
    monkeypatch.setattr(catalog_module, "load_catalog", lambda: catalog)
    monkeypatch.setattr(langchain_module, "load_catalog", lambda: catalog)
    azure = UseCase(app="learn-ai", name="tutor", profile="request", model=entry.id)

    with opik.record_traces_locally() as storage:
        chat(use_case=azure)
        [trace] = storage.trace_trees

    span = llm_span(trace)
    assert (span.provider, span.model) == ("azure", "gpt-4o")


def test_callbacks_called_from_another_thread_attach_to_the_root() -> None:
    model = Replies()
    with opik.record_traces_locally() as storage:
        with traced_chat(TUTOR, "ping") as traced:
            # A thread does not inherit the caller's context, so Opik's tracer
            # sees no current trace there.
            worker = threading.Thread(
                target=model.invoke, args=("ping", {"callbacks": [*traced.callbacks]})
            )
            worker.start()
            worker.join()
        traces = storage.trace_trees

    [trace] = traces
    [root] = trace.spans
    assert [span.type for span in root.spans] == ["llm"]


def test_model_level_callbacks_attach_to_the_root() -> None:
    model = Replies()
    with opik.record_traces_locally() as storage:
        with traced_chat(TUTOR, "ping") as traced:
            model.callbacks = [*traced.callbacks]
            agent_graph(model).invoke("ping")
        [trace] = storage.trace_trees

    [root] = trace.spans
    [span] = root.spans
    assert span.type == "llm"
    assert span.usage == OPENAI_USAGE


def test_metadata_goes_on_the_runs_and_the_graph_on_the_trace() -> None:
    graph = agent_graph(Replies()).get_graph()

    with opik.record_traces_locally() as storage:
        chat(metadata={"edx_module_id": "block-1"}, graph=graph)
        [trace] = storage.trace_trees

    assert trace.metadata is not None
    assert trace.metadata[GRAPH_METADATA_KEY] == {
        "format": "mermaid",
        "data": graph.draw_mermaid(),
    }
    assert "edx_module_id" not in trace.metadata
    [root] = trace.spans
    [agent] = root.spans
    assert (agent.metadata or {})["edx_module_id"] == "block-1"


def test_a_chat_inside_a_tracked_function_is_a_span_of_its_trace() -> None:
    @opik.track(project_name="learn-ai-consumer")
    def handler() -> None:
        chat()

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees

    assert trace.name == "handler"
    assert "use_case:tutor" in (trace.tags or [])
    [outer] = trace.spans
    [root] = outer.spans
    assert root.name == "tutor"
    assert {span.project_name for span in spans(trace)} == {"learn-ai-consumer"}


def test_a_failed_run_records_its_error_and_the_exception_passes_through() -> None:
    with opik.record_traces_locally() as storage:
        with pytest.raises(ProviderError) as raised:
            chat(Replies(error="the provider refused"))
        [trace] = storage.trace_trees

    assert str(raised.value) == "the provider refused"
    assert trace.error_info is not None
    assert trace.error_info["exception_type"] == "ProviderError"
    span = llm_span(trace)
    assert span.error_info is not None
    assert span.error_info["exception_type"] == "ProviderError"
    assert "the provider refused" in span.error_info.get("message", "")


def test_without_content_no_input_output_or_error_message_is_recorded() -> None:
    tracing.reset()
    configure_tracing(TracingConfig(include_content=False))

    with opik.record_traces_locally() as storage:
        chat()
        with pytest.raises(ProviderError):
            chat(Replies(error="the provider echoed: secret text"))
        ok, failed = storage.trace_trees

    for span in [*spans(ok), *spans(failed)]:
        assert not span.input, span.name
        assert not span.output, span.name
    assert not ok.input
    assert not ok.output
    assert llm_span(ok).usage == OPENAI_USAGE
    errors = [span.error_info for span in spans(failed) if span.error_info]
    assert errors
    for error in errors:
        assert error["exception_type"] == "ProviderError"
        assert error.get("message") is None
    assert "secret text" not in repr(failed)


def test_with_tracing_off_nothing_is_recorded() -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False))

    with opik.record_traces_locally() as storage:
        with traced_chat(TUTOR, "ping") as traced:
            callbacks = traced.callbacks
        traces = storage.trace_trees

    assert callbacks == ()
    assert traces == []


def test_with_tracing_disabled_at_runtime_nothing_is_recorded() -> None:
    with opik.record_traces_locally() as storage, tracing_disabled():
        chat()
        traces = storage.trace_trees

    assert traces == []


def test_an_unlisted_model_is_traced_tagged_and_warned_about_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unlisted = UseCase(
        app="learn-ai", name="tutor", profile="request", model="openai:gpt-9"
    )

    with opik.record_traces_locally() as storage, caplog.at_level(logging.WARNING):
        assert check_chat_model(unlisted) == ("model_status:uncataloged",)
        chat(use_case=unlisted)
        [trace] = storage.trace_trees

    assert "model_status:uncataloged" in (trace.tags or [])
    assert caplog.text.count("the model catalog does not list") == 1
    span = llm_span(trace)
    assert (span.provider, span.model) == ("openai", "gpt-9")


def test_a_deprecated_model_is_tagged(monkeypatch: pytest.MonkeyPatch) -> None:
    deprecated = CatalogEntry(
        id="openai:gpt-4-turbo",
        kind="chat",
        status="deprecated",
        retires_on=date(2027, 1, 1),
        input_usd_per_mtok=Decimal(10),
        output_usd_per_mtok=Decimal(30),
    )
    catalog = Catalog({**load_catalog().entries, deprecated.id: deprecated})
    monkeypatch.setattr(catalog_module, "load_catalog", lambda: catalog)
    use_case = UseCase(app="a", name="b", profile="request", model=deprecated.id)

    with opik.record_traces_locally() as storage:
        chat(use_case=use_case)
        [trace] = storage.trace_trees

    assert "model_status:deprecated" in (trace.tags or [])


def test_a_retired_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    retired = CatalogEntry(
        id="openai:gpt-3.5-turbo",
        kind="chat",
        status="retired",
        input_usd_per_mtok=Decimal("0.5"),
        output_usd_per_mtok=Decimal("1.5"),
    )
    catalog = Catalog({**load_catalog().entries, retired.id: retired})
    monkeypatch.setattr(catalog_module, "load_catalog", lambda: catalog)
    use_case = UseCase(app="a", name="b", profile="request", model=retired.id)

    with pytest.raises(ValueError, match="retired"):
        check_chat_model(use_case)
    with pytest.raises(ValueError, match="retired"):
        chat(use_case=use_case)


def test_an_embedding_model_is_refused() -> None:
    embedding = UseCase(
        app="a", name="b", profile="request", model="openai:text-embedding-3-small"
    )

    with pytest.raises(ValueError, match="kind 'embedding'"):
        chat(use_case=embedding)


def test_a_chat_reports_one_responses_usage_for_several_choices() -> None:
    with opik.record_traces_locally() as storage:
        chat(Choices())
        [trace] = storage.trace_trees

    assert llm_span(trace).usage == OPENAI_USAGE


def assert_ended(trace: TraceModel) -> None:
    assert trace.end_time is not None
    assert [span.name for span in spans(trace) if span.end_time is None] == []


def assert_cut_off(span: SpanModel) -> None:
    """The block cut the span's run off: it names its model and blames nothing."""
    assert (span.metadata or {}).get(CUT_OFF_METADATA_KEY) is True
    assert span.error_info is None
    assert not span.usage


def test_a_stream_left_with_break_ends_with_the_block() -> None:
    async def chat_twice() -> bool:
        with traced_chat(TUTOR, "ping") as traced:
            config: RunnableConfig = {"callbacks": [*traced.callbacks]}
            async for _ in streaming_graph(Replies(reply="a b c")).astream(
                "ping", config
            ):
                break
            traced.record("a")
        left_empty = opik_context_is_empty()
        chat()
        return left_empty

    with opik.record_traces_locally() as storage:
        assert asyncio.run(chat_twice())
        first, second = storage.trace_trees

    assert_ended(first)
    assert first.output == {"output": "a"}
    [root] = first.spans
    assert root.output == {"output": "a"}
    [sequence] = root.spans
    assert sequence.output is None
    assert [span.name for span in sequence.spans] == ["prepare", "Replies"]
    assert_cut_off(sequence)
    model = llm_span(first)
    assert_cut_off(model)
    assert (model.provider, model.model) == ("openai", "gpt-4o-mini")
    # The abandoned stream is closed after the trace ended, and not recorded.
    assert [span.name for span in spans(first) if span.error_info] == []
    assert_ended(second)
    assert [span.name for span in second.spans] == ["tutor"]


def test_a_stream_left_with_an_exception_ends_with_the_block_recording_it() -> None:
    closed = "socket closed"

    async def fail() -> None:
        with traced_chat(TUTOR, "ping") as traced:
            config: RunnableConfig = {"callbacks": [*traced.callbacks]}
            async for _ in streaming_graph(Replies()).astream("ping", config):
                raise ProviderError(closed)

    async def fail_then_chat() -> bool:
        with pytest.raises(ProviderError, match=closed):
            await fail()
        left_empty = opik_context_is_empty()
        chat()
        return left_empty

    with opik.record_traces_locally() as storage:
        assert asyncio.run(fail_then_chat())
        failed, second = storage.trace_trees

    assert_ended(failed)
    assert failed.error_info is not None
    assert failed.error_info["exception_type"] == "ProviderError"
    # The caller's error is the root's; the runs it cut off are not blamed for it.
    errors = {span.name: span.error_info for span in spans(failed) if span.error_info}
    assert set(errors) == {"tutor"}
    assert errors["tutor"]["exception_type"] == "ProviderError"
    model = llm_span(failed)
    assert_cut_off(model)
    assert (model.provider, model.model) == ("openai", "gpt-4o-mini")
    assert_ended(second)
    assert [span.name for span in second.spans] == ["tutor"]


def test_a_cancelled_stream_ends_with_the_block() -> None:
    left_empty: list[bool] = []

    async def cancel() -> None:
        entered = asyncio.Event()

        async def streaming() -> None:
            try:
                with traced_chat(TUTOR, "ping") as traced:
                    config: RunnableConfig = {"callbacks": [*traced.callbacks]}
                    async for _ in streaming_graph(Stalls()).astream("ping", config):
                        entered.set()
            except asyncio.CancelledError:
                left_empty.append(opik_context_is_empty())
                raise

        task = asyncio.create_task(streaming())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with opik.record_traces_locally() as storage:
        asyncio.run(cancel())
        [trace] = storage.trace_trees

    assert left_empty == [True]
    assert_ended(trace)
    # As a cancelled run's: the root trace closes without an error.
    assert trace.error_info is None
    # LangChain records the cancellation on the graph's run, which it reached; the
    # chat model's run it never ended is cut off.
    errors = {
        span.name: span.error_info["exception_type"]
        for span in spans(trace)
        if span.error_info
    }
    assert errors == {"RunnableSequence": "CancelledError"}
    model = llm_span(trace)
    assert_cut_off(model)
    assert (model.provider, model.model) == ("openai", "gpt-4o-mini")


def test_without_content_an_error_that_is_no_exception_keeps_its_type_and_frames() -> (
    None
):
    tracing.reset()
    configure_tracing(TracingConfig(include_content=False))

    def refuse(message: str) -> str:
        echoed = f"the provider echoed: secret text {message}"
        raise ProviderError(echoed)

    def agent(message: str) -> str:
        # A cancellation chained to an error whose message echoes the prompt.
        try:
            return refuse(message)
        except ProviderError as exc:
            raise asyncio.CancelledError from exc

    def cancelled_chat() -> None:
        with traced_chat(TUTOR, "ping") as traced:
            RunnableLambda(agent, name="agent").invoke(
                "ping", {"callbacks": [*traced.callbacks]}
            )

    with opik.record_traces_locally() as storage:
        with pytest.raises(asyncio.CancelledError):
            cancelled_chat()
        [trace] = storage.trace_trees

    [error] = [span.error_info for span in spans(trace) if span.error_info]
    assert error["exception_type"] == "CancelledError"
    assert error.get("message") is None
    assert error["traceback"]
    assert "secret text" not in repr(trace)


def test_a_run_another_thread_starts_as_the_block_closes_still_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker's run passes the tracer's admission check, then the block closes
    # before the run's span joins the tracer's started spans.
    admitted = threading.Event()
    closed = threading.Event()
    emit = langchain_module.ChatTracer._emit_start_span  # noqa: SLF001

    def start_late(tracer: langchain_module.ChatTracer, span: Any) -> None:  # noqa: ANN401
        if span.type == "llm":
            admitted.set()
            assert closed.wait(5)
        emit(tracer, span)

    monkeypatch.setattr(langchain_module.ChatTracer, "_emit_start_span", start_late)
    with opik.record_traces_locally() as storage:
        with traced_chat(TUTOR, "ping") as traced:
            config: RunnableConfig = {"callbacks": [*traced.callbacks]}
            worker = threading.Thread(target=Replies().invoke, args=("ping", config))
            worker.start()
            assert admitted.wait(5)
        closed.set()
        worker.join(5)
        [trace] = storage.trace_trees

    assert_ended(trace)
    model = llm_span(trace)
    assert_cut_off(model)
    assert (model.provider, model.model) == ("openai", "gpt-4o-mini")
