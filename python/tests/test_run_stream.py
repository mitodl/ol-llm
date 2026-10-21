import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx2
import opik
import pytest
from conftest import PromptLibrary
from ol_llm import (
    Attribution,
    LLMInvalidOutput,
    LLMRejected,
    LLMResult,
    LLMUnavailable,
    ManagedPrompt,
    PostProcessed,
    TextTemplate,
    UseCase,
    get_prompt,
    run_stream,
    run_sync,
    tracing,
)
from ol_llm.models import build_agent, override_models
from ol_llm.testing import tracing_disabled
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opik import opik_context
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

PRIMARY = "openai:gpt-4o-mini"
FALLBACK = "bedrock:global.anthropic.claude-haiku-4-5-20251001-v1:0"

SUMMARY = UseCase(app="mit-learn", name="summary", profile="request", model=PRIMARY)
SUMMARY_WITH_FALLBACK = UseCase(
    app="mit-learn",
    name="summary",
    profile="request",
    model=PRIMARY,
    fallback_models=(FALLBACK,),
)
ATTRIBUTION = Attribution(user_id="user-1", thread_id="thread-1", run_id="run-1")
CHUNKS = ("a ", "short ", "summary")


class Summary(BaseModel):
    text: str


class Scripted:
    """Raises each failure in turn when a request or stream opens, then answers."""

    def __init__(self, *failures: Exception) -> None:
        self.failures = list(failures)
        self.calls = 0

    def _next(self) -> None:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)

    def __call__(self, _m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        self._next()
        return ModelResponse(parts=[TextPart("".join(CHUNKS))])

    async def stream(self, _m: list[ModelMessage], _i: AgentInfo) -> AsyncIterator[str]:
        self._next()
        for chunk in CHUNKS:
            yield chunk


def scripted(*failures: Exception) -> tuple[Scripted, FunctionModel]:
    script = Scripted(*failures)
    return script, FunctionModel(script, stream_function=script.stream)


def failing(status: int) -> FunctionModel:
    def fail(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status, "gpt-4o-mini", body="provider said no")

    async def stream(_m: list[ModelMessage], _i: AgentInfo) -> AsyncIterator[str]:
        raise ModelHTTPError(status, "gpt-4o-mini", body="provider said no")
        yield ""  # pragma: no cover

    return FunctionModel(fail, stream_function=stream)


def agent_for(use_case: UseCase, model: object, **kwargs: object) -> Agent[None, str]:
    with override_models(model):  # type: ignore[arg-type]
        return build_agent(use_case, **kwargs)  # type: ignore[call-overload,no-any-return]


def streamed(
    agent: Agent[None, str],
    read: Callable[[object], Awaitable[object]] | None = None,
    **kwargs: object,
) -> LLMResult[str]:
    """Open the stream, read it with ``read`` if given, and return the result."""

    async def consume() -> LLMResult[str]:
        async with run_stream(agent, "summarize", **kwargs) as stream:  # type: ignore[arg-type]
            if read is not None:
                await read(stream)
            return await stream.result()

    return asyncio.run(consume())


async def read_text(stream: object) -> list[str]:
    return [chunk async for chunk in stream.stream_text()]  # type: ignore[attr-defined]


def test_the_result_matches_runs_for_the_same_input() -> None:
    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    ran = run_sync(agent, "summarize", attribution=ATTRIBUTION)
    result = streamed(agent, read_text, attribution=ATTRIBUTION)

    assert result.output == ran.output == "a short summary"
    assert (result.model, result.degraded, result.degraded_reasons) == (
        ran.model,
        ran.degraded,
        ran.degraded_reasons,
    )
    assert (result.prompt_name, result.prompt_version) == (None, None)
    assert result.usage.requests == ran.usage.requests == 1
    assert [type(m) for m in result.new_messages] == [type(m) for m in ran.new_messages]


def test_stream_text_yields_the_text_as_it_arrives() -> None:
    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    async def consume() -> tuple[list[str], list[str]]:
        async with run_stream(agent, "summarize") as stream:
            deltas = [chunk async for chunk in stream.stream_text()]
        async with run_stream(agent, "summarize") as stream:
            snapshots = [text async for text in stream.stream_text(delta=False)]
        return deltas, snapshots

    deltas, snapshots = asyncio.run(consume())

    assert "".join(deltas) == "a short summary"
    assert snapshots[-1] == "a short summary"


def test_stream_output_yields_the_structured_output() -> None:
    async def stream(
        _m: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[dict[int, DeltaToolCall]]:
        name = info.output_tools[0].name
        yield {0: DeltaToolCall(name=name, json_args='{"text": "sho')}
        yield {0: DeltaToolCall(json_args='rt"}')}

    agent = agent_for(
        SUMMARY, FunctionModel(stream_function=stream), output_type=Summary
    )

    async def consume() -> tuple[list[Summary], LLMResult[Summary]]:
        async with run_stream(agent, "summarize") as llm_stream:
            outputs = [output async for output in llm_stream.stream_output()]
            return outputs, await llm_stream.result()

    outputs, result = asyncio.run(consume())

    assert outputs[-1] == Summary(text="short")
    assert result.output == Summary(text="short")


def test_invalid_output_is_not_retried_and_raises_invalid_output() -> None:
    # PydanticAI's run_stream keeps the first output that matches the output type,
    # so the use case's output_retries do not apply (spec §6.6).
    calls = 0

    async def stream(
        _m: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[dict[int, DeltaToolCall]]:
        nonlocal calls
        calls += 1
        name = info.output_tools[0].name
        yield {0: DeltaToolCall(name=name, json_args='{"title": "no text"}')}

    agent = agent_for(
        SUMMARY, FunctionModel(stream_function=stream), output_type=Summary
    )

    async def consume() -> None:
        async with run_stream(agent, "summarize") as llm_stream:
            await llm_stream.result()

    assert SUMMARY.output_retries == 1
    with pytest.raises(LLMInvalidOutput) as caught:
        asyncio.run(consume())

    assert calls == 1
    assert caught.value.model == PRIMARY
    assert isinstance(caught.value.__cause__, UnexpectedModelBehavior)


def test_the_root_trace_matches_runs(exported: InMemorySpanExporter) -> None:
    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    with opik.record_traces_locally() as storage:
        run_sync(agent, "summarize", attribution=ATTRIBUTION)
        streamed(agent, read_text, attribution=ATTRIBUTION)
        ran, trace = storage.trace_trees
    tracing.flush()

    for field in ("name", "project_name", "thread_id", "input", "output", "metadata"):
        assert getattr(trace, field) == getattr(ran, field), field
    assert sorted(trace.tags or []) == sorted(ran.tags or [])
    assert trace.output == {"output": "a short summary"}
    spans = [
        span
        for span in exported.get_finished_spans()
        if (span.attributes or {}).get("opik.trace_id") == trace.id
    ]
    assert {span.name for span in spans} >= {"invoke_agent summary"}
    assert any(span.name.startswith("chat ") for span in spans)


def test_the_trace_closes_with_the_block() -> None:
    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    async def consume() -> tuple[bool, bool]:
        async with run_stream(agent, "summarize") as stream:
            inside = opik_context.get_current_trace_data() is not None
            await stream.result()
        return inside, opik_context.get_current_trace_data() is None

    with opik.record_traces_locally():
        inside, closed = asyncio.run(consume())

    assert inside
    assert closed


def test_a_stream_is_retried_until_it_opens(sleeps: list[float]) -> None:
    script, model = scripted(
        ModelHTTPError(503, "gpt-4o-mini"), ModelHTTPError(502, "gpt-4o-mini")
    )

    result = streamed(agent_for(SUMMARY, model), read_text)

    assert result.output == "a short summary"
    assert script.calls == 3
    assert len(sleeps) == 2


def test_a_failure_after_the_first_chunk_reaches_the_reader() -> None:
    calls: list[int] = []

    async def stream(_m: list[ModelMessage], _i: AgentInfo) -> AsyncIterator[str]:
        calls.append(1)
        yield "partial"
        msg = "stalled"
        raise httpx2.ReadTimeout(msg)

    agent = agent_for(SUMMARY, FunctionModel(stream_function=stream))

    inside: list[Exception] = []

    async def consume() -> None:
        async with run_stream(agent, "summarize") as llm_stream:
            try:
                await read_text(llm_stream)
            except LLMUnavailable as exc:
                inside.append(exc)
                raise

    with opik.record_traces_locally() as storage:
        with pytest.raises(LLMUnavailable) as caught:
            asyncio.run(consume())
        [trace] = storage.trace_trees

    assert calls == [1]
    assert inside == [caught.value]
    assert isinstance(caught.value.__cause__, httpx2.ReadTimeout)
    assert trace.error_info is not None
    assert trace.error_info["exception_type"] == "LLMUnavailable"


def test_a_failure_before_the_first_chunk_raises_on_entry() -> None:
    agent = agent_for(SUMMARY, failing(400))

    with pytest.raises(LLMRejected) as caught:
        streamed(agent)

    assert isinstance(caught.value.__cause__, ModelHTTPError)


@pytest.mark.parametrize(
    "error",
    [UnexpectedModelBehavior("from another call"), httpx2.ReadTimeout("elsewhere")],
)
def test_an_exception_from_the_callers_block_passes_through_unchanged(
    error: Exception,
) -> None:
    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    async def consume() -> None:
        async with run_stream(agent, "summarize") as stream:
            await read_text(stream)
            raise error

    with opik.record_traces_locally() as storage:
        with pytest.raises(type(error)) as caught:
            asyncio.run(consume())
        [trace] = storage.trace_trees

    assert caught.value is error
    assert trace.error_info is not None
    assert trace.error_info["exception_type"] == type(error).__name__


def test_a_fallback_answer_is_degraded_and_postprocessed() -> None:
    def partial(output: str) -> PostProcessed[str]:
        return PostProcessed(
            output=output.upper(), degraded_reasons=("partial_output",)
        )

    _, fallback = scripted()
    agent = agent_for(
        SUMMARY_WITH_FALLBACK, {PRIMARY: failing(503), FALLBACK: fallback}
    )

    with opik.record_traces_locally() as storage:
        result = streamed(agent, read_text, postprocess=partial)
        [trace] = storage.trace_trees

    assert result.output == "A SHORT SUMMARY"
    assert result.model == FALLBACK
    assert result.degraded_reasons == ("fallback_model", "partial_output")
    assert trace.output == {"output": "A SHORT SUMMARY"}
    assert trace.metadata is not None
    assert trace.metadata["degraded_reason"] == "fallback_model,partial_output"


def test_a_read_stream_is_finished_without_a_result_call() -> None:
    _, fallback = scripted()
    agent = agent_for(
        SUMMARY_WITH_FALLBACK, {PRIMARY: failing(503), FALLBACK: fallback}
    )

    async def consume() -> None:
        async with run_stream(agent, "summarize") as stream:
            await read_text(stream)

    with opik.record_traces_locally() as storage:
        asyncio.run(consume())
        [trace] = storage.trace_trees

    assert trace.output == {"output": "a short summary"}
    assert trace.metadata is not None
    assert trace.metadata["degraded_reason"] == "fallback_model"


@pytest.mark.parametrize("call_result", [True, False])
def test_an_exception_from_postprocess_reaches_the_caller_unchanged(
    call_result: bool,  # noqa: FBT001
) -> None:
    error = UnexpectedModelBehavior("the caller's own check")

    def broken(_output: str) -> PostProcessed[str]:
        raise error

    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    async def consume() -> None:
        async with run_stream(agent, "summarize", postprocess=broken) as stream:
            await read_text(stream)
            if call_result:
                await stream.result()

    with pytest.raises(UnexpectedModelBehavior) as caught:
        asyncio.run(consume())

    assert caught.value is error


def test_a_stream_inside_a_tracked_function_is_a_span_of_its_trace(
    exported: InMemorySpanExporter,
) -> None:
    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    @opik.track(project_name="ol-data-platform")
    async def handler() -> str:
        async with run_stream(agent, "hi", attribution=ATTRIBUTION) as stream:
            return "".join(await read_text(stream))

    with opik.record_traces_locally() as storage:
        asyncio.run(handler())
        [trace] = storage.trace_trees
        [root] = storage.span_trees
    tracing.flush()

    assert trace.name == "handler"
    assert trace.thread_id == "thread-1"
    assert "use_case:summary" in (trace.tags or [])
    assert [span.name for span in root.spans] == ["summary"]
    spans = exported.get_finished_spans()
    assert {(s.attributes or {}).get("opik.trace_id") for s in spans} == {trace.id}


def test_a_stream_with_tracing_disabled_records_nothing(
    exported: InMemorySpanExporter,
) -> None:
    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    with opik.record_traces_locally() as storage, tracing_disabled():
        result = streamed(agent, read_text)
        traces = storage.trace_trees
    tracing.flush()

    assert result.output == "a short summary"
    assert traces == []
    assert exported.get_finished_spans() == ()


def test_an_agent_not_from_build_agent_is_refused() -> None:
    with pytest.raises(TypeError, match="build_model"):
        streamed(Agent(TestModel()))


def test_a_managed_prompt_is_linked_to_the_stream(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "Summarize.", environments=("production",))
    prompt = get_prompt(
        "summary",
        project_name="mit-learn",
        default=TextTemplate(template="Summarize."),
        environment="production",
    )
    _, model = scripted()
    agent = agent_for(SUMMARY, model, instructions=prompt.format())

    with opik.record_traces_locally() as storage:
        result = streamed(agent, read_text, prompt=prompt)
        [trace] = storage.trace_trees

    assert (result.prompt_name, result.prompt_version) == ("summary", "v1")
    assert trace.metadata is not None
    assert trace.metadata["prompt_name"] == "summary"
    [linked] = trace.metadata["opik_prompts"]
    assert linked["version"]["version_number"] == "v1"


def test_an_embedded_prompt_marks_the_stream_degraded() -> None:
    prompt = ManagedPrompt(
        name="summary",
        template=TextTemplate(template="Summarize.", version="v3"),
        source="embedded",
        degraded_reasons=("embedded_prompt",),
    )
    _, model = scripted()
    agent = agent_for(SUMMARY, model)

    with opik.record_traces_locally() as storage:
        result = streamed(agent, read_text, prompt=prompt)
        [trace] = storage.trace_trees

    assert (result.prompt_name, result.prompt_version) == ("summary", "v3")
    assert result.degraded_reasons == ("embedded_prompt",)
    assert trace.metadata is not None
    assert trace.metadata["degraded_reason"] == "embedded_prompt"
