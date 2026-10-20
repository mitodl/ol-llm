import asyncio
import gc
import threading
import weakref
from collections.abc import Callable

import opik
import pytest
from ol_llm import (
    Attribution,
    LLMInvalidOutput,
    LLMRateLimited,
    PostProcessed,
    Settings,
    UseCase,
    complete,
    complete_sync,
    configure,
    run_sync,
    tracing,
)
from ol_llm.models import build_agent, override_models, shared_model
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel
from pydantic_ai import ToolOutput
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

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


class Summary(BaseModel):
    text: str


def replying(text: str, seen: list[AgentInfo] | None = None) -> FunctionModel:
    def reply(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.append(info)
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(reply)


def calling_output_tool(text: str, seen: list[AgentInfo]) -> FunctionModel:
    def reply(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        [tool] = info.output_tools
        return ModelResponse(parts=[ToolCallPart(tool.name, {"text": text})])

    return FunctionModel(reply)


def failing(status: int) -> FunctionModel:
    def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status, "gpt-4o-mini", body="provider said no")

    return FunctionModel(fail)


def complete_async(*args: object, **kwargs: object) -> object:
    return asyncio.run(complete(*args, **kwargs))  # type: ignore[call-overload]


ENTRIES = pytest.mark.parametrize(
    "entry", [complete_sync, complete_async], ids=["sync", "async"]
)


@ENTRIES
def test_complete_returns_the_output_and_the_answering_model(
    entry: Callable[..., object],
) -> None:
    with override_models(replying("a summary")):
        result = entry(SUMMARY, "summarize")

    assert result.output == "a summary"  # type: ignore[attr-defined]
    assert result.model == PRIMARY  # type: ignore[attr-defined]
    assert result.usage.requests == 1  # type: ignore[attr-defined]
    assert not result.degraded  # type: ignore[attr-defined]
    assert len(result.new_messages) == 2  # type: ignore[attr-defined]


def test_the_instructions_and_the_user_id_reach_the_model() -> None:
    seen: list[AgentInfo] = []

    with override_models(replying("ok", seen)):
        complete_sync(SUMMARY, "hi", instructions="Summarize.", attribution=ATTRIBUTION)

    assert seen[0].instructions == "Summarize."
    assert seen[0].model_settings is not None
    assert seen[0].model_settings.get("openai_user") == "user-1"


def test_structured_output_is_validated_into_the_schema() -> None:
    seen: list[AgentInfo] = []

    with override_models(calling_output_tool("short", seen)):
        result = complete_sync(SUMMARY, "summarize", output_type=Summary)

    assert result.output == Summary(text="short")


def test_a_tool_output_forces_a_call_to_the_named_tool() -> None:
    seen: list[AgentInfo] = []
    output = ToolOutput(Summary, name="record_summary", description="Record it.")

    with override_models(calling_output_tool("short", seen)):
        result = complete_sync(SUMMARY, "summarize", output_type=output)

    assert result.output == Summary(text="short")
    assert not seen[0].allow_text_output
    assert not seen[0].function_tools
    assert [(t.name, t.description) for t in seen[0].output_tools] == [
        ("record_summary", "Record it.")
    ]


def test_a_message_list_is_sent_as_the_conversation() -> None:
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt("first"),
        ModelResponse(parts=[TextPart("answer")]),
        ModelRequest.user_text_prompt("second"),
    ]
    sent: list[list[ModelMessage]] = []

    def reply(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        sent.append(messages)
        return ModelResponse(parts=[TextPart("ok")])

    with override_models(FunctionModel(reply)):
        result = complete_sync(SUMMARY, history)

    assert len(sent[0]) == 3
    assert result.output == "ok"
    # The request was the caller's; the run added only the response.
    assert [type(m) for m in result.new_messages] == [ModelResponse]


@ENTRIES
def test_the_trace_matches_a_run_over_build_agent(
    entry: Callable[..., object], exported: InMemorySpanExporter
) -> None:
    with override_models(replying("a summary")):
        agent = build_agent(SUMMARY, instructions="Summarize.")
        with opik.record_traces_locally() as storage:
            run_sync(agent, "summarize", attribution=ATTRIBUTION)
            entry(
                SUMMARY, "summarize", instructions="Summarize.", attribution=ATTRIBUTION
            )
            ran, completed = storage.trace_trees
    tracing.flush()

    def shape(trace: object) -> tuple[object, ...]:
        fields = ("name", "project_name", "thread_id", "input", "output")
        metadata = getattr(trace, "metadata", None) or {}
        return (
            *(getattr(trace, field) for field in fields),
            sorted(getattr(trace, "tags", None) or []),
            {k: v for k, v in metadata.items() if k != "created_from"},
        )

    assert shape(completed) == shape(ran)
    spans_by_trace: dict[object, list[str]] = {}
    for span in exported.get_finished_spans():
        trace_id = (span.attributes or {}).get("opik.trace_id")
        spans_by_trace.setdefault(trace_id, []).append(span.name)
    assert sorted(spans_by_trace[completed.id]) == sorted(spans_by_trace[ran.id])
    assert "invoke_agent summary" in spans_by_trace[completed.id]
    [chat] = [
        span
        for span in exported.get_finished_spans()
        if (span.attributes or {}).get("opik.trace_id") == completed.id
        and span.name.startswith("chat ")
    ]
    assert (chat.attributes or {}).get("gen_ai.usage.input_tokens") is not None


def test_an_answer_from_a_fallback_model_is_degraded_and_postprocessed() -> None:
    def partial(output: str) -> PostProcessed[str]:
        return PostProcessed(
            output=output.upper(), degraded_reasons=("partial_output",)
        )

    models = {PRIMARY: failing(503), FALLBACK: replying("backup")}
    with override_models(models), opik.record_traces_locally() as storage:
        result = complete_sync(SUMMARY_WITH_FALLBACK, "hi", postprocess=partial)
        [trace] = storage.trace_trees

    assert result.output == "BACKUP"
    assert result.model == FALLBACK
    assert result.degraded_reasons == ("fallback_model", "partial_output")
    assert trace.metadata is not None
    assert trace.metadata["degraded_reason"] == "fallback_model,partial_output"


@ENTRIES
def test_a_provider_failure_reaches_the_caller_as_the_library_error(
    entry: Callable[..., object],
) -> None:
    with override_models(failing(429)), pytest.raises(LLMRateLimited) as caught:
        entry(SUMMARY, "hi")

    assert isinstance(caught.value.__cause__, ModelHTTPError)


def test_invalid_structured_output_raises_llm_invalid_output() -> None:
    with (
        override_models(replying("not a tool call")),
        pytest.raises(LLMInvalidOutput) as caught,
    ):
        complete_sync(SUMMARY, "hi", output_type=Summary)

    assert caught.value.model == PRIMARY


@pytest.mark.usefixtures("provider_env")
def test_the_model_is_built_once_per_use_case_and_settings() -> None:
    first = shared_model(SUMMARY)

    assert shared_model(SUMMARY) is first
    configure(Settings(environment="qa"))
    assert shared_model(SUMMARY) is not first


def test_an_override_never_reaches_the_shared_model() -> None:
    with override_models(replying("one")):
        first = shared_model(SUMMARY)
    with override_models(replying("two")):
        second = shared_model(SUMMARY)

    assert first is not second


def test_complete_sync_works_from_several_threads(real_client: UseCase) -> None:
    outputs: list[str] = []
    errors: list[BaseException] = []

    def call() -> None:
        try:
            outputs.append(complete_sync(real_client, "hi").output)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    for _ in range(2):
        thread = threading.Thread(target=call)
        thread.start()
        thread.join()

    assert errors == []
    assert outputs == ["ok", "ok"]


def test_a_finished_threads_loop_is_released(real_client: UseCase) -> None:
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    def call() -> None:
        complete_sync(real_client, "hi")
        loops.append(weakref.ref(asyncio.get_event_loop()))

    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    gc.collect()

    assert loops[0]() is None


def test_complete_works_under_successive_event_loops(real_client: UseCase) -> None:
    outputs = [asyncio.run(complete(real_client, "hi")).output for _ in range(2)]

    assert outputs == ["ok", "ok"]
