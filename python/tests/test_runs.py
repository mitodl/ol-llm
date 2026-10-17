import asyncio
from collections.abc import Callable
from dataclasses import replace

import opik
import pytest
from ol_llm import (
    Attribution,
    LLMInvalidOutput,
    LLMRateLimited,
    PostProcessed,
    UseCase,
    run,
    run_sync,
    tracing,
)
from ol_llm.models import build_agent, override_models
from ol_llm.testing import tracing_disabled
from ol_llm.tracing import configure_tracing
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

PRIMARY = "openai:gpt-4o-mini"
FALLBACK = "anthropic:claude-haiku-4-5"

SUMMARY = UseCase(
    app="mit-learn",
    name="summary",
    profile="request",
    model=PRIMARY,
)
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


def replying(text: str) -> FunctionModel:
    def reply(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(reply)


def failing(status: int) -> FunctionModel:
    def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status, "gpt-4o-mini", body="provider said no")

    return FunctionModel(fail)


def recording(seen: list[AgentInfo]) -> FunctionModel:
    def reply(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(reply)


def agent_for(use_case: UseCase, model: object, **kwargs: object) -> Agent[None, str]:
    with override_models(model):  # type: ignore[arg-type]
        return build_agent(use_case, **kwargs)  # type: ignore[call-overload,no-any-return]


def test_run_returns_the_output_and_the_answering_model() -> None:
    result = asyncio.run(run(agent_for(SUMMARY, replying("a summary")), "summarize"))

    assert result.output == "a summary"
    assert result.model == PRIMARY
    assert result.usage.requests == 1
    assert not result.degraded
    assert result.degraded_reasons == ()
    assert len(result.new_messages) == 2


def test_run_sync_matches_run() -> None:
    result = run_sync(agent_for(SUMMARY, replying("a summary")), "summarize")

    assert (result.output, result.model, result.degraded) == (
        "a summary",
        PRIMARY,
        False,
    )


@pytest.mark.parametrize("sync", [False, True])
def test_the_root_trace_carries_the_attribution_fields(sync: bool) -> None:  # noqa: FBT001
    agent = agent_for(SUMMARY, replying("a summary"))

    with opik.record_traces_locally() as storage:
        if sync:
            run_sync(agent, "summarize", attribution=ATTRIBUTION)
        else:
            asyncio.run(run(agent, "summarize", attribution=ATTRIBUTION))
        [trace] = storage.trace_trees

    assert trace.name == "summary"
    assert trace.project_name == "mit-learn"
    assert trace.thread_id == "thread-1"
    assert sorted(trace.tags or []) == [
        "app:mit-learn",
        "env:ci",
        "use_case:summary",
    ]
    assert trace.metadata is not None
    assert {k: trace.metadata[k] for k in ("app", "use_case", "user_id", "run_id")} == {
        "app": "mit-learn",
        "use_case": "summary",
        "user_id": "user-1",
        "run_id": "run-1",
    }
    assert trace.input == {"user_prompt": "summarize"}
    assert trace.output == {"output": "a summary"}


def test_model_spans_land_in_the_root_trace(exported: InMemorySpanExporter) -> None:
    agent = agent_for(SUMMARY, replying("a summary"))

    with opik.record_traces_locally() as storage:
        run_sync(agent, "summarize")
        [trace] = storage.trace_trees
    tracing.flush()

    spans = exported.get_finished_spans()
    assert {span.name for span in spans} >= {"invoke_agent summary"}
    assert {(s.attributes or {}).get("opik.trace_id") for s in spans} == {trace.id}


def test_the_user_id_reaches_the_provider() -> None:
    seen: list[AgentInfo] = []

    run_sync(agent_for(SUMMARY, recording(seen)), "hi", attribution=ATTRIBUTION)

    assert seen[0].model_settings is not None
    assert seen[0].model_settings.get("openai_user") == "user-1"


def test_an_answer_from_a_fallback_model_is_degraded() -> None:
    agent = agent_for(
        SUMMARY_WITH_FALLBACK, {PRIMARY: failing(503), FALLBACK: replying("backup")}
    )

    with opik.record_traces_locally() as storage:
        result = run_sync(agent, "summarize")
        [trace] = storage.trace_trees

    assert result.output == "backup"
    assert result.model == FALLBACK
    assert result.degraded_reasons == ("fallback_model",)
    assert trace.metadata is not None
    assert trace.metadata["degraded"] is True
    assert trace.metadata["degraded_reason"] == "fallback_model"


def test_postprocess_replaces_the_output_and_adds_reasons() -> None:
    def partial(output: str) -> PostProcessed[str]:
        return PostProcessed(
            output=output.upper(), degraded_reasons=("partial_output",)
        )

    agent = agent_for(
        SUMMARY_WITH_FALLBACK, {PRIMARY: failing(503), FALLBACK: replying("backup")}
    )

    with opik.record_traces_locally() as storage:
        result = run_sync(agent, "summarize", postprocess=partial)
        [trace] = storage.trace_trees

    assert result.output == "BACKUP"
    assert result.degraded_reasons == ("fallback_model", "partial_output")
    assert trace.output == {"output": "BACKUP"}
    assert trace.metadata is not None
    assert trace.metadata["degraded_reason"] == "fallback_model,partial_output"


def test_an_exception_from_postprocess_reaches_the_caller_unchanged() -> None:
    def broken(_output: str) -> PostProcessed[str]:
        msg = "segment"
        raise KeyError(msg)

    with pytest.raises(KeyError, match="segment"):
        run_sync(agent_for(SUMMARY, replying("ok")), "hi", postprocess=broken)


def test_a_provider_failure_reaches_the_caller_as_the_library_error() -> None:
    agent = agent_for(SUMMARY, failing(429))

    with opik.record_traces_locally() as storage:
        with pytest.raises(LLMRateLimited) as caught:
            run_sync(agent, "hi")
        [trace] = storage.trace_trees

    assert isinstance(caught.value.__cause__, ModelHTTPError)
    assert trace.error_info is not None
    assert trace.error_info["exception_type"] == "LLMRateLimited"


def test_invalid_output_reaches_the_caller_as_llm_invalid_output() -> None:
    agent = agent_for(SUMMARY, replying("not a tool call"), output_type=Summary)

    with pytest.raises(LLMInvalidOutput) as caught:
        asyncio.run(run(agent, "hi"))

    assert isinstance(caught.value.__cause__, UnexpectedModelBehavior)
    assert caught.value.use_case == "summary"


def test_without_content_the_trace_omits_input_and_output() -> None:
    config = replace(tracing.get_tracing_config(), include_content=False)
    tracing.reset()
    configure_tracing(config)
    agent = agent_for(SUMMARY, replying("a summary"))

    with opik.record_traces_locally() as storage:
        run_sync(agent, "summarize")
        [trace] = storage.trace_trees

    assert not trace.input
    assert not trace.output
    [capability] = [
        c for c in agent.root_capability.capabilities if isinstance(c, Instrumentation)
    ]
    assert not capability.settings.include_content


def test_with_tracing_off_agents_are_not_instrumented() -> None:
    tracing.reset()
    configure_tracing(replace(tracing.get_tracing_config(), enabled=False))
    agent = agent_for(SUMMARY, replying("ok"))

    assert not [
        c for c in agent.root_capability.capabilities if isinstance(c, Instrumentation)
    ]
    assert run_sync(agent, "hi").output == "ok"


def test_a_run_with_tracing_disabled_records_nothing(
    exported: InMemorySpanExporter,
) -> None:
    agent = agent_for(SUMMARY, replying("ok"))

    with opik.record_traces_locally() as storage, tracing_disabled():
        run_sync(agent, "hi")
        traces = storage.trace_trees
    tracing.flush()

    assert traces == []
    assert exported.get_finished_spans() == ()


def test_a_run_inside_a_tracked_function_is_a_span_of_its_trace() -> None:
    agent = agent_for(SUMMARY, replying("ok"))

    @opik.track(project_name="mit-learn")
    def handler() -> str:
        return run_sync(agent, "hi", attribution=ATTRIBUTION).output

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees
        [root] = storage.span_trees

    assert trace.name == "handler"
    assert trace.thread_id == "thread-1"
    assert [span.name for span in root.spans] == ["summary"]
    assert "use_case:summary" in (root.spans[0].tags or [])


@pytest.mark.parametrize("entry", [run_sync, lambda *a: asyncio.run(run(*a))])
def test_an_agent_not_from_build_agent_is_refused(
    entry: Callable[..., object],
) -> None:
    with pytest.raises(TypeError, match="build_model"):
        entry(Agent(TestModel()), "hi")
