import asyncio
import gc
import threading
import weakref
from collections.abc import Callable
from dataclasses import replace

import opik
import pytest
from conftest import GEMINI, PromptLibrary, assert_error_kept_without_content
from ol_llm import (
    Attribution,
    LLMInvalidOutput,
    LLMRateLimited,
    LLMRejected,
    ManagedPrompt,
    PostProcessed,
    TextTemplate,
    UseCase,
    get_prompt,
    run,
    run_sync,
    tracing,
)
from ol_llm.models import build_agent, override_models
from ol_llm.testing import tracing_disabled
from ol_llm.tracing import PROJECT_ATTRIBUTE, TracingConfig, configure_tracing
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opik import context_storage
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

PRIMARY = "openai:gpt-4o-mini"
FALLBACK = "bedrock:global.anthropic.claude-haiku-4-5-20251001-v1:0"

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
# A provider error body that quotes the request.
ECHO = "echoed: the learner's essay"


class Summary(BaseModel):
    text: str


def replying(text: str) -> FunctionModel:
    def reply(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(reply)


def failing(status: int, body: str = "provider said no") -> FunctionModel:
    def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status, "gpt-4o-mini", body=body)

    return FunctionModel(fail)


def stalling(entered: asyncio.Event) -> FunctionModel:
    async def stall(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError  # pragma: no cover

    return FunctionModel(stall)


def opik_context_is_empty() -> bool:
    return (
        context_storage.get_trace_data() is None
        and context_storage.span_data_stack_empty()
    )


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


@pytest.mark.parametrize("entry", [run_sync, lambda *a: asyncio.run(run(*a))])
def test_without_content_a_failure_keeps_only_its_type_and_frames(
    entry: Callable[..., object], exported: InMemorySpanExporter
) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(include_content=False))
    agent = agent_for(SUMMARY, failing(400, ECHO))

    with opik.record_traces_locally() as storage:
        with pytest.raises(LLMRejected, match=ECHO):
            entry(agent, "hi")
        [trace] = storage.trace_trees
    tracing.flush()

    assert trace.error_info is not None
    assert trace.error_info["exception_type"] == "LLMRejected"
    assert_error_kept_without_content(trace, exported.get_finished_spans(), ECHO)


def test_with_content_a_failure_is_described() -> None:
    agent = agent_for(SUMMARY, failing(400, ECHO))

    with opik.record_traces_locally() as storage:
        with pytest.raises(LLMRejected):
            run_sync(agent, "hi")
        [trace] = storage.trace_trees

    assert trace.error_info is not None
    assert ECHO in trace.error_info["message"]
    assert ECHO in trace.error_info["traceback"]


def test_a_cancelled_run_closes_its_trace() -> None:
    left_empty: list[bool] = []

    async def cancel() -> None:
        entered = asyncio.Event()
        agent = agent_for(SUMMARY, stalling(entered))

        async def running() -> None:
            try:
                await run(agent, "hi")
            except asyncio.CancelledError:
                left_empty.append(opik_context_is_empty())
                raise

        task = asyncio.create_task(running())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with opik.record_traces_locally() as storage:
        asyncio.run(cancel())
        [trace] = storage.trace_trees

    assert left_empty == [True]
    assert trace.end_time is not None
    assert trace.error_info is None


def test_a_run_that_times_out_in_the_callers_task_closes_its_trace() -> None:
    async def time_out() -> bool:
        agent = agent_for(SUMMARY, stalling(asyncio.Event()))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(run(agent, "hi"), timeout=0.01)
        return opik_context_is_empty()

    with opik.record_traces_locally() as storage:
        left_empty = asyncio.run(time_out())
        [trace] = storage.trace_trees

    assert left_empty
    assert trace.end_time is not None
    assert trace.error_info is None


def test_invalid_output_reaches_the_caller_as_llm_invalid_output() -> None:
    agent = agent_for(SUMMARY, replying("not a tool call"), output_type=Summary)

    with pytest.raises(LLMInvalidOutput) as caught:
        asyncio.run(run(agent, "hi"))

    assert isinstance(caught.value.__cause__, UnexpectedModelBehavior)
    assert caught.value.use_case == "summary"
    assert caught.value.model == PRIMARY


@pytest.mark.parametrize("entry", [run_sync, lambda *a: asyncio.run(run(*a))])
def test_invalid_output_names_the_model_that_gave_it(
    entry: Callable[..., object],
) -> None:
    agent = agent_for(
        SUMMARY_WITH_FALLBACK,
        {PRIMARY: failing(503), FALLBACK: replying("not a tool call")},
        output_type=Summary,
    )

    with pytest.raises(LLMInvalidOutput) as caught:
        entry(agent, "hi")

    assert caught.value.model == FALLBACK


def test_runs_in_one_trace_keep_each_degraded_reason() -> None:
    def partial(output: str) -> PostProcessed[str]:
        return PostProcessed(output=output, degraded_reasons=("partial_output",))

    fallback = agent_for(
        SUMMARY_WITH_FALLBACK, {PRIMARY: failing(503), FALLBACK: replying("backup")}
    )
    primary = agent_for(SUMMARY, replying("ok"))

    @opik.track(project_name="mit-learn")
    def handler() -> None:
        run_sync(fallback, "hi")
        run_sync(primary, "hi", postprocess=partial)

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees

    assert trace.metadata is not None
    assert trace.metadata["degraded_reason"] == "fallback_model,partial_output"


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
    configure_tracing(TracingConfig(enabled=False))
    agent = agent_for(SUMMARY, replying("ok"))

    assert not [
        c for c in agent.root_capability.capabilities if isinstance(c, Instrumentation)
    ]
    assert run_sync(agent, "hi").output == "ok"


def test_with_tracing_off_global_instrumentation_records_nothing() -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False))
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    try:
        run_sync(agent_for(SUMMARY, replying("ok")), "hi")
    finally:
        Agent.instrument_all(False)  # noqa: FBT003

    assert exporter.get_finished_spans() == ()


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


def test_a_run_inside_a_tracked_function_is_a_span_of_its_trace(
    exported: InMemorySpanExporter,
) -> None:
    agent = agent_for(SUMMARY, replying("ok"))

    @opik.track(project_name="ol-data-platform")
    def handler() -> str:
        return run_sync(agent, "hi", attribution=ATTRIBUTION).output

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees
        [root] = storage.span_trees
    tracing.flush()

    assert trace.name == "handler"
    assert trace.project_name == "ol-data-platform"
    assert trace.thread_id == "thread-1"
    assert "use_case:summary" in (trace.tags or [])
    assert [span.name for span in root.spans] == ["summary"]
    spans = exported.get_finished_spans()
    assert {(s.attributes or {}).get("opik.trace_id") for s in spans} == {trace.id}
    assert {(s.attributes or {}).get(PROJECT_ATTRIBUTE) for s in spans} == {
        "ol-data-platform"
    }


def test_a_nested_run_keeps_the_callers_thread() -> None:
    agent = agent_for(SUMMARY, replying("ok"))

    @opik.track(project_name="mit-learn")
    def handler() -> None:
        opik.opik_context.update_current_trace(thread_id="caller-thread")
        run_sync(agent, "hi", attribution=ATTRIBUTION)

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees

    assert trace.thread_id == "caller-thread"


def test_with_tracing_off_a_run_opens_no_trace() -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False))
    agent = agent_for(SUMMARY, replying("ok"))

    with opik.record_traces_locally() as storage:
        result = run_sync(agent, "hi", attribution=ATTRIBUTION)
        traces = storage.trace_trees

    assert result.output == "ok"
    assert traces == []


@pytest.mark.parametrize("entry", [run_sync, lambda *a: asyncio.run(run(*a))])
def test_an_agent_not_from_build_agent_is_refused(
    entry: Callable[..., object],
) -> None:
    with pytest.raises(TypeError, match="build_model"):
        entry(Agent(TestModel()), "hi")


# One agent built once and run on several event loops, over real provider clients
# against a local keep-alive server (the real_client fixture).

THREAD_TIMEOUT_SECONDS = 30


def run_in_threads(target: Callable[[], None], count: int) -> list[BaseException]:
    """Run ``target`` on ``count`` threads at once and return what they raised."""
    errors: list[BaseException] = []
    barrier = threading.Barrier(count)

    def call() -> None:
        barrier.wait()
        try:
            target()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(THREAD_TIMEOUT_SECONDS)
        assert not thread.is_alive(), "a run hung"
    return errors


def test_run_sync_works_from_several_threads(real_client: UseCase) -> None:
    agent = build_agent(real_client)
    outputs: list[str] = []

    def call() -> None:
        outputs.append(run_sync(agent, "hi").output)

    for _ in range(2):
        assert run_in_threads(call, 1) == []

    assert outputs == ["ok", "ok"]


def test_run_sync_works_from_concurrent_threads(real_client: UseCase) -> None:
    agent = build_agent(real_client)
    outputs: list[str] = []

    def call() -> None:
        outputs.extend(run_sync(agent, "hi").output for _ in range(3))

    assert run_in_threads(call, 4) == []
    assert outputs == ["ok"] * 12


def test_run_works_under_successive_event_loops(real_client: UseCase) -> None:
    agent = build_agent(real_client)

    outputs = [asyncio.run(run(agent, "hi")).output for _ in range(2)]

    assert outputs == ["ok", "ok"]


def test_a_finished_threads_loop_is_released_by_the_agent(
    real_client: UseCase,
) -> None:
    agent = build_agent(real_client)
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    def call() -> None:
        run_sync(agent, "hi")
        loops.append(weakref.ref(asyncio.get_event_loop()))

    assert run_in_threads(call, 1) == []
    gc.collect()

    assert loops[0]() is None


def test_a_dropped_loops_clients_log_no_errors_on_a_later_loop(
    real_client: UseCase, caplog: pytest.LogCaptureFixture
) -> None:
    agent = build_agent(real_client)

    async def call() -> None:
        await run(agent, "hi")
        # Collect the earlier loops' clients while this loop runs, which is when
        # an SDK client's __del__ schedules its close on the wrong loop.
        gc.collect()

    for _ in range(5):
        asyncio.run(call())

    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


@pytest.mark.parametrize("real_client", [(GEMINI,)], indirect=True, ids=["google"])
def test_run_stream_works_under_successive_event_loops(real_client: UseCase) -> None:
    agent = build_agent(real_client)

    async def stream() -> str:
        async with agent.run_stream("hi") as result:
            output: str = await result.get_output()
            return output

    outputs = [asyncio.run(stream()) for _ in range(2)]

    assert outputs == ["ok", "ok"]


@pytest.mark.parametrize("sync", [False, True])
def test_a_managed_prompt_is_linked_to_the_run(
    prompt_library: PromptLibrary,
    sync: bool,  # noqa: FBT001
) -> None:
    prompt_library.add("summary", "Summarize.", environments=("production",))
    prompt = get_prompt(
        "summary",
        project_name="mit-learn",
        default=TextTemplate(template="Summarize."),
        environment="production",
    )
    agent = agent_for(SUMMARY, replying("ok"), instructions=prompt.format())

    with opik.record_traces_locally() as storage:
        if sync:
            result = run_sync(agent, "hi", prompt=prompt)
        else:
            result = asyncio.run(run(agent, "hi", prompt=prompt))
        [trace] = storage.trace_trees

    assert (result.prompt_name, result.prompt_version) == ("summary", "v1")
    assert not result.degraded
    assert trace.metadata is not None
    assert trace.metadata["prompt_name"] == "summary"
    assert trace.metadata["prompt_version"] == "v1"
    [linked] = trace.metadata["opik_prompts"]
    assert linked["version"]["version_number"] == "v1"


def test_an_embedded_prompt_marks_the_result_and_the_trace() -> None:
    prompt = ManagedPrompt(
        name="summary",
        template=TextTemplate(template="Summarize.", version="v3"),
        source="embedded",
        degraded_reasons=("embedded_prompt",),
    )
    agent = agent_for(
        SUMMARY_WITH_FALLBACK, {PRIMARY: failing(503), FALLBACK: replying("backup")}
    )

    with opik.record_traces_locally() as storage:
        result = run_sync(agent, "summarize", prompt=prompt)
        [trace] = storage.trace_trees

    assert (result.prompt_name, result.prompt_version) == ("summary", "v3")
    assert result.degraded_reasons == ("fallback_model", "embedded_prompt")
    assert trace.metadata is not None
    assert trace.metadata["degraded_reason"] == "fallback_model,embedded_prompt"
    assert trace.metadata["prompt_version"] == "v3"
    assert "opik_prompts" not in trace.metadata


@pytest.mark.parametrize("source", ["opik", "embedded"])
def test_a_nested_run_names_its_prompt_on_the_callers_trace(
    prompt_library: PromptLibrary, source: str
) -> None:
    prompt_library.add("summary", "Summarize.", environments=("production",))
    prompt = (
        get_prompt(
            "summary",
            project_name="mit-learn",
            default=TextTemplate(template="Summarize."),
            environment="production",
        )
        if source == "opik"
        else ManagedPrompt(
            name="summary",
            template=TextTemplate(template="Summarize.", version="v1"),
            source="embedded",
        )
    )
    agent = agent_for(SUMMARY, replying("ok"), instructions=prompt.format())

    @opik.track(project_name="mit-learn")
    def handler() -> None:
        run_sync(agent, "hi", prompt=prompt)

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees

    assert trace.name == "handler"
    assert trace.metadata is not None
    assert trace.metadata["prompt_name"] == "summary"
    assert trace.metadata["prompt_version"] == "v1"


@pytest.mark.parametrize("sync", [False, True])
def test_every_prompt_is_linked_and_an_embedded_one_marks_the_run(
    prompt_library: PromptLibrary,
    sync: bool,  # noqa: FBT001
) -> None:
    for name in ("summary.system", "summary.user"):
        prompt_library.add(name, "Summarize.", environments=("production",))
    system, user = (
        get_prompt(
            name,
            project_name="mit-learn",
            default=TextTemplate(template="Summarize."),
            environment="production",
        )
        for name in ("summary.system", "summary.user")
    )
    embedded = ManagedPrompt(
        name="summary.example",
        template=TextTemplate(template="For example:", version="v2"),
        source="embedded",
        degraded_reasons=("embedded_prompt",),
    )
    agent = agent_for(SUMMARY, replying("ok"), instructions=system.format())
    prompts = [system, user, embedded]

    with opik.record_traces_locally() as storage:
        if sync:
            result = run_sync(agent, "hi", prompt=prompts)
        else:
            result = asyncio.run(run(agent, "hi", prompt=prompts))
        [trace] = storage.trace_trees

    assert (result.prompt_name, result.prompt_version) == ("summary.system", "v1")
    assert result.degraded_reasons == ("embedded_prompt",)
    assert trace.metadata is not None
    assert trace.metadata["prompt_name"] == "summary.system"
    assert trace.metadata["degraded_reason"] == "embedded_prompt"
    linked = [p["name"] for p in trace.metadata["opik_prompts"]]
    assert linked == ["summary.system", "summary.user"]


def test_without_a_prompt_the_result_names_none() -> None:
    result = run_sync(agent_for(SUMMARY, replying("ok")), "hi")

    assert (result.prompt_name, result.prompt_version) == (None, None)
