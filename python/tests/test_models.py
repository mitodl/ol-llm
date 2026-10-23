import asyncio
import pickle
import threading
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine
from typing import Any, Self

import httpx
import httpx2
import pytest
from ol_llm import LLMRateLimited, LLMRejected, LLMUnavailable, UseCase
from ol_llm.attribution import Attribution
from ol_llm.errors import (
    LLMBudgetExceeded,
    LLMContentFiltered,
    LLMError,
    LLMInvalidOutput,
)
from ol_llm.models import (
    PolicyFallbackModel,
    PolicyModel,
    attribution_model_settings,
    build_agent,
    build_model,
    classify_failure,
    classify_run_failure,
    override_models,
    usage_limits,
)
from ol_llm.testing import policy_model, tracing_disabled
from pydantic import BaseModel
from pydantic_ai import Agent, DeferredToolRequests, ModelRetry, ToolOutput
from pydantic_ai.capabilities import RaiseContentFilterError, Toolset
from pydantic_ai.exceptions import (
    ContentFilterError,
    FallbackExceptionGroup,
    ModelAPIError,
    ModelHTTPError,
    PydanticAIDeprecationWarning,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import NativeOutput, OutputSpec
from pydantic_ai.toolsets import FunctionToolset

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
FEEDBACK = UseCase(
    app="ol-data-platform", name="feedback", profile="batch", model=PRIMARY
)


def http_error(status: int, headers: dict[str, str] | None = None) -> ModelHTTPError:
    return ModelHTTPError(
        status, "gpt-4o-mini", body="provider said no", headers=headers
    )


class Scripted:
    """A FunctionModel body that plays its steps in order, then answers.

    Each step is a failure to raise or a reply to give.
    """

    def __init__(self, *steps: Exception | str, answer: str = "ok") -> None:
        self.steps = list(steps)
        self.answer = answer
        self.calls: list[AgentInfo] = []

    def _next(self, info: AgentInfo) -> str:
        self.calls.append(info)
        step = self.steps.pop(0) if self.steps else self.answer
        if isinstance(step, Exception):
            raise step
        return step

    def __call__(self, _messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(self._next(info))])

    async def stream(
        self, _messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        yield self._next(info)


def always(exc: Callable[[], Exception], count: int = 20) -> list[Exception]:
    return [exc() for _ in range(count)]


# Classification


@pytest.mark.parametrize(
    ("exc", "expected", "status"),
    [
        (http_error(429), LLMRateLimited, None),
        (http_error(408), LLMUnavailable, None),
        (http_error(424), LLMRejected, 424),
        (
            ModelHTTPError(424, "gpt-4o-mini", body="stream error", in_stream=True),
            LLMUnavailable,
            None,
        ),
        (http_error(500), LLMUnavailable, None),
        (http_error(502), LLMUnavailable, None),
        (http_error(503), LLMUnavailable, None),
        (http_error(504), LLMUnavailable, None),
        (http_error(529), LLMUnavailable, None),
        (http_error(400), LLMRejected, 400),
        (http_error(401), LLMRejected, 401),
        (http_error(403), LLMRejected, 403),
        (http_error(404), LLMRejected, 404),
        (http_error(422), LLMRejected, 422),
        (http_error(501), LLMRejected, 501),
        (http_error(505), LLMRejected, 505),
        (ModelAPIError("gpt-4o-mini", "Request timed out."), LLMUnavailable, None),
        (httpx2.ReadTimeout("timed out"), LLMUnavailable, None),
        (httpx2.ConnectError("refused"), LLMUnavailable, None),
        (httpx2.RemoteProtocolError("peer closed"), LLMUnavailable, None),
        (httpx.ReadTimeout("timed out"), LLMUnavailable, None),
        (httpx2.UnsupportedProtocol("ftp://"), LLMRejected, None),
        (httpx2.LocalProtocolError("bad request"), LLMRejected, None),
        (httpx.ProxyError("proxy"), LLMRejected, None),
    ],
)
def test_failures_are_classified_by_type_and_status(
    exc: Exception, expected: type[LLMError], status: int | None
) -> None:
    error = classify_failure(exc, use_case="summary", model=PRIMARY)

    assert type(error) is expected
    assert error.use_case == "summary"
    assert error.model == PRIMARY
    if isinstance(error, LLMRejected):
        assert error.status_code == status


def test_rate_limit_carries_retry_after() -> None:
    error = classify_failure(
        http_error(429, {"Retry-After": "7"}), use_case="summary", model=PRIMARY
    )

    assert isinstance(error, LLMRateLimited)
    assert error.retry_after == 7


def test_other_exceptions_are_not_model_failures() -> None:
    assert classify_failure(ValueError("x"), use_case="s", model=PRIMARY) is None


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ContentFilterError("filtered"), LLMContentFiltered),
        (UnexpectedModelBehavior("bad output"), LLMInvalidOutput),
        (UsageLimitExceeded("too many requests"), LLMBudgetExceeded),
    ],
)
def test_run_failures_are_classified(exc: Exception, expected: type[LLMError]) -> None:
    assert type(classify_run_failure(exc, use_case="s", model=PRIMARY)) is expected


# The retry wrapper


def test_success_needs_one_attempt(sleeps: list[float]) -> None:
    script = Scripted()
    with override_models(FunctionModel(script)):
        result = build_agent(SUMMARY).run_sync("hi")

    assert result.output == "ok"
    assert len(script.calls) == 1
    assert sleeps == []


def test_every_request_gets_the_policy_timeout() -> None:
    script = Scripted()
    with override_models(FunctionModel(script)):
        build_agent(SUMMARY).run_sync("hi", model_settings={"timeout": 999})

    assert script.calls[0].model_settings is not None
    assert script.calls[0].model_settings["timeout"] == 60


def test_an_agents_model_settings_reach_every_request_under_the_policy() -> None:
    script = Scripted()
    with override_models(FunctionModel(script)):
        agent = build_agent(SUMMARY, model_settings={"max_tokens": 100})
        agent.run_sync("hi", model_settings={"timeout": 999})

    assert script.calls[0].model_settings is not None
    assert script.calls[0].model_settings["max_tokens"] == 100
    assert script.calls[0].model_settings["timeout"] == 60


def lookup(term: str) -> str:
    return term  # pragma: no cover


@pytest.mark.parametrize(
    "tools",
    [
        {"tools": [lookup]},
        {"toolsets": [FunctionToolset([lookup])]},
        {"capabilities": [Toolset(FunctionToolset([lookup]))]},
    ],
    ids=["tools", "toolsets", "capability"],
)
@pytest.mark.parametrize("choice", ["required", ["lookup"]])
def test_a_forced_function_tool_is_refused(
    tools: dict[str, Any], choice: str | list[str]
) -> None:
    script = Scripted()
    with override_models(FunctionModel(script)):
        agent = build_agent(SUMMARY, model_settings={"tool_choice": choice}, **tools)  # type: ignore[call-overload]
        with pytest.raises(ValueError, match="never give a final response"):
            agent.run_sync("hi")

    assert script.calls == []


def test_the_output_tool_may_be_forced_on_an_agent_with_tools() -> None:
    def record(_m: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [tool] = info.output_tools
        return ModelResponse(parts=[ToolCallPart(tool.name, {"text": "short"})])

    output = ToolOutput(Summary, name="record_summary")
    with override_models(FunctionModel(record)):
        agent = build_agent(
            SUMMARY,
            output_type=output,
            tools=[lookup],
            model_settings={"tool_choice": ["record_summary"]},
        )
        result = agent.run_sync("hi")

    assert result.output == Summary(text="short")


@pytest.mark.parametrize("choice", ["required", ["lookup", "record_summary"]])
def test_a_choice_that_still_reaches_the_output_tool_is_allowed(
    choice: str | list[str],
) -> None:
    def record(_m: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [tool] = info.output_tools
        return ModelResponse(parts=[ToolCallPart(tool.name, {"text": "short"})])

    output = ToolOutput(Summary, name="record_summary")
    with override_models(FunctionModel(record)):
        agent = build_agent(
            SUMMARY,
            output_type=output,
            tools=[lookup],
            model_settings={"tool_choice": choice},
        )
        result = agent.run_sync("hi")

    assert result.output == Summary(text="short")


@pytest.mark.parametrize("override", ["auto", "none"])
def test_a_runs_choice_lifts_an_agents_forced_function_tool(override: str) -> None:
    script = Scripted()
    with override_models(FunctionModel(script)):
        agent = build_agent(
            SUMMARY, tools=[lookup], model_settings={"tool_choice": ["lookup"]}
        )
        agent.run_sync("hi", model_settings={"tool_choice": override})  # type: ignore[typeddict-item]

    assert script.calls[0].model_settings is not None
    assert script.calls[0].model_settings["tool_choice"] == override


def test_environment_override_reaches_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OL_LLM_SUMMARY_TIMEOUT_SECONDS", "5")
    script = Scripted()
    with override_models(FunctionModel(script)):
        build_agent(SUMMARY).run_sync("hi")

    assert script.calls[0].model_settings is not None
    assert script.calls[0].model_settings["timeout"] == 5


def test_transient_failure_is_retried(sleeps: list[float]) -> None:
    script = Scripted(http_error(503))
    with override_models(FunctionModel(script)):
        result = build_agent(SUMMARY).run_sync("hi")

    assert result.output == "ok"
    assert len(script.calls) == 2
    assert len(sleeps) == 1
    assert 0.25 <= sleeps[0] <= 0.5


def test_retries_stop_at_the_policy_limit(sleeps: list[float]) -> None:
    script = Scripted(*always(lambda: http_error(503)))
    with (
        override_models(FunctionModel(script)),
        pytest.raises(LLMUnavailable) as caught,
    ):
        build_agent(SUMMARY).run_sync("hi")

    assert len(script.calls) == 3
    assert len(sleeps) == 2
    assert isinstance(caught.value.__cause__, ModelHTTPError)
    assert caught.value.use_case == "summary"
    assert caught.value.model == PRIMARY


def test_raw_transport_timeout_is_retried() -> None:
    script = Scripted(*always(lambda: httpx2.ReadTimeout("timed out")))
    with override_models(FunctionModel(script)), pytest.raises(LLMUnavailable):
        build_agent(SUMMARY).run_sync("hi")

    assert len(script.calls) == 3


def test_rejected_request_is_not_retried(sleeps: list[float]) -> None:
    script = Scripted(http_error(400))
    with override_models(FunctionModel(script)), pytest.raises(LLMRejected) as caught:
        build_agent(SUMMARY).run_sync("hi")

    assert len(script.calls) == 1
    assert sleeps == []
    assert caught.value.status_code == 400


def test_non_model_errors_pass_through_unchanged() -> None:
    script = Scripted(KeyError("bug in a tool"))
    with override_models(FunctionModel(script)), pytest.raises(KeyError):
        build_agent(SUMMARY).run_sync("hi")

    assert len(script.calls) == 1


def test_request_profile_backoff_is_a_total(sleeps: list[float]) -> None:
    use_case = UseCase(
        app="a",
        name="tight",
        profile="request",
        model=PRIMARY,
        max_retries=10,
        max_backoff_seconds=2,
        policy_reason="test",
    )
    script = Scripted(*always(lambda: http_error(503)))
    with override_models(FunctionModel(script)), pytest.raises(LLMUnavailable):
        build_agent(use_case).run_sync("hi")

    assert sum(sleeps) == pytest.approx(2)
    assert len(script.calls) == len(sleeps) + 1 < 11


def test_request_profile_gives_up_on_a_long_retry_after(sleeps: list[float]) -> None:
    script = Scripted(http_error(429, {"retry-after": "30"}))
    with (
        override_models(FunctionModel(script)),
        pytest.raises(LLMRateLimited) as caught,
    ):
        build_agent(SUMMARY).run_sync("hi")

    assert len(script.calls) == 1
    assert sleeps == []
    assert caught.value.retry_after == 30


def test_request_profile_honors_a_short_retry_after(sleeps: list[float]) -> None:
    script = Scripted(http_error(429, {"retry-after": "3"}))
    with override_models(FunctionModel(script)):
        build_agent(SUMMARY).run_sync("hi")

    assert sleeps == [3]


def test_batch_profile_waits_the_cap_on_a_long_retry_after(
    sleeps: list[float],
) -> None:
    script = Scripted(http_error(429, {"retry-after": "900"}))
    with override_models(FunctionModel(script)):
        result = build_agent(FEEDBACK).run_sync("hi")

    assert result.output == "ok"
    assert sleeps == [300]


def test_batch_profile_backs_off_exponentially(sleeps: list[float]) -> None:
    script = Scripted(*always(lambda: http_error(503)))
    with override_models(FunctionModel(script)), pytest.raises(LLMUnavailable):
        build_agent(FEEDBACK).run_sync("hi")

    assert len(script.calls) == 6
    assert len(sleeps) == 5
    for attempt, delay in enumerate(sleeps):
        assert 0.25 * 2**attempt <= delay <= 0.5 * 2**attempt


# Streams


def run_stream(use_case: UseCase) -> str:
    async def consume() -> str:
        async with build_agent(use_case).run_stream("hi") as stream:
            output: str = await stream.get_output()
            return output

    return asyncio.run(consume())


def test_stream_is_retried_while_opening() -> None:
    script = Scripted(http_error(503), http_error(502))
    with override_models(FunctionModel(script, stream_function=script.stream)):
        assert run_stream(SUMMARY) == "ok"

    assert len(script.calls) == 3


def test_stream_failure_after_opening_is_mapped_and_not_retried() -> None:
    calls: list[int] = []

    async def stream(_m: list[ModelMessage], _i: AgentInfo) -> AsyncIterator[str]:
        calls.append(1)
        yield "partial"
        msg = "stalled"
        raise httpx2.ReadTimeout(msg)

    with (
        override_models(FunctionModel(stream_function=stream)),
        pytest.raises(LLMUnavailable) as caught,
    ):
        run_stream(SUMMARY)

    assert calls == [1]
    assert isinstance(caught.value.__cause__, httpx2.ReadTimeout)


# Fallback


def fallback_models(primary: Scripted, fallback: Scripted) -> dict[str, FunctionModel]:
    return {
        PRIMARY: FunctionModel(primary, stream_function=primary.stream),
        FALLBACK: FunctionModel(
            fallback, stream_function=fallback.stream, model_name="fallback"
        ),
    }


def test_build_model_composes_the_fallback_chain() -> None:
    with override_models(TestModel()):
        model = build_model(SUMMARY_WITH_FALLBACK)

    assert isinstance(model, PolicyFallbackModel)
    assert [m.catalog_id for m in model.models] == [PRIMARY, FALLBACK]  # type: ignore[attr-defined]


def test_fallback_runs_after_the_primary_spends_its_retries() -> None:
    primary = Scripted(*always(lambda: http_error(503)))
    fallback = Scripted(answer="from fallback")
    with override_models(fallback_models(primary, fallback)):
        result = build_agent(SUMMARY_WITH_FALLBACK).run_sync("hi")

    assert result.output == "from fallback"
    assert result.response.model_name == "fallback"
    assert len(primary.calls) == 3
    assert len(fallback.calls) == 1


def test_all_models_failing_raises_the_primary_error() -> None:
    primary = Scripted(*always(lambda: http_error(503)))
    fallback = Scripted(*always(lambda: http_error(429)))
    with (
        override_models(fallback_models(primary, fallback)),
        pytest.raises(LLMUnavailable) as caught,
    ):
        build_agent(SUMMARY_WITH_FALLBACK).run_sync("hi")

    assert caught.value.model == PRIMARY
    group = caught.value.__cause__
    assert isinstance(group, FallbackExceptionGroup)
    assert [type(e) for e in group.exceptions] == [LLMUnavailable, LLMRateLimited]
    assert isinstance(group.exceptions[0].__cause__, ModelHTTPError)
    assert len(primary.calls) == 3
    assert len(fallback.calls) == 3


def test_rejected_primary_does_not_fall_back() -> None:
    primary = Scripted(http_error(401))
    fallback = Scripted()
    with (
        override_models(fallback_models(primary, fallback)),
        pytest.raises(LLMRejected),
    ):
        build_agent(SUMMARY_WITH_FALLBACK).run_sync("hi")

    assert len(primary.calls) == 1
    assert fallback.calls == []


def test_stream_falls_back_while_opening() -> None:
    primary = Scripted(*always(lambda: http_error(503)))
    fallback = Scripted(answer="from fallback")
    with override_models(fallback_models(primary, fallback)):
        assert run_stream(SUMMARY_WITH_FALLBACK) == "from fallback"


def test_stream_all_failing_raises_the_primary_error() -> None:
    primary = Scripted(*always(lambda: http_error(503)))
    fallback = Scripted(*always(lambda: http_error(503)))
    with (
        override_models(fallback_models(primary, fallback)),
        pytest.raises(LLMUnavailable) as caught,
    ):
        run_stream(SUMMARY_WITH_FALLBACK)

    assert caught.value.model == PRIMARY


# Agents


class Summary(BaseModel):
    text: str


def test_agent_carries_the_library_defaults() -> None:
    use_case = UseCase(
        app="a", name="extract", profile="batch", model=PRIMARY, output_retries=3
    )
    with override_models(TestModel()):
        agent = build_agent(use_case, output_type=Summary, instructions="Summarize.")

    assert isinstance(agent, Agent)
    assert agent.name == "extract"
    assert isinstance(agent.model, PolicyModel)
    assert any(
        isinstance(c, RaiseContentFilterError)
        for c in agent.root_capability.capabilities
    )
    assert agent.run_sync("hi").output == Summary(text="a")


def test_invalid_output_after_retries_is_classified() -> None:
    def never_valid(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("not json")])

    with override_models(FunctionModel(never_valid)):
        agent = build_agent(SUMMARY, output_type=NativeOutput(Summary))
        with pytest.raises(UnexpectedModelBehavior) as caught:
            agent.run_sync("hi")

    assert isinstance(
        classify_run_failure(caught.value, use_case="summary", model=PRIMARY),
        LLMInvalidOutput,
    )


SECOND_FALLBACK = "mistral:mistral-large-latest"
VALID = '{"text": "from fallback"}'


def run_for_summary(use_case: UseCase, *, stream: bool) -> Summary:
    agent = build_agent(use_case, output_type=NativeOutput(Summary))
    if not stream:
        return agent.run_sync("hi").output

    # run_stream does not retry invalid output; a run with an event handler streams
    # every request and does.
    async def drain(_ctx: object, events: AsyncIterable[object]) -> None:
        async for _ in events:
            pass

    return agent.run_sync("hi", event_stream_handler=drain).output


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_output_is_retried_on_the_model_that_gave_it(stream: bool) -> None:  # noqa: FBT001
    # The primary has recovered by the retry; it must still not answer it.
    primary = Scripted(
        *always(lambda: http_error(503), 3), answer='{"text": "from primary"}'
    )
    fallback = Scripted("not json", answer=VALID)
    with override_models(fallback_models(primary, fallback)):
        output = run_for_summary(SUMMARY_WITH_FALLBACK, stream=stream)

    assert output == Summary(text="from fallback")
    assert len(primary.calls) == 3
    assert len(fallback.calls) == 2


@pytest.mark.parametrize("stream", [False, True])
def test_a_retry_falls_back_only_past_the_model_that_gave_it(stream: bool) -> None:  # noqa: FBT001
    primary = Scripted(
        *always(lambda: http_error(503), 3), answer='{"text": "from primary"}'
    )
    fallback = Scripted("not json", *always(lambda: http_error(503), 3))
    second = Scripted(answer=VALID)
    use_case = UseCase(
        app="mit-learn",
        name="summary",
        profile="request",
        model=PRIMARY,
        fallback_models=(FALLBACK, SECOND_FALLBACK),
    )
    models = {
        **fallback_models(primary, fallback),
        SECOND_FALLBACK: FunctionModel(second, stream_function=second.stream),
    }
    with override_models(models):
        output = run_for_summary(use_case, stream=stream)

    assert output == Summary(text="from fallback")
    assert len(primary.calls) == 3
    assert len(fallback.calls) == 4
    assert len(second.calls) == 1


class ToolCalling:
    """A fallback that calls a tool on its first request, then on every later one.

    :param tool: the function tool to call, or ``None`` for the run's output tool.
    """

    def __init__(self, tool: str | None) -> None:
        self.tool = tool
        self.calls: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls.append(messages)
        if self.tool is None:
            [output_tool] = info.output_tools
            text = "" if len(self.calls) == 1 else "from fallback"
            return ModelResponse(parts=[ToolCallPart(output_tool.name, {"text": text})])
        if len(self.calls) == 1:
            return ModelResponse(parts=[ToolCallPart(self.tool, {"query": "x"})])
        return ModelResponse(parts=[TextPart("from fallback")])


def test_a_tools_model_retry_is_retried_on_the_model_that_gave_it() -> None:
    primary = Scripted(*always(lambda: http_error(503), 3), answer="from primary")
    fallback = ToolCalling("lookup")
    looked_up: list[str] = []
    models = {
        PRIMARY: FunctionModel(primary),
        FALLBACK: FunctionModel(fallback, model_name="fallback"),
    }
    with override_models(models):
        agent = build_agent(SUMMARY_WITH_FALLBACK)

        @agent.tool_plain
        def lookup(query: str) -> str:
            looked_up.append(query)
            if len(looked_up) == 1:
                msg = "try again"
                raise ModelRetry(msg)
            return "found"

        output = agent.run_sync("hi").output

    assert output == "from fallback"
    assert (len(primary.calls), len(fallback.calls)) == (3, 2)


def test_an_output_tools_retry_is_retried_on_the_model_that_gave_it() -> None:
    primary = Scripted(*always(lambda: http_error(503), 3), answer="from primary")
    fallback = ToolCalling(None)
    models = {
        PRIMARY: FunctionModel(primary),
        FALLBACK: FunctionModel(fallback, model_name="fallback"),
    }

    def not_empty(summary: Summary) -> Summary:
        if not summary.text:
            msg = "empty"
            raise ModelRetry(msg)
        return summary

    with override_models(models):
        agent = build_agent(SUMMARY_WITH_FALLBACK, output_type=ToolOutput(Summary))
        agent.output_validator(not_empty)
        output = agent.run_sync("hi").output

    assert output == Summary(text="from fallback")
    assert (len(primary.calls), len(fallback.calls)) == (3, 2)


def test_a_legacy_retry_part_built_in_memory_is_retried_on_the_model_that_gave_it() -> (
    None
):
    primary = Scripted(*always(lambda: http_error(503), 3), answer="from primary")
    fallback = Scripted("first", answer="from fallback")
    with override_models(fallback_models(primary, fallback)):
        agent = build_agent(SUMMARY_WITH_FALLBACK)
        history = agent.run_sync("hi").all_messages()
        with pytest.warns(PydanticAIDeprecationWarning, match="RetryPromptPart"):
            legacy = RetryPromptPart(content="Say it again.")
        history.append(ModelRequest(parts=[legacy]))
        output = agent.run_sync(message_history=history).output

    assert output == "from fallback"
    assert (len(primary.calls), len(fallback.calls)) == (3, 2)


@pytest.mark.parametrize(
    "output_type",
    [
        NativeOutput(Summary),
        [NativeOutput(Summary)],
        [NativeOutput(Summary), None],
        [[NativeOutput(Summary)], DeferredToolRequests],
    ],
)
def test_native_output_needs_model_support(
    monkeypatch: pytest.MonkeyPatch, output_type: OutputSpec[Any]
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    use_case = UseCase(
        app="a", name="extract", profile="batch", model="mistral:mistral-large-latest"
    )

    with pytest.raises(ValueError, match="does not support NativeOutput"):
        build_agent(use_case, output_type=output_type)


def test_usage_limits_follow_the_use_case() -> None:
    limits = usage_limits(SUMMARY)

    assert limits.request_limit == 50
    assert limits.cost_limit is None


def test_user_id_goes_to_each_provider_field() -> None:
    assert attribution_model_settings(Attribution()) == {}
    assert attribution_model_settings(Attribution(user_id="anon-1")) == {
        "openai_user": "anon-1",
        "anthropic_metadata": {"user_id": "anon-1"},
    }


def test_policy_model_wraps_an_agent_override() -> None:
    agent = Agent(TestModel())
    script = Scripted(http_error(503))
    with agent.override(model=policy_model(SUMMARY, FunctionModel(script))):
        assert agent.run_sync("hi").output == "ok"

    assert len(script.calls) == 2


def test_tracing_disabled_restores_the_previous_state() -> None:
    import opik  # noqa: PLC0415

    before = opik.is_tracing_active()
    with tracing_disabled():
        assert opik.is_tracing_active() is False
    assert opik.is_tracing_active() is before


def test_errors_survive_pickling() -> None:
    original = LLMRejected("m", use_case="u", model=PRIMARY, status_code=400)
    restored = pickle.loads(pickle.dumps(original))  # noqa: S301

    assert type(restored) is LLMRejected
    assert (restored.use_case, restored.status_code, restored.args) == (
        "u",
        400,
        ("m",),
    )


def test_error_copy_keeps_fields() -> None:
    original = LLMRateLimited("m", use_case="u", model=PRIMARY, retry_after=2.0)
    clone = original.copy()

    assert clone is not original
    assert (clone.use_case, clone.model, clone.retry_after, clone.args) == (
        "u",
        PRIMARY,
        2.0,
        ("m",),
    )


class RecordingModel(TestModel):
    """Records the event loop each enter and exit runs on, and what it exits with."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, asyncio.AbstractEventLoop]] = []
        self.exited_with: list[BaseException | None] = []

    async def __aenter__(self) -> Self:
        self.calls.append(("enter", asyncio.get_running_loop()))
        return self

    async def __aexit__(
        self, _type: object, exc: BaseException | None, _tb: object
    ) -> None:
        self.calls.append(("exit", asyncio.get_running_loop()))
        self.exited_with.append(exc)


def waited(event: threading.Event) -> None:
    """Wait for another thread's step, failing rather than hanging if it never comes."""
    if not event.wait(10):
        msg = "the other thread never reached its step"
        raise TimeoutError(msg)


def test_the_fallback_members_exit_with_the_runs_exception() -> None:
    recording = RecordingModel()
    with override_models(recording):
        model = build_model(SUMMARY_WITH_FALLBACK)
    error = RuntimeError("the run failed")

    async def run_failing() -> bool | None:
        await model.__aenter__()
        return await model.__aexit__(RuntimeError, error, None)

    suppressed = asyncio.run(run_failing())

    assert recording.exited_with == [error, error]
    assert not suppressed


def test_runs_on_two_threads_each_exit_the_fallback_members_they_entered() -> None:
    recording = RecordingModel()
    with override_models(recording):
        model = build_model(SUMMARY_WITH_FALLBACK)
    first_entered, second_entered, first_exited = (threading.Event() for _ in "123")

    async def first() -> None:
        await model.__aenter__()
        first_entered.set()
        waited(second_entered)
        await model.__aexit__(None, None, None)
        first_exited.set()

    async def second() -> None:
        waited(first_entered)
        await model.__aenter__()
        second_entered.set()
        waited(first_exited)
        await model.__aexit__(None, None, None)

    errors: list[BaseException] = []

    def target(run: Callable[[], Coroutine[Any, Any, None]]) -> None:
        try:
            asyncio.run(run())
        except BaseException as exc:
            errors.append(exc)
            raise

    threads = [
        threading.Thread(target=target, args=(run,), daemon=True)
        for run in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    loops = {loop for _, loop in recording.calls}
    assert len(loops) == 2
    for loop in loops:
        kinds = [kind for kind, on in recording.calls if on is loop]
        assert kinds == ["enter", "enter", "exit", "exit"]
