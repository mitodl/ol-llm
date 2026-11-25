import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import ol_llm.langchain as langchain_module
import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    BaseCallbackHandler,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableBinding
from ol_llm import PolicyChatModel, UseCase, chat_litellm_kwargs, with_call_policy
from ol_llm.retries import classify_http_failure
from pydantic import Field

TUTOR = UseCase(
    app="learn-ai", name="tutor", profile="request", model="openai:gpt-4o-mini"
)
ENVIRON: dict[str, str] = {}


class ProviderError(Exception):
    """Stands in for a litellm or OpenAI SDK error: a status and maybe headers."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        if headers is not None:
            self.litellm_response_headers = headers


class ScriptedChatModel(BaseChatModel):
    """Fails or answers per attempt; ``"ok"`` answers, ``"cut"`` fails mid-stream."""

    script: list[Any] = Field(default_factory=list)
    attempts: int = 0
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _step(self) -> Any:  # noqa: ANN401
        step = self.script[min(self.attempts, len(self.script) - 1)]
        self.attempts += 1
        if isinstance(step, BaseException):
            raise step
        return step

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> ChatResult:
        self._step()
        return ChatResult(generations=[ChatGeneration(message=AIMessage("answer"))])

    def _chunks(self) -> Iterator[ChatGenerationChunk]:
        step = self._step()
        yield ChatGenerationChunk(message=AIMessageChunk("part0 "))
        if step == "cut":
            raise ProviderError(503)
        yield ChatGenerationChunk(message=AIMessageChunk("part1"))

    def _stream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Iterator[ChatGenerationChunk]:
        yield from self._chunks()

    async def _astream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._chunks():
            yield chunk

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.bind(tools=list(tools), **kwargs)


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def record(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(langchain_module, "_sleep", record)
    monkeypatch.setattr(langchain_module, "_sync_sleep", recorded.append)
    return recorded


def _wrapped(*script: Any, streaming: bool = True) -> PolicyChatModel:  # noqa: ANN401
    inner = ScriptedChatModel(script=list(script), streaming=streaming)
    return with_call_policy(inner, TUTOR, environ=ENVIRON)


def _attempts(model: PolicyChatModel) -> int:
    assert isinstance(model.inner, ScriptedChatModel)
    return model.inner.attempts


def _astream(model: PolicyChatModel, received: list[str]) -> None:
    """Stream asynchronously, keeping the chunks that arrived before a failure."""

    async def collect() -> None:
        async for chunk in model.astream("hi"):
            if chunk.content:
                received.append(str(chunk.content))  # noqa: PERF401

    asyncio.run(collect())


def test_chat_litellm_kwargs_apply_the_profile_default_timeout() -> None:
    assert chat_litellm_kwargs(TUTOR, environ={})["request_timeout"] == 60


def test_chat_litellm_kwargs_set_the_timeout_and_turn_client_retries_off() -> None:
    environ = {"OL_LLM_TUTOR_TIMEOUT_SECONDS": "120", "OL_LLM_REQUEST_MAX_RETRIES": "5"}

    assert chat_litellm_kwargs(TUTOR, environ=environ) == {
        "request_timeout": 120.0,
        "max_retries": 1,
        "model_kwargs": {"max_retries": 0},
    }


def test_a_failure_before_the_first_chunk_is_retried(waits: list[float]) -> None:
    model = _wrapped(ProviderError(503), "ok")
    received: list[str] = []

    _astream(model, received)

    assert received == ["part0 ", "part1"]
    assert _attempts(model) == 2
    assert len(waits) == 1


def test_a_failure_after_the_first_chunk_is_not_retried(waits: list[float]) -> None:
    model = _wrapped("cut", "ok")
    received: list[str] = []

    with pytest.raises(ProviderError):
        _astream(model, received)

    assert received == ["part0 "]
    assert _attempts(model) == 1
    assert waits == []


def test_retries_stop_at_the_policy(waits: list[float]) -> None:
    model = _wrapped(ProviderError(503))

    with pytest.raises(ProviderError):
        _astream(model, [])

    assert _attempts(model) == 3
    assert len(waits) == 2
    assert sum(waits) <= 10


def test_a_retry_after_past_the_budget_ends_the_call(waits: list[float]) -> None:
    model = _wrapped(ProviderError(429, {"Retry-After": "30"}), "ok")

    with pytest.raises(ProviderError):
        _astream(model, [])

    assert _attempts(model) == 1
    assert waits == []


def test_a_retry_after_within_the_budget_is_waited_out(waits: list[float]) -> None:
    model = _wrapped(ProviderError(429, {"retry-after-ms": "2500"}), "ok")
    received: list[str] = []

    _astream(model, received)

    assert received == ["part0 ", "part1"]
    assert waits == [2.5]


@pytest.mark.usefixtures("waits")
def test_a_rejection_is_not_retried() -> None:
    model = _wrapped(ProviderError(400), "ok")

    with pytest.raises(ProviderError):
        _astream(model, [])

    assert _attempts(model) == 1


@pytest.mark.usefixtures("waits")
def test_a_transport_timeout_in_the_cause_chain_is_retried() -> None:
    wrapped = RuntimeError("connection failed")
    wrapped.__cause__ = httpx.ReadTimeout("read timed out")
    model = _wrapped(wrapped, "ok")
    received: list[str] = []

    _astream(model, received)

    assert received == ["part0 ", "part1"]
    assert _attempts(model) == 2


@pytest.mark.usefixtures("waits")
def test_an_unclassified_error_is_not_retried() -> None:
    model = _wrapped(KeyError("bug"), "ok")

    with pytest.raises(KeyError):
        _astream(model, [])

    assert _attempts(model) == 1


@pytest.mark.usefixtures("waits")
def test_ainvoke_on_a_streaming_model_follows_the_stream_rule() -> None:
    model = _wrapped("cut", "ok")

    with pytest.raises(ProviderError):
        asyncio.run(model.ainvoke("hi"))

    assert _attempts(model) == 1


@pytest.mark.usefixtures("waits")
def test_ainvoke_on_a_non_streaming_model_retries() -> None:
    model = _wrapped(ProviderError(502), "ok", streaming=False)

    reply = asyncio.run(model.ainvoke("hi"))

    assert reply.content == "answer"
    assert _attempts(model) == 2


@pytest.mark.usefixtures("waits")
def test_invoke_and_stream_retry_in_sync_code() -> None:
    streamed = _wrapped(ProviderError(503), "ok")
    invoked = _wrapped(ProviderError(503), "ok", streaming=False)

    chunks = [str(chunk.content) for chunk in streamed.stream("hi") if chunk.content]
    reply = invoked.invoke("hi")

    assert chunks == ["part0 ", "part1"]
    assert reply.content == "answer"
    assert (_attempts(streamed), _attempts(invoked)) == (2, 2)


def test_bind_tools_binds_the_wrapper_with_the_models_tool_settings() -> None:
    model = _wrapped("ok")

    bound = model.bind_tools(["search"], tool_choice="auto")

    assert isinstance(bound, RunnableBinding)
    assert bound.bound is model
    assert bound.kwargs == {"tools": ["search"], "tool_choice": "auto"}


def test_bind_tools_refuses_a_binding_it_cannot_move() -> None:
    class OtherBinding(ScriptedChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
            return ScriptedChatModel().bind(tools=list(tools))

    model = with_call_policy(OtherBinding(), TUTOR, environ=ENVIRON)

    with pytest.raises(TypeError, match="did not bind the model itself"):
        model.bind_tools(["search"])


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, "LLMRateLimited"),
        (503, "LLMUnavailable"),
        (408, "LLMUnavailable"),
        (400, "LLMRejected"),
        # litellm's status for an error it could not map, with no response behind it.
        (500, "LLMRejected"),
    ],
)
def test_classify_http_failure_by_status(status: int, expected: str) -> None:
    error = classify_http_failure(
        ProviderError(status), use_case="tutor", model="openai:gpt-4o-mini"
    )

    assert type(error).__name__ == expected


def _http_error(status: int, headers: dict[str, str] | None = None) -> ProviderError:
    """A provider error whose chain holds the HTTP response, as litellm's does."""
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    response = httpx.Response(status, headers=headers, request=request)
    error = ProviderError(status)
    error.__context__ = httpx.HTTPStatusError(
        "failed", request=request, response=response
    )
    return error


def test_a_500_from_the_provider_is_retried(waits: list[float]) -> None:
    model = _wrapped(_http_error(500), "ok")
    received: list[str] = []

    _astream(model, received)

    assert received == ["part0 ", "part1"]
    assert _attempts(model) == 2
    assert len(waits) == 1


@pytest.mark.usefixtures("waits")
def test_a_500_with_no_response_behind_it_is_not_retried() -> None:
    model = _wrapped(ProviderError(500), "ok")

    with pytest.raises(ProviderError):
        _astream(model, [])

    assert _attempts(model) == 1


def test_the_responses_status_wins_over_the_exceptions() -> None:
    error = ProviderError(500)
    error.__context__ = _http_error(400).__context__

    classified = classify_http_failure(
        error, use_case="tutor", model="openai:gpt-4o-mini"
    )

    assert type(classified).__name__ == "LLMRejected"


def test_retry_after_is_read_from_the_response_in_the_chain(waits: list[float]) -> None:
    model = _wrapped(_http_error(429, {"Retry-After": "3"}), "ok")

    _astream(model, [])

    assert waits == [3.0]


def test_the_models_run_settings_move_to_the_wrapper() -> None:
    handler = BaseCallbackHandler()
    inner = ScriptedChatModel(
        script=["ok"], callbacks=[handler], tags=["tutor"], metadata={"course": "x"}
    )

    model = with_call_policy(inner, TUTOR, environ=ENVIRON)

    assert model.callbacks == [handler]
    assert model.tags == ["tutor"]
    assert (model.metadata or {})["course"] == "x"


def test_with_structured_output_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="with_structured_output"):
        _wrapped("ok").with_structured_output({"type": "object"})
