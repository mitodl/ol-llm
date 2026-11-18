"""Call spans for chat calls that a LangChain or LangGraph call site makes itself.

``traced_chat`` gives a LangChain chat model's calls the call span the entry
points open (usage policies spec §4.1, library spec §6.6): the §4 attribution
fields, the catalog checks ``build_model`` runs, and a ``model_status`` tag for a
deprecated or unlisted model. It yields the LangChain callbacks the trace
destinations give for recording the call site's runs; for Opik, its LangChain
tracer, which names and prices every chat model span by the catalog
(``ol_llm.trace_destinations.opik_langchain``). Timeouts stay with the caller's
client; ``chat_litellm_kwargs`` sets a langchain-litellm ``ChatLiteLLM``'s from the
use case's policy. ``with_call_policy`` wraps a chat model so its calls retry
under that policy.

Needs the ``langchain`` extra.
"""

import asyncio
import email.utils
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn

import httpx
import httpx2
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    BaseCallbackHandler,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.base import LangSmithParams
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.runnables.graph import Graph
from pydantic import ConfigDict, InstanceOf

from ol_llm.attribution import NO_ATTRIBUTION, Attribution
from ol_llm.catalog import check_chat_model
from ol_llm.errors import LLMError, LLMRateLimited, LLMRejected, LLMUnavailable
from ol_llm.policy import CallPolicy, UseCase, resolve_policy
from ol_llm.retries import (
    RATE_LIMITED_STATUS,
    RETRYABLE_STATUSES,
    UNAVAILABLE_TRANSPORT_ERRORS,
    RetryBudget,
)
from ol_llm.traces import TracedCall, TraceOutput, current_call
from ol_llm.tracing import destinations, enabled

MILLISECONDS_PER_SECOND = 1000
UNMAPPED_ERROR_STATUS = 500
"""The status litellm gives an error it cannot map to a provider response."""

# Patched in tests so retries do not wait.
_sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
_sync_sleep: Callable[[float], Any] = time.sleep


def chat_litellm_kwargs(
    use_case: UseCase, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Return ``ChatLiteLLM`` settings for the use case's timeout, with retries off.

    The timeout becomes ``request_timeout``. litellm hands it to the provider's
    OpenAI-SDK client as the httpx timeout, which bounds each read, so on a stream
    it bounds the wait for the response and the gap between chunks, never the
    whole reply (checked in litellm 1.105.7 and openai 2.54.0).

    Both of the client's retry layers are turned off, so that ``with_call_policy``
    is the only one. The OpenAI SDK's (``model_kwargs["max_retries"]``) waits out
    a Retry-After of up to 120 s whatever is left of ``max_backoff_seconds``, and
    litellm caches an Azure client without keying on its retry count.
    ``ChatLiteLLM``'s own (``max_retries``, which counts attempts, so 1 is none)
    waits 4 to 10 s per retry with no budget. Do not set litellm's
    ``num_retries``, which retries the whole call on top.

    For example, with the caller's own ``model_kwargs`` merged in::

        settings = ol_llm.chat_litellm_kwargs(TUTOR)
        llm = ChatLiteLLM(
            model=model,
            streaming=True,
            **{**settings, "model_kwargs": {**settings["model_kwargs"], **extra}},
        )
        llm = ol_llm.with_call_policy(llm, TUTOR)

    :param use_case: the call site.
    :param environ: the environment ``resolve_policy`` reads; ``os.environ`` when
        omitted.
    :returns: ``request_timeout``, ``max_retries`` and ``model_kwargs`` for the
        ``ChatLiteLLM`` constructor.
    :rtype: dict[str, Any]
    :raises ValueError: if an environment override does not parse or is out of
        range.
    """
    policy = resolve_policy(use_case, environ)
    return {
        "request_timeout": policy.timeout_seconds,
        "max_retries": 1,
        "model_kwargs": {"max_retries": 0},
    }


def with_call_policy(
    model: BaseChatModel, use_case: UseCase, environ: Mapping[str, str] | None = None
) -> "PolicyChatModel":
    """Wrap a LangChain chat model so its calls retry under the use case's policy.

    It retries what the entry points retry (429, 408, 500, 502, 503, 504 and 529,
    and transport timeouts and disconnects), with their backoff and budget: on the
    request
    profile the waits share ``max_backoff_seconds``, and a Retry-After longer than
    what is left ends the call. A stream is retried only until its first chunk;
    a failure after that reaches the caller. The model's own client must not
    retry, or the layers multiply (``chat_litellm_kwargs`` turns a
    ``ChatLiteLLM``'s off). The timeout stays with the client.

    A failure is classified from its exception chain (``classify_chat_failure``),
    so no provider SDK is imported. It reaches the caller as the model raised it,
    not as an ``LLMError``. Retried attempts are not reported to callbacks; a
    trace shows one span. ``fallback_models`` is not applied.

    Set the model's own fields, e.g. ``temperature``, before wrapping, and bind
    tools on the wrapper: ``with_call_policy(llm, TUTOR).bind_tools(tools)``. The
    model's callbacks, tags, metadata, cache and rate limiter move to the wrapper.
    ``with_structured_output`` is refused, since LangChain's generic version would
    switch a model like ``ChatLiteLLM`` from its JSON-schema method to tool calls.

    :param model: the chat model, with its client's retries off.
    :param use_case: the call site.
    :param environ: the environment ``resolve_policy`` reads; ``os.environ`` when
        omitted.
    :returns: the wrapped model.
    :rtype: PolicyChatModel
    :raises ValueError: if an environment override does not parse or is out of
        range.
    """
    # The wrapper calls the model's generate and stream methods directly, so the
    # run-level settings that BaseChatModel reads move to the wrapper.
    return PolicyChatModel(
        inner=model,
        use_case=use_case,
        policy=resolve_policy(use_case, environ),
        callbacks=model.callbacks,
        tags=model.tags,
        metadata=model.metadata,
        cache=model.cache,
        rate_limiter=model.rate_limiter,
        verbose=model.verbose,
    )


def classify_chat_failure(
    exc: BaseException, *, use_case: str, model: str
) -> LLMError | None:
    """Map a LangChain chat model's failure to the library error for its class.

    The status comes from the HTTP response in the exception's chain
    (``original_exception``, ``__cause__``, ``__context__``) when there is one,
    else from an httpx timeout or disconnect there, else from the exception's
    ``status_code``. A bare 500 is a rejection: litellm gives 500 to whatever it
    cannot map, e.g. an error object inside a 200 stream or a failing credential
    hook, and nothing in it says the provider is down.

    :param exc: what the model raised.
    :param use_case: the use case name, for the error.
    :param model: the model id, for the error.
    :returns: the library error, or ``None`` when ``exc`` is not a model request
        failure.
    :rtype: LLMError | None
    """
    message = str(exc)
    chain = list(_exception_chain(exc))
    status = next(
        (
            link.response.status_code
            for link in chain
            if isinstance(link, httpx.HTTPStatusError | httpx2.HTTPStatusError)
        ),
        None,
    )
    if status is None:
        if any(isinstance(link, UNAVAILABLE_TRANSPORT_ERRORS) for link in chain):
            return LLMUnavailable(message, use_case=use_case, model=model)
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int) or isinstance(status, bool):
            return None
        if status == UNMAPPED_ERROR_STATUS:
            return LLMRejected(
                message, use_case=use_case, model=model, status_code=None
            )
    if status == RATE_LIMITED_STATUS:
        return LLMRateLimited(
            message, use_case=use_case, model=model, retry_after=_retry_after(chain)
        )
    if status in RETRYABLE_STATUSES:
        return LLMUnavailable(message, use_case=use_case, model=model)
    return LLMRejected(message, use_case=use_case, model=model, status_code=status)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and every exception it wraps, each once."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        link = pending.pop(0)
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        yield link
        original = getattr(link, "original_exception", None)
        pending.extend(
            (
                original if isinstance(original, BaseException) else None,
                link.__cause__,
                link.__context__,
            )
        )


def _retry_after(chain: Sequence[BaseException]) -> float | None:
    """Read the seconds a rate-limited response asked to wait, if it said."""
    for link in chain:
        if headers := getattr(link, "litellm_response_headers", None):
            return _parse_retry_after(headers)
    for link in chain:
        response = getattr(link, "response", None)
        if isinstance(response, httpx.Response | httpx2.Response) and response.headers:
            return _parse_retry_after(response.headers)
    return None


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    if (milliseconds := lowered.get("retry-after-ms")) is not None:
        try:
            return float(milliseconds) / MILLISECONDS_PER_SECOND
        except ValueError:
            pass
    if (value := lowered.get("retry-after")) is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(when.timestamp() - time.time(), 0.0)


class PolicyChatModel(BaseChatModel):
    """A chat model whose calls retry under a use case's policy.

    Built by ``with_call_policy``; see there for what it retries. It calls the
    wrapped model's own generate and stream methods with the same run manager,
    so the runs callbacks see are the wrapper's, one per call.

    :param inner: the wrapped chat model.
    :param use_case: the call site.
    :param policy: the resolved policy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: BaseChatModel
    use_case: InstanceOf[UseCase]
    policy: InstanceOf[CallPolicy]

    @property
    def _llm_type(self) -> str:
        return self.inner._llm_type  # noqa: SLF001

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return dict(self.inner._identifying_params)  # noqa: SLF001

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> LangSmithParams:
        return self.inner._get_ls_params(stop=stop, **kwargs)  # noqa: SLF001

    def _should_stream(
        self,
        *,
        async_api: bool,
        run_manager: CallbackManagerForLLMRun
        | AsyncCallbackManagerForLLMRun
        | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> bool:
        return self.inner._should_stream(  # noqa: SLF001
            async_api=async_api, run_manager=run_manager, **kwargs
        )

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,  # noqa: ANN401
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools the way the wrapped model binds them.

        :param tools: the tools.
        :returns: the wrapper bound with the wrapped model's tool settings.
        :rtype: Runnable[LanguageModelInput, AIMessage]
        :raises TypeError: if the wrapped model's ``bind_tools`` does not bind the
            model itself, so its settings cannot move to the wrapper.
        """
        bound = self.inner.bind_tools(tools, **kwargs)
        if not isinstance(bound, RunnableBinding) or bound.bound is not self.inner:
            msg = (
                f"{type(self.inner).__name__}.bind_tools did not bind the model "
                "itself, so its tools cannot be bound on the policy wrapper"
            )
            raise TypeError(msg)
        return self.bind(**bound.kwargs)

    def with_structured_output(
        self,
        schema: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> NoReturn:
        """Refuse, rather than change how the wrapped model asks for structure.

        LangChain's generic ``with_structured_output`` forces a tool call and
        drops ``method`` and ``strict``, where e.g. ``ChatLiteLLM`` defaults to a
        JSON-schema response format.

        :raises NotImplementedError: always.
        """
        msg = (
            "PolicyChatModel does not support with_structured_output: LangChain's "
            f"generic version would replace {type(self.inner).__name__}'s own method"
        )
        raise NotImplementedError(msg)

    def _delay(self, exc: Exception, budget: RetryBudget) -> float | None:
        error = classify_chat_failure(
            exc, use_case=self.use_case.name, model=self.use_case.model
        )
        return None if error is None else budget.next_delay(error)

    def _inner_streams(self, kwargs: Mapping[str, Any]) -> bool:
        # A model like ChatLiteLLM streams inside its own generate when its
        # streaming field is on, so that path goes through the stream's rule.
        return bool(kwargs.get("stream", getattr(self.inner, "streaming", False)))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ChatResult:
        if self._inner_streams(kwargs):
            return generate_from_stream(
                self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )
        budget = RetryBudget(self.policy, self.use_case.profile)
        while True:
            try:
                return self.inner._generate(  # noqa: SLF001
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except Exception as exc:
                if (delay := self._delay(exc, budget)) is None:
                    raise
            _sync_sleep(delay)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ChatResult:
        if self._inner_streams(kwargs):
            return await agenerate_from_stream(
                self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )
        budget = RetryBudget(self.policy, self.use_case.profile)
        while True:
            try:
                return await self.inner._agenerate(  # noqa: SLF001
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except Exception as exc:
                if (delay := self._delay(exc, budget)) is None:
                    raise
            await _sleep(delay)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Iterator[ChatGenerationChunk]:
        budget = RetryBudget(self.policy, self.use_case.profile)
        while True:
            started = False
            try:
                for chunk in self.inner._stream(  # noqa: SLF001
                    messages, stop=stop, run_manager=run_manager, **kwargs
                ):
                    started = True
                    yield chunk
            except Exception as exc:
                if started or (delay := self._delay(exc, budget)) is None:
                    raise
            else:
                return
            _sync_sleep(delay)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> AsyncIterator[ChatGenerationChunk]:
        budget = RetryBudget(self.policy, self.use_case.profile)
        while True:
            started = False
            try:
                async for chunk in self.inner._astream(  # noqa: SLF001
                    messages, stop=stop, run_manager=run_manager, **kwargs
                ):
                    started = True
                    yield chunk
            except Exception as exc:
                if started or (delay := self._delay(exc, budget)) is None:
                    raise
            else:
                return
            await _sleep(delay)


def _root_input(user_prompt: str | None) -> None:
    """Stand in for a traced function, so the call's input matches ``run``'s."""


@dataclass(frozen=True, kw_only=True)
class ChatTrace:
    """What ``traced_chat`` yields: the callbacks for the call site's runs.

    :param callbacks: pass these to the graph or chat model, e.g. in the
        ``config`` of ``astream``. Empty when the call is not traced.
    :param output: what the call records as its output once ``record`` is
        called.
    """

    callbacks: Sequence[BaseCallbackHandler]
    output: TraceOutput

    def record(self, output: object) -> None:
        """Record the reply the caller assembled as the call's output.

        Recorded only when ``include_content`` is on.

        :param output: the reply, e.g. the streamed text joined.
        """
        self.output.value = output


@contextmanager
def traced_chat(
    use_case: UseCase,
    user_prompt: str | None,
    *,
    attribution: Attribution = NO_ATTRIBUTION,
    metadata: Mapping[str, Any] | None = None,
    graph: Graph | None = None,
) -> Generator[ChatTrace]:
    """Trace the LangChain runs of one chat turn under the entry points' call span.

    Every chat model run the callbacks see must call the use case's model: its span
    names that model. A fallback the client tries when the primary fails is still
    named, and priced, as the use case's model. The block's exception passes through
    unchanged and the call records it. Inputs, outputs, and error messages are
    recorded only when ``include_content`` is on; without it an error keeps its type
    and frames.

    However the block exits, e.g. a ``break`` out of the stream or a cancelled task,
    the runs it left open end with it, marked ``cut_off`` rather than failed, and
    the caller's trace context is left as the block found it.

    The call is a child of the current trace, or a new trace. The runs attach to it
    whatever thread or task LangChain calls the callbacks from.

    For example, around a LangGraph agent's stream::

        with ol_llm.traced_chat(TUTOR, message, attribution=attribution) as chat:
            config = {"callbacks": [*chat.callbacks]}
            reply = "".join(
                [chunk async for chunk in stream(agent, message, config)]
            )
            chat.record(reply)

    :param use_case: the call site, naming a chat model in ``provider:model`` form.
    :param user_prompt: the user's message, recorded as the call's input.
    :param attribution: the per-call attribution fields.
    :param metadata: call-site fields for the spans of the runs, e.g. the course a
        tutor is answering about. The call keeps the §4 fields only.
    :param graph: the LangGraph graph, from ``get_graph(xray=True)``, for a
        destination that shows it (Opik's agent graph view).
    :returns: the trace, whose callbacks record the runs.
    :rtype: Generator[ChatTrace]
    :raises ValueError: if the catalog check rejects the use case.
    """
    call = TracedCall(
        use_case=use_case,
        attribution=attribution,
        fields=TracedCall.fields_for(
            use_case, attribution, model_tags=check_chat_model(use_case)
        ),
        traced=enabled(),
    )
    with call.trace_scope(_root_input, user_prompt) as output:
        if (entered := current_call()) is None:
            yield ChatTrace(callbacks=(), output=output)
            return
        with ExitStack() as stack:
            callbacks = [
                callback
                for destination in destinations()
                for callback in stack.enter_context(
                    destination.chat_callbacks(
                        entered,
                        use_case=use_case,
                        metadata=dict(metadata or {}),
                        graph=graph,
                    )
                )
            ]
            yield ChatTrace(callbacks=tuple(callbacks), output=output)
