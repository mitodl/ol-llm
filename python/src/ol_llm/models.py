"""Model construction: the only supported way to get a PydanticAI model (spec §6.3).

``build_model`` turns every provider SDK's own retries off, wraps each model in
``PolicyModel`` so there is exactly one retry layer and every request has a
timeout, and composes a declared fallback chain. Needs the ``pydantic-ai`` extra and
the extra for each provider a use case names.
"""

import asyncio
import os
import random
import sys
import threading
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
    contextmanager,
)
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Any, Generic, NoReturn, Self, TypeVar, cast, overload

import anyio
import httpx
import httpx2
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    Instrumentation,
    RaiseContentFilterError,
)
from pydantic_ai.exceptions import (
    ContentFilterError,
    FallbackExceptionGroup,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    StreamedResponse,
    infer_model,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import NativeOutput, OutputSpec
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.usage import RequestUsage, UsageLimits

from ol_llm.attribution import USE_CASE_HEADER, Attribution
from ol_llm.catalog import CatalogEntry, catalog_entries, status_tags
from ol_llm.errors import (
    LLMBudgetExceeded,
    LLMContentFiltered,
    LLMError,
    LLMInvalidOutput,
    LLMRateLimited,
    LLMRejected,
    LLMUnavailable,
)
from ol_llm.policy import CallPolicy, Profile, UseCase, resolve_policy
from ol_llm.settings import GatewayRoute, Settings, get_settings
from ol_llm.tracing import include_content, tracer_provider

OutputT = TypeVar("OutputT")
_T = TypeVar("_T")
_V = TypeVar("_V")
_NO_DEPS: type[None] = type(None)
DepsT = TypeVar("DepsT")

RETRYABLE_STATUSES = frozenset({408, 500, 502, 503, 504, 529})
"""HTTP statuses that mean the provider is unavailable (spec §3.2 rule 2)."""

RATE_LIMITED_STATUS = 429
INITIAL_BACKOFF_SECONDS = 0.5
FALLBACK_ON = (LLMRateLimited, LLMUnavailable)

# Transport failures that mean no answer came back. Google and Mistral raise these
# raw from httpx2 rather than as ModelAPIError (spec §6.1); any other transport
# error is permanent (a bad URL scheme, a proxy failure) and is not retried.
_UNAVAILABLE_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx2.TimeoutException,
    httpx2.NetworkError,
    httpx2.RemoteProtocolError,
)
_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    httpx.TransportError,
    httpx2.TransportError,
)

_OPENAI_PROVIDERS = frozenset({"openai", "openai-chat", "openai-responses"})
_AZURE_PROVIDERS = frozenset({"azure", "azure-responses"})
# TogetherProvider's own base URL; the provider cannot be given a client built
# with retries off any other way.
_TOGETHER_BASE_URL = "https://api.together.xyz/v1"

SUPPORTED_PROVIDERS = frozenset(
    {
        *_OPENAI_PROVIDERS,
        *_AZURE_PROVIDERS,
        "anthropic",
        "bedrock",
        "google",
        "google-cloud",
        "mistral",
        "together",
    }
)

# Tests may replace the provider models build_model creates; see ol_llm.testing.
_model_override: ContextVar[Model | Mapping[str, Model] | None] = ContextVar(
    "ol_llm_model_override", default=None
)

# Where recording_answers collects model ids. A list, not a value: the run's
# tasks get a copy of the context, and an append is what reaches the caller.
_answers: ContextVar[list[str] | None] = ContextVar("ol_llm_answers", default=None)


class LoopLocal(threading.local, Generic[_V]):
    """A value built once for each event loop that uses it, kept by the loop's thread.

    For what holds provider clients, which bind to the loop they first run on: a
    pooled ``httpx`` connection belongs to that loop's transport, and SDK clients
    keep asyncio locks and events (a provider's own enter lock among them). Shared
    across loops, they fail with "bound to a different event loop", or "Event loop
    is closed" once the first loop is gone, which is what ``run_sync`` from a
    thread pool and ``asyncio.run`` per task do. A loop runs on one thread, so
    keeping the values per thread loses nothing, and a worker thread's loop, which
    ``run_sync`` never closes, is released with the thread.

    :param build: builds the value; called again for each new loop.
    """

    def __init__(self, build: Callable[[], _V]) -> None:
        """Start empty; ``threading.local`` runs this again in each new thread."""
        self._build = build
        self._values: dict[asyncio.AbstractEventLoop, _V] = {}

    def get(self) -> _V:
        """Return the running loop's value, dropping the values of closed loops.

        :returns: the value.
        :rtype: _V
        :raises RuntimeError: if no event loop is running.
        """
        loop = asyncio.get_running_loop()
        for closed in [other for other in self._values if other.is_closed()]:
            del self._values[closed]
        if (value := self._values.get(loop)) is None:
            value = self._values[loop] = self._build()
        return value


# The models complete runs on, by use case, with the settings each was built with.
# The lock keeps two threads' first calls from building a use case twice, which for
# Bedrock would also create boto3 clients concurrently from the default session.
_shared_models: dict[UseCase, tuple[Settings, Model]] = {}
_shared_models_lock = threading.Lock()

# Patched in tests so retries do not wait.
_sleep: Callable[[float], Any] = asyncio.sleep


def classify_failure(  # noqa: PLR0911
    exc: BaseException, *, use_case: str, model: str
) -> LLMError | None:
    """Map a failed model request to the library error for its class.

    The table is closed: every ``ModelAPIError``, every ``httpx`` or ``httpx2``
    transport error, and every botocore transport error maps to a class, so
    nothing a provider raises is unclassified. An exception group, which
    Bedrock's Titan embedding model raises for its concurrent per-text requests,
    maps to its first error that is not retryable, else its first error.

    :param exc: what the request raised.
    :param use_case: the use case name, for the error.
    :param model: the model id, for the error.
    :returns: the library error, or ``None`` when ``exc`` is not a model request
        failure and should propagate unchanged.
    :rtype: LLMError | None
    """
    if isinstance(exc, BaseExceptionGroup):
        return _classify_group(exc, use_case=use_case, model=model)
    message = str(exc)
    if isinstance(exc, ModelHTTPError):
        if exc.status_code == RATE_LIMITED_STATUS:
            return LLMRateLimited(
                message, use_case=use_case, model=model, retry_after=exc.retry_after
            )
        if exc.status_code in RETRYABLE_STATUSES:
            return LLMUnavailable(message, use_case=use_case, model=model)
        return LLMRejected(
            message, use_case=use_case, model=model, status_code=exc.status_code
        )
    unavailable: tuple[type[Exception], ...] = (
        ModelAPIError,
        *_UNAVAILABLE_TRANSPORT_ERRORS,
        *_botocore_transport_errors(),
    )
    if isinstance(exc, unavailable):
        return LLMUnavailable(message, use_case=use_case, model=model)
    if isinstance(exc, _TRANSPORT_ERRORS):
        return LLMRejected(message, use_case=use_case, model=model, status_code=None)
    return None


def _classify_group(
    group: BaseExceptionGroup[BaseException], *, use_case: str, model: str
) -> LLMError | None:
    errors = [
        classify_failure(exc, use_case=use_case, model=model)
        for exc in group.exceptions
    ]
    classified = [error for error in errors if error is not None]
    if len(classified) < len(errors):
        return None
    return next((e for e in classified if not e.retryable), classified[0])


def _botocore_transport_errors() -> tuple[type[Exception], ...]:
    """Return the botocore errors that mean no answer came back.

    PydanticAI's Bedrock chat model wraps these in ``ModelAPIError``; its embedding
    model catches only ``ClientError`` and lets them through raw. Without botocore
    imported, none can have been raised.
    """
    if (botocore := sys.modules.get("botocore.exceptions")) is None:
        return ()
    return (botocore.HTTPClientError, botocore.ConnectionError)


def classify_run_failure(
    exc: BaseException, *, use_case: str, model: str
) -> LLMError | None:
    """Map a failed agent run to the library error callers see (spec §6.7).

    :param exc: what the run raised.
    :param use_case: the use case name, for the error.
    :param model: the model id, for the error.
    :returns: the library error, or ``None`` when ``exc`` is not a model failure.
    :rtype: LLMError | None
    """
    message = str(exc)
    if isinstance(exc, LLMError):
        return exc
    # ContentFilterError subclasses UnexpectedModelBehavior, so it goes first.
    if isinstance(exc, ContentFilterError):
        return LLMContentFiltered(message, use_case=use_case, model=model)
    if isinstance(exc, UnexpectedModelBehavior):
        return LLMInvalidOutput(message, use_case=use_case, model=model)
    if isinstance(exc, UsageLimitExceeded):
        return LLMBudgetExceeded(message, use_case=use_case, model=model)
    return None


@dataclass
class _RetryBudget:
    policy: CallPolicy
    profile: Profile
    retries: int = 0
    waited: float = 0

    def next_delay(self, error: LLMError) -> float | None:
        """Return how long to wait before the next attempt, or None to give up."""
        if not error.retryable or self.retries >= self.policy.max_retries:
            return None
        backoff = INITIAL_BACKOFF_SECONDS * 2**self.retries
        # Jitter spreads out callers that failed together; it is not security.
        delay = random.uniform(backoff / 2, backoff)  # noqa: S311
        retry_after = error.retry_after if isinstance(error, LLMRateLimited) else None
        if retry_after is not None:
            delay = max(delay, retry_after)
        if self.profile == "request":
            # A total budget: a Retry-After longer than what is left ends the call
            # so the caller can tell the user instead of holding a request open.
            # Once the budget is spent, retrying with no wait would hammer a
            # provider that is already failing.
            remaining = self.policy.max_backoff_seconds - self.waited
            if remaining <= 0 or (retry_after is not None and retry_after > remaining):
                return None
            delay = min(delay, remaining)
        else:
            delay = min(delay, self.policy.max_backoff_seconds)
        self.retries += 1
        self.waited += delay
        return delay


async def retrying(
    attempt: Callable[[], Awaitable[_T]],
    *,
    policy: CallPolicy,
    profile: Profile,
    use_case: str,
    model: str,
) -> _T:
    """Make one call under the retry policy.

    :param attempt: makes one attempt.
    :param policy: the resolved policy.
    :param profile: the use case's profile, which decides how backoff is capped.
    :param use_case: the use case name, for the error.
    :param model: the model id, for the error.
    :returns: what the first successful attempt returned.
    :rtype: _T
    :raises LLMError: once a failure is not retryable or the retries are spent,
        with the provider's exception as ``__cause__``. Anything that is not a
        model request failure is re-raised unchanged.
    """
    budget = _RetryBudget(policy, profile)
    while True:
        try:
            return await attempt()
        except Exception as exc:  # noqa: BLE001
            delay = _retry_delay(exc, budget, use_case=use_case, model=model)
        await _sleep(delay)


def _retry_delay(
    exc: Exception, budget: _RetryBudget, *, use_case: str, model: str
) -> float:
    """Return the wait before the next attempt, or raise what the caller sees.

    Callers catch ``Exception`` and hand it here; anything not classified is
    re-raised unchanged.
    """
    if (error := classify_failure(exc, use_case=use_case, model=model)) is None:
        raise exc
    if (delay := budget.next_delay(error)) is None:
        raise error from exc
    return delay


_METADATA_KEY = "ol_llm"


def _stamp_answered_by(
    response: ModelResponse | StreamedResponse, catalog_id: str
) -> None:
    """Record which chain member answered, where the run's history keeps it."""
    if response.metadata is None:
        response.metadata = {}
    response.metadata.setdefault(_METADATA_KEY, {})["model"] = catalog_id
    if (answers := _answers.get()) is not None:
        answers.append(catalog_id)


@contextmanager
def recording_answers() -> Generator[list[str]]:
    """Collect the model id of every response given inside the block, in order.

    A run that fails on a response (invalid output, a content filter) has no
    history to read, so this is how its error names the model that gave it.

    :returns: the list the ids are appended to.
    :rtype: list[str]
    """
    answers: list[str] = []
    token = _answers.set(answers)
    try:
        yield answers
    finally:
        _answers.reset(token)


def _answered_by(response: ModelResponse) -> str | None:
    answered_by: str | None = (
        (response.metadata or {}).get(_METADATA_KEY, {}).get("model")
    )
    return answered_by


def answering_model(messages: Sequence[ModelMessage]) -> str | None:
    """Return the model id that gave the last response in a run's messages.

    :param messages: the run's messages.
    :returns: the id, or ``None`` when no response came from a ``PolicyModel``.
    :rtype: str | None
    """
    return next(
        (
            _answered_by(message)
            for message in reversed(messages)
            if isinstance(message, ModelResponse)
        ),
        None,
    )


_NO_EVENT = object()


@dataclass
class _PrimedEvents:
    """A Bedrock event stream whose first event has already been read."""

    first: object
    rest: Iterator[object]
    source: Any

    def __iter__(self) -> Iterator[object]:
        if self.first is not _NO_EVENT:
            yield self.first
        yield from self.rest

    def close(self) -> None:
        self.source.close()


async def _read_first_bedrock_event(stream: StreamedResponse) -> None:
    """Read a Bedrock stream's first event while the stream can still be retried.

    Every other PydanticAI provider reads the first chunk before handing back its
    stream. BedrockModel does not, so a throttle or outage on the first read would
    skip both retries and fallback. The event is put back for the consumer. This
    reaches into BedrockStreamedResponse's private fields; remove it once
    PydanticAI reads the first event itself:
    https://github.com/pydantic/pydantic-ai/issues/10364
    """
    bedrock = sys.modules.get("pydantic_ai.models.bedrock")
    if bedrock is None or not isinstance(stream, bedrock.BedrockStreamedResponse):
        return
    primed = cast("Any", stream)
    source = primed._event_stream  # noqa: SLF001
    events = iter(source)
    try:
        with bedrock._map_api_errors(stream.model_name):  # noqa: SLF001
            first = await anyio.to_thread.run_sync(next, events, _NO_EVENT)
    except Exception:
        # Leaving the provider's context does not close Bedrock's event stream.
        await stream.cancel()
        raise
    primed._event_stream = _PrimedEvents(first, events, source)  # noqa: SLF001


@dataclass(init=False)
class PolicyModel(WrapperModel):
    """A provider model under the call policy: one retry layer and a timeout.

    Every request gets the policy's timeout, whatever the caller passed. Failures
    are classified by type and status; retryable ones are retried with
    exponential backoff and jitter, and what reaches the caller is an ``LLMError``
    with the provider's exception as ``__cause__``. A stream is retried while it
    opens; a failure after that reaches the caller.

    Requests go to a provider model built for the running event loop (see
    ``LoopLocal``). ``wrapped`` answers for the model's name, profile, and settings
    and never makes a request.
    """

    use_case: str
    call_site: UseCase
    catalog_id: str
    catalog_entry: CatalogEntry | None
    profile_name: Profile
    policy: CallPolicy

    def __init__(  # noqa: PLR0913
        self,
        wrapped: Model,
        *,
        use_case: UseCase,
        model_id: str,
        catalog_entry: CatalogEntry | None,
        policy: CallPolicy,
        build: Callable[[], Model] | None = None,
    ) -> None:
        """Wrap a provider model.

        :param wrapped: the provider model, with its SDK's own retries off.
        :param use_case: the call site.
        :param model_id: the id the model was built from, for errors.
        :param catalog_entry: the model's catalog entry, for tracing to tag a
            deprecated model; ``None`` for a model the catalog does not list.
        :param policy: the resolved policy.
        :param build: builds the same provider model again, once for each event
            loop requests run on; ``None`` sends every request to ``wrapped``.
        """
        super().__init__(wrapped)
        self.use_case = use_case.name
        self.call_site = use_case
        self.catalog_id = model_id
        self.catalog_entry = catalog_entry
        self.profile_name = use_case.profile
        self.policy = policy
        self._loop_models = None if build is None else LoopLocal(build)

    def _loop_model(self) -> Model:
        """Return the provider model for the running event loop."""
        if self._loop_models is None:
            return self.wrapped
        return self._loop_models.get()

    async def __aenter__(self) -> Self:
        """Enter the running loop's provider model, for its HTTP client lifecycle."""
        await self._loop_model().__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit the running loop's provider model."""
        return await self._loop_model().__aexit__(exc_type, exc_val, exc_tb)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """Count tokens with the running loop's provider model."""
        return await self._loop_model().count_tokens(
            messages, model_settings, model_request_parameters
        )

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        """Cancel a suspended response with the running loop's provider model."""
        await self._loop_model().cancel_suspended_response(response)

    async def compact_messages(
        self,
        request_context: ModelRequestContext,
        *,
        instructions: str | None = None,
    ) -> ModelResponse:
        """Compact messages with the running loop's provider model."""
        return await self._loop_model().compact_messages(
            request_context, instructions=instructions
        )

    def _with_timeout(self, model_settings: ModelSettings | None) -> ModelSettings:
        timeout = ModelSettings(timeout=self.policy.timeout_seconds)
        return merge_model_settings(model_settings, timeout) or timeout

    def _classify(self, exc: BaseException) -> LLMError | None:
        return classify_failure(exc, use_case=self.use_case, model=self.catalog_id)

    def _retry_delay(self, exc: Exception, budget: _RetryBudget) -> float:
        return _retry_delay(exc, budget, use_case=self.use_case, model=self.catalog_id)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make the request, retrying transient failures under the policy."""
        settings = self._with_timeout(model_settings)
        provider = self._loop_model()
        response = await retrying(
            lambda: provider.request(messages, settings, model_request_parameters),
            policy=self.policy,
            profile=self.profile_name,
            use_case=self.use_case,
            model=self.catalog_id,
        )
        _stamp_answered_by(response, self.catalog_id)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Open a stream, retrying transient failures until the first chunk."""
        settings = self._with_timeout(model_settings)
        budget = _RetryBudget(self.policy, self.profile_name)
        provider = self._loop_model()
        while True:
            stream_cm = provider.request_stream(
                messages, settings, model_request_parameters, run_context
            )
            try:
                stream = await stream_cm.__aenter__()
            except Exception as exc:  # noqa: BLE001
                await _sleep(self._retry_delay(exc, budget))
                continue
            try:
                await _read_first_bedrock_event(stream)
            except Exception as exc:  # noqa: BLE001
                await stream_cm.__aexit__(type(exc), exc, exc.__traceback__)
                await _sleep(self._retry_delay(exc, budget))
                continue
            break
        _stamp_answered_by(stream, self.catalog_id)
        try:
            yield stream
        except Exception as exc:
            error = self._classify(exc)
            suppressed = await stream_cm.__aexit__(type(exc), exc, exc.__traceback__)
            if error is not None:
                raise error from exc
            if not suppressed:
                raise
        except BaseException as exc:
            await stream_cm.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            await stream_cm.__aexit__(None, None, None)


@dataclass(init=False)
class PolicyFallbackModel(FallbackModel):
    """A fallback chain of ``PolicyModel``s that raises the primary's error.

    Each model in the chain spends its own retries before the next is tried, and
    only rate limits and outages fall back (spec §6.4). When every model fails the
    caller gets the error of the first model tried, with the whole group as
    ``__cause__``.

    A request that asks the model to retry its last response (invalid output, or a
    tool's ``ModelRetry``) starts at the model that gave that response, so the retry
    neither waits out an outage the run already fell back from nor gets answered
    by a model that never saw the mistake.

    Each run enters and exits every member itself. ``FallbackModel`` keeps one
    enter count behind an anyio lock for all runs, so runs on two threads' loops
    would wait on each other's lock and exit a member they did not enter.
    """

    async def __aenter__(self) -> Self:
        """Enter every member for the running loop."""
        async with AsyncExitStack() as stack:
            for model in self.models:
                await stack.enter_async_context(model)
            stack.pop_all()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit every member for the running loop, last entered first.

        Each member gets the exception the run is exiting with, and the result says
        whether a member suppressed it.
        """
        stack = AsyncExitStack()
        for model in self.models:
            stack.push_async_exit(model)
        return await stack.__aexit__(exc_type, exc_val, exc_tb)

    def _starting_at(self, messages: list[ModelMessage]) -> FallbackModel:
        """Return the chain a request runs on, read from the run's own history."""
        last = messages[-1] if messages else None
        if not isinstance(last, ModelRequest) or not any(
            isinstance(part, RetryPromptPart) for part in last.parts
        ):
            return self
        answered_by = answering_model(messages)
        ids = [cast("PolicyModel", model).catalog_id for model in self.models]
        if answered_by not in ids[1:]:
            return self
        start = ids.index(answered_by)
        return FallbackModel(*self.models[start:], fallback_on=FALLBACK_ON)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Try each model in turn."""
        try:
            return await FallbackModel.request(
                self._starting_at(messages),
                messages,
                model_settings,
                model_request_parameters,
            )
        except FallbackExceptionGroup as group:
            if (primary := _primary_error(group)) is None:
                raise
            raise primary from group

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Open a stream on each model in turn; fall back only while opening."""
        stream_cm = FallbackModel.request_stream(
            self._starting_at(messages),
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        )
        try:
            stream = await stream_cm.__aenter__()
        except FallbackExceptionGroup as group:
            if (primary := _primary_error(group)) is None:
                raise
            raise primary from group
        async with _exit_with(stream_cm):
            yield stream


@asynccontextmanager
async def _exit_with(
    stream_cm: AbstractAsyncContextManager[Any],
) -> AsyncGenerator[None]:
    """Exit an already-entered async context manager when this one exits."""
    try:
        yield
    except BaseException as exc:
        if not await stream_cm.__aexit__(type(exc), exc, exc.__traceback__):
            raise
    else:
        await stream_cm.__aexit__(None, None, None)


def _primary_error(group: FallbackExceptionGroup) -> LLMError | None:
    """Return a copy of the primary model's error to raise in place of the group."""
    primary = group.exceptions[0]
    return primary.copy() if isinstance(primary, LLMError) else None


def _gateway_headers(route: GatewayRoute | None, use_case: UseCase) -> dict[str, str]:
    return {USE_CASE_HEADER: use_case.name} if route is not None else {}


def _raise_missing_extra(extra: str, exc: ImportError) -> NoReturn:
    msg = f"This model needs the ol-llm[{extra}] extra: {exc}"
    raise ImportError(msg) from exc


# The clients of providers with embedding models carry the policy timeout too.
# Embedding requests have no per-request timeout to set; chat requests override
# the client's with ModelSettings.timeout.


def _openai_provider(
    route: GatewayRoute | None, use_case: UseCase, policy: CallPolicy
) -> Provider[Any]:
    try:
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient  # noqa: PLC0415
        from pydantic_ai.providers.openai import OpenAIProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("openai", exc)
    client = AsyncOpenAI(
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(),
        timeout=policy.timeout_seconds,
        base_url=route.base_url if route else None,
        api_key=route.api_key if route else None,
        default_headers=_gateway_headers(route, use_case),
    )
    return OpenAIProvider(openai_client=client)


def _azure_provider(settings: Settings, policy: CallPolicy) -> Provider[Any]:
    try:
        from openai import AsyncAzureOpenAI, DefaultAsyncHttpxClient  # noqa: PLC0415
        from pydantic_ai.providers.azure import AzureProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("openai", exc)
    # Endpoint, API version, and key come from AZURE_OPENAI_ENDPOINT,
    # OPENAI_API_VERSION, and AZURE_OPENAI_API_KEY unless a token provider is set.
    client = AsyncAzureOpenAI(
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(),
        timeout=policy.timeout_seconds,
        azure_ad_token_provider=settings.azure_ad_token_provider,
    )
    return AzureProvider(openai_client=client)


def _anthropic_provider(route: GatewayRoute | None, use_case: UseCase) -> Provider[Any]:
    try:
        from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient  # noqa: PLC0415
        from pydantic_ai.providers.anthropic import AnthropicProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("anthropic", exc)
    client = AsyncAnthropic(
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(),
        base_url=route.base_url if route else None,
        api_key=route.api_key if route else None,
        default_headers=_gateway_headers(route, use_case),
    )
    return AnthropicProvider(anthropic_client=client)


def _bedrock_client(policy: CallPolicy) -> Any:  # noqa: ANN401
    try:
        import boto3  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("bedrock", exc)
    # Bedrock ignores ModelSettings.timeout, so the timeout lives on the client,
    # and the client is the only place botocore's own retries can be turned off.
    config = Config(
        read_timeout=policy.timeout_seconds,
        connect_timeout=policy.timeout_seconds,
        retries={"total_max_attempts": 1},
    )
    return boto3.client("bedrock-runtime", config=config)


def _bedrock_provider(client: Any) -> Provider[Any]:  # noqa: ANN401
    try:
        from pydantic_ai.providers.bedrock import BedrockProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("bedrock", exc)
    return BedrockProvider(bedrock_client=client)


def _google_provider(provider: str, policy: CallPolicy) -> Provider[Any]:
    try:
        from pydantic_ai.providers.google import GoogleProvider  # noqa: PLC0415
        from pydantic_ai.providers.google_cloud import (  # noqa: PLC0415
            GoogleCloudProvider,
        )
    except ImportError as exc:
        _raise_missing_extra("google", exc)
    # The SDK retries only when given a retry config, and none is passed. The
    # provider reads the request timeout from the client's read timeout.
    new_client = partial(httpx2.AsyncClient, timeout=policy.timeout_seconds)
    http_client = new_client()
    google: Provider[Any] = (
        GoogleCloudProvider(http_client=http_client)
        if provider == "google-cloud"
        else GoogleProvider(http_client=http_client)
    )
    # A provider leaves a client it was given open. Give it this one as if it had
    # made it, so exiting the provider closes it and entering again makes a new
    # one, as without a client.
    google._own_http_client = http_client  # noqa: SLF001
    google._http_client_factory = new_client  # noqa: SLF001
    return google


def _mistral_provider() -> Provider[Any]:
    try:
        from pydantic_ai.providers.mistral import MistralProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("mistral", exc)
    # The SDK retries only when given a retry config, and none is passed.
    return MistralProvider()


def _together_provider() -> Provider[Any]:
    try:
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient  # noqa: PLC0415
        from pydantic_ai.providers.together import TogetherProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("openai", exc)
    # AsyncOpenAI reads OPENAI_API_KEY by default; Together's key has its own name.
    client = AsyncOpenAI(
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(),
        base_url=_TOGETHER_BASE_URL,
        api_key=os.environ["TOGETHER_API_KEY"],
    )
    return TogetherProvider(openai_client=client)


def provider_factory(
    use_case: UseCase, policy: CallPolicy, settings: Settings
) -> Callable[[str], Provider[Any]]:
    """Return the factory that builds each provider with its SDK's retries off.

    :param use_case: the call site, for the gateway header.
    :param policy: the resolved policy, for the client timeouts.
    :param settings: the gateway routes and Azure credentials.
    :returns: a factory from a provider name to the provider.
    :rtype: Callable[[str], Provider[Any]]

    It runs again for each event loop a model's requests run on (see
    ``PolicyModel``). The boto3 client is synchronous, so every loop shares the
    first one built: creating clients from boto3's default session on several
    threads at once is not thread-safe.
    """
    bedrock_clients: list[Any] = []

    def factory(provider: str) -> Provider[Any]:  # noqa: PLR0911
        if provider in _OPENAI_PROVIDERS:
            return _openai_provider(
                settings.gateway_routes.get("openai"), use_case, policy
            )
        if provider in _AZURE_PROVIDERS:
            return _azure_provider(settings, policy)
        if provider == "anthropic":
            return _anthropic_provider(
                settings.gateway_routes.get("anthropic"), use_case
            )
        if provider == "bedrock":
            if not bedrock_clients:
                bedrock_clients.append(_bedrock_client(policy))
            return _bedrock_provider(bedrock_clients[0])
        if provider in {"google", "google-cloud"}:
            return _google_provider(provider, policy)
        if provider == "mistral":
            return _mistral_provider()
        if provider == "together":
            return _together_provider()
        msg = (
            f"ol-llm does not build {provider!r} models; "
            f"supported providers are {sorted(SUPPORTED_PROVIDERS)}"
        )
        raise ValueError(msg)

    return factory


def _provider_model(
    model_id: str,
    override: Model | Mapping[str, Model] | None,
    factory: Callable[[str], Provider[Any]] | None,
) -> Model:
    if override is not None:
        return override if isinstance(override, Model) else override[model_id]
    if factory is None:  # pragma: no cover - set whenever override is None
        raise AssertionError
    return infer_model(model_id, provider_factory=factory)


def build_model(use_case: UseCase, *, settings: Settings | None = None) -> Model:
    """Build the model for a use case: the primary and any fallbacks, under policy.

    Build once per use case and reuse it: each call creates new provider clients.

    :param use_case: the call site.
    :param settings: overrides the configured settings, mainly for tests.
    :returns: a ``PolicyModel``, or a ``PolicyFallbackModel`` over one per model id
        when the use case declares fallbacks.
    :rtype: Model
    :raises ValueError: on a model id not in ``provider:model`` form or that the
        catalog lists as retired or as an embedding model, fallbacks behind a
        route that falls back on its own, a provider the library does not build,
        or an environment policy value that does not parse.
    """
    entries = catalog_entries(use_case, kind="chat")
    policy = resolve_policy(use_case)
    override = _model_override.get()
    factory = (
        None
        if override is not None
        else provider_factory(use_case, policy, settings or get_settings())
    )
    models = [
        PolicyModel(
            _provider_model(model_id, override, factory),
            use_case=use_case,
            model_id=model_id,
            catalog_entry=entry,
            policy=policy,
            # An override's models are the test's own and are not rebuilt.
            build=None
            if factory is None
            else partial(infer_model, model_id, provider_factory=factory),
        )
        for model_id, entry in zip(use_case.model_ids, entries, strict=True)
    ]
    if len(models) == 1:
        return models[0]
    return PolicyFallbackModel(*models, fallback_on=FALLBACK_ON)


def _chain(model: Model) -> Sequence[Model]:
    return model.models if isinstance(model, FallbackModel) else (model,)


def call_site(model: Model) -> UseCase:
    """Return the use case a model was built for.

    :param model: a model from ``build_model``.
    :returns: the use case.
    :rtype: UseCase
    :raises TypeError: if the model did not come from ``build_model``.
    """
    primary = _chain(model)[0]
    if not isinstance(primary, PolicyModel):
        msg = f"{model!r} was not built by ol_llm.build_model"
        raise TypeError(msg)
    return primary.call_site


def catalog_tags(model: Model) -> tuple[str, ...]:
    """Return the trace tags for the catalog status of a model's chain.

    ``model_status:deprecated`` when any member is deprecated, and
    ``model_status:uncataloged`` when the catalog does not list one, so Opik
    filters and rules can find the traces that still need a catalog change. The
    tags describe the chain the use case names, whether or not a fallback answered,
    so they count traces to fix, not calls to each model.

    :param model: a model from ``build_model``.
    :returns: the tags, in a stable order.
    :rtype: tuple[str, ...]
    :raises TypeError: if a member of the chain did not come from ``build_model``.
    """
    entries = []
    for member in _chain(model):
        if not isinstance(member, PolicyModel):
            msg = f"{model!r} was not built by ol_llm.build_model"
            raise TypeError(msg)
        entries.append(member.catalog_entry)
    return status_tags(entries)


def instrumentation_settings(use_case: UseCase) -> InstrumentationSettings | None:
    """Return the settings that send a use case's model spans to Opik.

    :param use_case: the call site; its app is the Opik project.
    :returns: the settings, or ``None`` when tracing is off.
    :rtype: InstrumentationSettings | None
    """
    if (provider := tracer_provider(use_case.app)) is None:
        return None
    content = include_content()
    return InstrumentationSettings(
        tracer_provider=provider,
        include_content=content,
        include_binary_content=content,
    )


def instrumentation(use_case: UseCase) -> list[AbstractCapability[Any]]:
    """Return the capability that sends a use case's model spans to Opik.

    :param use_case: the call site; its app is the Opik project.
    :returns: the ``Instrumentation`` capability, or nothing when tracing is off.
    :rtype: list[AbstractCapability[Any]]
    """
    if (settings := instrumentation_settings(use_case)) is None:
        return []
    return [Instrumentation(settings=settings)]


def _uses_native_output(output_type: object) -> bool:
    # PydanticAI flattens nested sequences, so [NativeOutput(X), None] is native too.
    if isinstance(output_type, NativeOutput):
        return True
    if isinstance(output_type, Sequence) and not isinstance(output_type, str):
        return any(_uses_native_output(member) for member in output_type)
    return False


def _check_output_mode(output_type: object, model: Model) -> None:
    if not _uses_native_output(output_type):
        return
    for member in _chain(model):
        if not member.profile.get("supports_json_schema_output", False):
            label = getattr(member, "catalog_id", member.model_name)
            msg = f"{label} does not support NativeOutput; use ToolOutput"
            raise ValueError(msg)


@overload
def build_agent(
    use_case: UseCase,
    *,
    instructions: str | None = None,
    tools: Sequence[Any] = (),
    toolsets: Sequence[Any] | None = None,
    capabilities: Sequence[Any] | None = None,
    settings: Settings | None = None,
) -> Agent[None, str]: ...
@overload
def build_agent(
    use_case: UseCase,
    *,
    output_type: OutputSpec[OutputT],
    instructions: str | None = None,
    tools: Sequence[Any] = (),
    toolsets: Sequence[Any] | None = None,
    capabilities: Sequence[Any] | None = None,
    settings: Settings | None = None,
) -> Agent[None, OutputT]: ...
@overload
def build_agent(
    use_case: UseCase,
    *,
    deps_type: type[DepsT],
    instructions: str | None = None,
    tools: Sequence[Any] = (),
    toolsets: Sequence[Any] | None = None,
    capabilities: Sequence[Any] | None = None,
    settings: Settings | None = None,
) -> Agent[DepsT, str]: ...
@overload
def build_agent(
    use_case: UseCase,
    *,
    deps_type: type[DepsT],
    output_type: OutputSpec[OutputT],
    instructions: str | None = None,
    tools: Sequence[Any] = (),
    toolsets: Sequence[Any] | None = None,
    capabilities: Sequence[Any] | None = None,
    settings: Settings | None = None,
) -> Agent[DepsT, OutputT]: ...
def build_agent(  # noqa: PLR0913
    use_case: UseCase,
    *,
    deps_type: type[Any] = _NO_DEPS,
    output_type: OutputSpec[Any] = str,
    instructions: str | None = None,
    tools: Sequence[Any] = (),
    toolsets: Sequence[Any] | None = None,
    capabilities: Sequence[Any] | None = None,
    settings: Settings | None = None,
) -> Agent[Any, Any]:
    """Build a PydanticAI agent over the use case's model.

    The agent raises on a content-filtered response even when the provider sent
    partial text, retries invalid output on the same model
    ``use_case.output_retries`` times, and sends its spans to Opik unless tracing
    is off.

    :param use_case: the call site.
    :param deps_type: the type of the run's dependencies.
    :param output_type: the output spec; ``ToolOutput`` is the default mode, and
        ``NativeOutput`` needs every model in the chain to support it.
    :param instructions: the agent's instructions.
    :param tools: function tools.
    :param toolsets: toolsets.
    :param capabilities: capabilities added after the library's own.
    :param settings: overrides the configured settings, mainly for tests.
    :returns: the agent.
    :rtype: Agent
    :raises ValueError: if ``NativeOutput`` is asked of a model that lacks it, or
        ``build_model`` rejects the use case.
    """
    return agent_over(
        build_model(use_case, settings=settings),
        deps_type=deps_type,
        output_type=output_type,
        instructions=instructions,
        tools=tools,
        toolsets=toolsets,
        capabilities=capabilities,
    )


def agent_over(  # noqa: PLR0913
    model: Model,
    *,
    deps_type: type[Any] = _NO_DEPS,
    output_type: OutputSpec[Any] = str,
    instructions: str | None = None,
    tools: Sequence[Any] = (),
    toolsets: Sequence[Any] | None = None,
    capabilities: Sequence[Any] | None = None,
) -> Agent[Any, Any]:
    """Build the agent ``build_agent`` would, over a model already built.

    :param model: a model from ``build_model``.
    :param deps_type: the type of the run's dependencies.
    :param output_type: the output spec.
    :param instructions: the agent's instructions.
    :param tools: function tools.
    :param toolsets: toolsets.
    :param capabilities: capabilities added after the library's own.
    :returns: the agent.
    :rtype: Agent
    :raises ValueError: if ``NativeOutput`` is asked of a model that lacks it.
    :raises TypeError: if the model did not come from ``build_model``.
    """
    use_case = call_site(model)
    _check_output_mode(output_type, model)
    agent: Agent[Any, Any] = Agent(
        model,
        output_type=output_type,
        instructions=instructions,
        deps_type=deps_type,
        name=use_case.name,
        retries={"output": use_case.output_retries},
        tools=tools,
        toolsets=toolsets,
        capabilities=[
            RaiseContentFilterError(),
            *instrumentation(use_case),
            *(capabilities or ()),
        ],
    )
    # None would take Agent.instrument_all()'s setting when tracing is off. With
    # tracing on, the Instrumentation capability takes precedence over this.
    agent.instrument = False
    return agent


def shared_model(use_case: UseCase) -> Model:
    """Return the model ``complete`` runs a use case on.

    Built on the first call for a use case and reused on every thread and event
    loop, since ``PolicyModel`` keeps a provider model per loop. A model is rebuilt
    after ``configure`` replaces the settings. Under ``override_models`` every call
    builds a new one, so an override never outlives its block.

    :param use_case: the call site.
    :returns: the model.
    :rtype: Model
    :raises ValueError: if ``build_model`` rejects the use case.
    """
    if _model_override.get() is not None:
        return build_model(use_case)
    settings = get_settings()
    with _shared_models_lock:
        cached = _shared_models.get(use_case)
        if cached is None or cached[0] is not settings:
            cached = (settings, build_model(use_case, settings=settings))
            _shared_models[use_case] = cached
    return cached[1]


def usage_limits(use_case: UseCase) -> UsageLimits:
    """Return the usage limits every run of the use case gets (spec §6.8).

    :param use_case: the call site.
    :returns: the limits.
    :rtype: UsageLimits
    """
    return UsageLimits(
        request_limit=use_case.request_limit, cost_limit=use_case.cost_limit
    )


def attribution_model_settings(attribution: Attribution) -> ModelSettings:
    """Return the model settings that send the user id to the provider (spec §4.1).

    The keys are provider-prefixed, so each model in a fallback chain reads its own
    and ignores the rest. Providers with no such field get nothing.

    :param attribution: the per-call fields.
    :returns: the settings, empty when there is no user id.
    :rtype: ModelSettings
    """
    if attribution.user_id is None:
        return ModelSettings()
    return cast(
        "ModelSettings",
        {
            "openai_user": attribution.user_id,
            "anthropic_metadata": {"user_id": attribution.user_id},
        },
    )


@contextmanager
def override_models(models: Model | Mapping[str, Model]) -> Generator[None]:
    """Make ``build_model`` use the given models in place of real providers.

    The models are still wrapped in ``PolicyModel``, so timeouts, retries, error
    classification, and fallback behave as in production.

    :param models: one model for every id, or a mapping from model id to model.
    """
    token = _model_override.set(models)
    try:
        yield
    finally:
        _model_override.reset(token)
