"""Model construction: the only supported way to get a PydanticAI model (spec §6.3).

``build_model`` turns every provider SDK's own retries off, wraps each model in
``PolicyModel`` so there is exactly one retry layer and every request has a
timeout, and composes a declared fallback chain. Needs the ``pydantic-ai`` extra and
the extra for each provider a use case names.
"""

import asyncio
import logging
import os
import random
import sys
from collections.abc import (
    AsyncGenerator,
    Callable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, NoReturn, TypeVar, cast, overload

import anyio
import httpx
import httpx2
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import RaiseContentFilterError
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
    ModelRequestParameters,
    StreamedResponse,
    infer_model,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import NativeOutput, OutputSpec
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.usage import UsageLimits

from ol_llm.attribution import USE_CASE_HEADER, Attribution
from ol_llm.catalog import CatalogEntry, load_catalog
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

log = logging.getLogger(__name__)

OutputT = TypeVar("OutputT")
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

# Patched in tests so retries do not wait.
_sleep: Callable[[float], Any] = asyncio.sleep


def classify_failure(
    exc: BaseException, *, use_case: str, model: str
) -> LLMError | None:
    """Map a failed model request to the library error for its class.

    The table is closed: every ``ModelAPIError`` and every ``httpx`` or ``httpx2``
    transport error maps to a class, so nothing a provider raises is unclassified.

    :param exc: what the request raised.
    :param use_case: the use case name, for the error.
    :param model: the model id, for the error.
    :returns: the library error, or ``None`` when ``exc`` is not a model request
        failure and should propagate unchanged.
    :rtype: LLMError | None
    """
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
    if isinstance(exc, (ModelAPIError, *_UNAVAILABLE_TRANSPORT_ERRORS)):
        return LLMUnavailable(message, use_case=use_case, model=model)
    if isinstance(exc, _TRANSPORT_ERRORS):
        return LLMRejected(message, use_case=use_case, model=model, status_code=None)
    return None


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


_METADATA_KEY = "ol_llm"


def _stamp_answered_by(
    response: ModelResponse | StreamedResponse, catalog_id: str
) -> None:
    """Record which chain member answered, where the run's history keeps it."""
    if response.metadata is None:
        response.metadata = {}
    response.metadata.setdefault(_METADATA_KEY, {})["model"] = catalog_id


def _answered_by(response: ModelResponse) -> str | None:
    answered_by: str | None = (
        (response.metadata or {}).get(_METADATA_KEY, {}).get("model")
    )
    return answered_by


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
    """

    use_case: str
    catalog_id: str
    catalog_entry: CatalogEntry
    profile_name: Profile
    policy: CallPolicy

    def __init__(
        self,
        wrapped: Model,
        *,
        use_case: UseCase,
        model_id: str,
        catalog_entry: CatalogEntry,
        policy: CallPolicy,
    ) -> None:
        """Wrap a provider model.

        :param wrapped: the provider model, with its SDK's own retries off.
        :param use_case: the call site.
        :param model_id: the id the model was built from, for errors.
        :param catalog_entry: the model's catalog entry; tracing reads its status
            and Opik name from here.
        :param policy: the resolved policy.
        """
        super().__init__(wrapped)
        self.use_case = use_case.name
        self.catalog_id = model_id
        self.catalog_entry = catalog_entry
        self.profile_name = use_case.profile
        self.policy = policy

    def _with_timeout(self, model_settings: ModelSettings | None) -> ModelSettings:
        timeout = ModelSettings(timeout=self.policy.timeout_seconds)
        return merge_model_settings(model_settings, timeout) or timeout

    def _classify(self, exc: BaseException) -> LLMError | None:
        return classify_failure(exc, use_case=self.use_case, model=self.catalog_id)

    def _retry_delay(self, exc: Exception, budget: _RetryBudget) -> float:
        """Return the wait before the next attempt, or raise what the caller sees.

        Callers catch ``Exception`` and hand it here; anything not classified is
        re-raised unchanged.
        """
        if (error := self._classify(exc)) is None:
            raise exc
        if (delay := budget.next_delay(error)) is None:
            raise error from exc
        return delay

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make the request, retrying transient failures under the policy."""
        settings = self._with_timeout(model_settings)
        budget = _RetryBudget(self.policy, self.profile_name)
        while True:
            try:
                response = await self.wrapped.request(
                    messages, settings, model_request_parameters
                )
            except Exception as exc:  # noqa: BLE001
                delay = self._retry_delay(exc, budget)
            else:
                _stamp_answered_by(response, self.catalog_id)
                return response
            await _sleep(delay)

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
        while True:
            stream_cm = self.wrapped.request_stream(
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
    """

    def _starting_at(self, messages: list[ModelMessage]) -> FallbackModel:
        """Return the chain a request runs on, read from the run's own history."""
        last = messages[-1] if messages else None
        if not isinstance(last, ModelRequest) or not any(
            isinstance(part, RetryPromptPart) for part in last.parts
        ):
            return self
        answered_by = next(
            (
                _answered_by(message)
                for message in reversed(messages)
                if isinstance(message, ModelResponse)
            ),
            None,
        )
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


def _openai_provider(route: GatewayRoute | None, use_case: UseCase) -> Provider[Any]:
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415
        from pydantic_ai.providers.openai import OpenAIProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("openai", exc)
    client = AsyncOpenAI(
        max_retries=0,
        base_url=route.base_url if route else None,
        api_key=route.api_key if route else None,
        default_headers=_gateway_headers(route, use_case),
    )
    return OpenAIProvider(openai_client=client)


def _azure_provider(settings: Settings) -> Provider[Any]:
    try:
        from openai import AsyncAzureOpenAI  # noqa: PLC0415
        from pydantic_ai.providers.azure import AzureProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("openai", exc)
    # Endpoint, API version, and key come from AZURE_OPENAI_ENDPOINT,
    # OPENAI_API_VERSION, and AZURE_OPENAI_API_KEY unless a token provider is set.
    client = AsyncAzureOpenAI(
        max_retries=0, azure_ad_token_provider=settings.azure_ad_token_provider
    )
    return AzureProvider(openai_client=client)


def _anthropic_provider(route: GatewayRoute | None, use_case: UseCase) -> Provider[Any]:
    try:
        from anthropic import AsyncAnthropic  # noqa: PLC0415
        from pydantic_ai.providers.anthropic import AnthropicProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("anthropic", exc)
    client = AsyncAnthropic(
        max_retries=0,
        base_url=route.base_url if route else None,
        api_key=route.api_key if route else None,
        default_headers=_gateway_headers(route, use_case),
    )
    return AnthropicProvider(anthropic_client=client)


def _bedrock_provider(policy: CallPolicy) -> Provider[Any]:
    try:
        import boto3  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415
        from pydantic_ai.providers.bedrock import BedrockProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("bedrock", exc)
    # Bedrock ignores ModelSettings.timeout, so the timeout lives on the client,
    # and the client is the only place botocore's own retries can be turned off.
    config = Config(
        read_timeout=policy.timeout_seconds,
        connect_timeout=policy.timeout_seconds,
        retries={"total_max_attempts": 1},
    )
    return BedrockProvider(
        bedrock_client=boto3.client("bedrock-runtime", config=config)
    )


def _google_provider(provider: str) -> Provider[Any]:
    try:
        from pydantic_ai.providers.google import GoogleProvider  # noqa: PLC0415
        from pydantic_ai.providers.google_cloud import (  # noqa: PLC0415
            GoogleCloudProvider,
        )
    except ImportError as exc:
        _raise_missing_extra("google", exc)
    # The SDK retries only when given a retry config, and none is passed.
    if provider == "google-cloud":
        return GoogleCloudProvider()
    return GoogleProvider()


def _mistral_provider() -> Provider[Any]:
    try:
        from pydantic_ai.providers.mistral import MistralProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("mistral", exc)
    # The SDK retries only when given a retry config, and none is passed.
    return MistralProvider()


def _together_provider() -> Provider[Any]:
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415
        from pydantic_ai.providers.together import TogetherProvider  # noqa: PLC0415
    except ImportError as exc:
        _raise_missing_extra("openai", exc)
    # AsyncOpenAI reads OPENAI_API_KEY by default; Together's key has its own name.
    client = AsyncOpenAI(
        max_retries=0,
        base_url=_TOGETHER_BASE_URL,
        api_key=os.environ["TOGETHER_API_KEY"],
    )
    return TogetherProvider(openai_client=client)


def _provider_factory(
    use_case: UseCase, policy: CallPolicy, settings: Settings
) -> Callable[[str], Provider[Any]]:
    def factory(provider: str) -> Provider[Any]:  # noqa: PLR0911
        if provider in _OPENAI_PROVIDERS:
            return _openai_provider(settings.gateway_routes.get("openai"), use_case)
        if provider in _AZURE_PROVIDERS:
            return _azure_provider(settings)
        if provider == "anthropic":
            return _anthropic_provider(
                settings.gateway_routes.get("anthropic"), use_case
            )
        if provider == "bedrock":
            return _bedrock_provider(policy)
        if provider in {"google", "google-cloud"}:
            return _google_provider(provider)
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
    provider, sep, name = model_id.partition(":")
    if not (provider and sep and name):
        msg = f"Model id {model_id!r} is not in provider:model form"
        raise ValueError(msg)
    if override is not None:
        return override if isinstance(override, Model) else override[model_id]
    if factory is None:  # pragma: no cover - set whenever override is None
        raise AssertionError
    return infer_model(model_id, provider_factory=factory)


def _catalog_entries(use_case: UseCase) -> list[CatalogEntry]:
    """Look up every model the use case names; only catalog chat models build."""
    catalog = load_catalog()
    entries = [
        catalog.require(model_id, kind="chat") for model_id in use_case.model_ids
    ]
    if use_case.fallback_models and entries[0].route_falls_back:
        msg = (
            f"Use case {use_case.name!r} declares fallback_models, but the route to "
            f"{use_case.model!r} already falls back on its own (spec §6.4)"
        )
        raise ValueError(msg)
    for model_id, entry in zip(use_case.model_ids, entries, strict=True):
        if entry.status == "deprecated":
            log.warning(
                "Use case %s names %s, which is deprecated and retires on %s",
                use_case.name,
                model_id,
                entry.retires_on,
            )
    return entries


def build_model(use_case: UseCase, *, settings: Settings | None = None) -> Model:
    """Build the model for a use case: the primary and any fallbacks, under policy.

    Build once per use case and reuse it: each call creates new provider clients.

    :param use_case: the call site.
    :param settings: overrides the configured settings, mainly for tests.
    :returns: a ``PolicyModel``, or a ``PolicyFallbackModel`` over one per model id
        when the use case declares fallbacks.
    :rtype: Model
    :raises ValueError: on a model id that is not a chat model in the catalog or
        is retired, fallbacks behind a route that falls back on its own, a
        provider the library does not build, or an environment policy value that
        does not parse.
    """
    entries = _catalog_entries(use_case)
    policy = resolve_policy(use_case)
    override = _model_override.get()
    factory = (
        None
        if override is not None
        else _provider_factory(use_case, policy, settings or get_settings())
    )
    models = [
        PolicyModel(
            _provider_model(model_id, override, factory),
            use_case=use_case,
            model_id=model_id,
            catalog_entry=entry,
            policy=policy,
        )
        for model_id, entry in zip(use_case.model_ids, entries, strict=True)
    ]
    if len(models) == 1:
        return models[0]
    return PolicyFallbackModel(*models, fallback_on=FALLBACK_ON)


def _chain(model: Model) -> Sequence[Model]:
    return model.models if isinstance(model, FallbackModel) else (model,)


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
    partial text, and retries invalid output on the same model
    ``use_case.output_retries`` times.

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
    model = build_model(use_case, settings=settings)
    _check_output_mode(output_type, model)
    return Agent(
        model,
        output_type=output_type,
        instructions=instructions,
        deps_type=deps_type,
        name=use_case.name,
        retries={"output": use_case.output_retries},
        tools=tools,
        toolsets=toolsets,
        capabilities=[RaiseContentFilterError(), *(capabilities or ())],
    )


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
