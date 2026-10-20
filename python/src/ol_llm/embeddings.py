"""The embedding entry points (usage policies spec §4.1): one Opik root trace per call.

``embed`` and ``embed_sync`` take a use case and texts, like ``complete``. They run
on a PydanticAI ``Embedder`` that ``build_embedder`` makes once per use case and
vector length: a catalog embedding model wrapped in ``PolicyEmbeddingModel``, with
the chat path's provider clients (SDK retries off, the policy timeout on the client),
retry layer, and error classes (spec §6.3). Each call opens the root trace with the
§4 attribution fields, and PydanticAI's embedding span, with the token usage, lands
inside it.

Hedging, caching, batching to provider limits, and the vector store stay in the
application. Store each vector with ``EmbedResult.model`` and
``EmbedResult.dimensions``: vectors from another model, or from the same model at
another size, are not comparable, so changing either means re-embedding.
"""

import json
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import cast

from pydantic_ai.embeddings import (
    Embedder,
    EmbeddingModel,
    EmbeddingResult,
    EmbeddingSettings,
    WrapperEmbeddingModel,
    infer_embedding_model,
)
from pydantic_ai.embeddings.result import EmbedInputType
from pydantic_ai.exceptions import ContentFilterError, UnexpectedModelBehavior
from pydantic_ai.usage import RequestUsage

from ol_llm.attribution import NO_ATTRIBUTION, Attribution
from ol_llm.catalog import CatalogEntry
from ol_llm.models import (
    LoopLocal,
    catalog_entries,
    classify_failure,
    classify_run_failure,
    instrumentation_settings,
    provider_factory,
    retrying,
    status_tags,
)
from ol_llm.policy import CallPolicy, UseCase, resolve_policy
from ol_llm.runs import TracedCall
from ol_llm.settings import Settings, get_settings
from ol_llm.tracing import enabled

# The providers PydanticAI has an embedding model for that the library builds.
EMBEDDING_PROVIDERS = frozenset(
    {"openai", "azure", "bedrock", "google", "google-cloud"}
)

# Tests may replace the provider model build_embedder creates; see ol_llm.testing.
_model_override: ContextVar[EmbeddingModel | None] = ContextVar(
    "ol_llm_embedding_model_override", default=None
)

# The embedders embed runs on, by use case and vector length, with the settings each
# was built with: each build creates provider clients, and calls should reuse them.
_shared_embedders: dict[tuple[UseCase, int | None], tuple[Settings, Embedder]] = {}

# What a provider embedding model raises on a response it cannot read. Bedrock's
# Cohere and Nova handlers raise UnexpectedModelBehavior, Titan's indexes the body
# for its vector (KeyError), and a body that is not JSON fails to decode.
_MALFORMED: tuple[type[Exception], ...] = (
    UnexpectedModelBehavior,
    KeyError,
    json.JSONDecodeError,
)

# Providers whose embeddings API takes the end user's id, as its ``user`` field.
_USER_FIELD_PROVIDERS = frozenset({"openai", "azure"})


@dataclass(init=False)
class PolicyEmbeddingModel(WrapperEmbeddingModel):
    """A provider embedding model under the call policy: one retry layer.

    Failures are classified as for chat (``ol_llm.models.classify_failure``);
    retryable ones are retried with exponential backoff and jitter, and what
    reaches the caller is an ``LLMError`` with the provider's exception as
    ``__cause__``. A call is retried whole: Bedrock's Titan models send one request
    per text, and one failed text sends them all again.

    A response the provider model cannot read, alone or in Titan's group of
    per-text failures, or one with a vector count that does not match the inputs,
    raises ``UnexpectedModelBehavior`` and is not retried. A content filter's
    refusal, alone or in a group, raises ``ContentFilterError`` instead.

    Calls go to a provider model built for the running event loop, as for
    ``PolicyModel`` (see ``LoopLocal``); ``wrapped`` answers for the model's name
    and never makes a call.
    """

    call_site: UseCase
    catalog_id: str
    catalog_entry: CatalogEntry | None
    policy: CallPolicy

    def __init__(  # noqa: PLR0913
        self,
        wrapped: EmbeddingModel,
        *,
        use_case: UseCase,
        model_id: str,
        catalog_entry: CatalogEntry | None,
        policy: CallPolicy,
        build: Callable[[], EmbeddingModel] | None = None,
    ) -> None:
        """Wrap a provider embedding model.

        :param wrapped: the provider model, with its SDK's own retries off.
        :param use_case: the call site.
        :param model_id: the id the model was built from, for errors.
        :param catalog_entry: the model's catalog entry, or ``None`` when the
            catalog does not list it.
        :param policy: the resolved policy.
        :param build: builds the same provider model again, once for each event
            loop calls run on; ``None`` sends every call to ``wrapped``.
        """
        super().__init__(wrapped)
        self.call_site = use_case
        self.catalog_id = model_id
        self.catalog_entry = catalog_entry
        self.policy = policy
        self._loop_models = None if build is None else LoopLocal(build)

    def _loop_model(self) -> EmbeddingModel:
        """Return the provider model for the running event loop."""
        if self._loop_models is None:
            return self.wrapped
        return self._loop_models.get()

    async def max_input_tokens(self) -> int | None:
        """Ask the running loop's provider model for its input limit."""
        return await self._loop_model().max_input_tokens()

    async def count_tokens(self, text: str) -> int:
        """Count tokens with the running loop's provider model."""
        return await self._loop_model().count_tokens(text)

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        """Embed the inputs, retrying transient failures under the policy."""
        expected = 1 if isinstance(inputs, str) else len(inputs)
        provider = self._loop_model()

        async def attempt() -> EmbeddingResult:
            try:
                result = await provider.embed(
                    inputs, input_type=input_type, settings=settings
                )
            except Exception as exc:
                if self._malformed(exc):
                    if _filtered(exc):
                        msg = f"{self.catalog_id} filtered a text"
                        raise ContentFilterError(msg) from exc
                    msg = f"{self.catalog_id} sent a response that could not be read"
                    raise UnexpectedModelBehavior(msg) from exc
                raise
            # Google's model drops vectors that come back empty.
            if len(result.embeddings) != expected:
                msg = (
                    f"{self.catalog_id} returned {len(result.embeddings)} vectors "
                    f"for {expected} texts"
                )
                raise UnexpectedModelBehavior(msg)
            return result

        return await retrying(
            attempt,
            policy=self.policy,
            profile=self.call_site.profile,
            use_case=self.call_site.name,
            model=self.catalog_id,
        )

    def _malformed(self, exc: BaseException) -> bool:
        """Whether ``exc`` means a response that could not be read.

        A group is when any member is and every other member is a classified
        provider failure, so a bug raised beside a malformed response still
        surfaces as itself.
        """
        if not isinstance(exc, BaseExceptionGroup):
            return isinstance(exc, _MALFORMED)
        members = exc.exceptions
        return any(self._malformed(member) for member in members) and all(
            self._malformed(member)
            or classify_failure(
                member, use_case=self.call_site.name, model=self.catalog_id
            )
            is not None
            for member in members
        )


def _filtered(exc: BaseException) -> bool:
    """Whether ``exc`` is a content filter's refusal, or a group holding one."""
    if isinstance(exc, BaseExceptionGroup):
        return any(_filtered(member) for member in exc.exceptions)
    return isinstance(exc, ContentFilterError)


def build_embedder(
    use_case: UseCase,
    *,
    dimensions: int | None = None,
    settings: Settings | None = None,
) -> Embedder:
    """Build a PydanticAI embedder for a use case's model, under policy.

    Each call creates new provider clients, so ``embed`` goes through
    ``shared_embedder``, which builds one per use case and vector length. The
    embedder sends its spans to Opik unless tracing is off.

    :param use_case: the call site. Its model may not be retired or listed as a
        chat model, and it may declare no ``fallback_models``.
    :param dimensions: the vector length to ask for, for a model that can shorten
        its vectors. Omitted, the model returns its default length, which for a
        listed model is the catalog entry's ``dimensions``.
    :param settings: overrides the configured settings, mainly for tests.
    :returns: the embedder.
    :rtype: Embedder
    :raises ValueError: if the use case declares fallbacks, its model is retired
        or listed as a chat model, the library does not build
        embeddings for its provider, or an environment policy value does not parse.
    """
    if use_case.fallback_models:
        msg = (
            f"Use case {use_case.name!r} declares fallback_models; an embedding use "
            "case takes none, since another model's vectors are not comparable"
        )
        raise ValueError(msg)
    [entry] = catalog_entries(use_case, kind="embedding")
    if (provider := use_case.model.partition(":")[0]) not in EMBEDDING_PROVIDERS:
        msg = (
            f"ol-llm does not build {provider!r} embedding models; "
            f"supported providers are {sorted(EMBEDDING_PROVIDERS)}"
        )
        raise ValueError(msg)
    policy = resolve_policy(use_case)
    # An override's model is the test's own and is not rebuilt per loop.
    build: Callable[[], EmbeddingModel] | None = None
    if (wrapped := _model_override.get()) is None:
        factory = provider_factory(use_case, policy, settings or get_settings())
        build = partial(infer_embedding_model, use_case.model, provider_factory=factory)
        wrapped = build()
    model = PolicyEmbeddingModel(
        wrapped,
        use_case=use_case,
        model_id=use_case.model,
        catalog_entry=entry,
        policy=policy,
        build=build,
    )
    shorter = None if dimensions is None else EmbeddingSettings(dimensions=dimensions)
    return Embedder(
        model, settings=shorter, instrument=instrumentation_settings(use_case)
    )


@dataclass(frozen=True, kw_only=True)
class EmbedResult:
    """The result of an embedding call.

    :param embeddings: one vector per input text, in input order.
    :param model: the catalog id of the model that produced them.
    :param dimensions: the length of each vector.
    :param usage: the input tokens the provider reported. The Gemini API reports
        none, so its usage is zero.
    """

    embeddings: Sequence[Sequence[float]]
    model: str
    dimensions: int
    usage: RequestUsage


def _trace_output(result: EmbedResult) -> Mapping[str, object]:
    """Return what the root trace records as output: everything but the vectors."""
    return {
        "model": result.model,
        "dimensions": result.dimensions,
        "embeddings": len(result.embeddings),
        "input_tokens": result.usage.input_tokens,
    }


def shared_embedder(use_case: UseCase, dimensions: int | None = None) -> Embedder:
    """Return the embedder ``embed`` runs a use case on.

    Built on the first call for a use case and vector length and reused on every
    thread and event loop, since ``PolicyEmbeddingModel`` keeps a provider model
    per loop. An embedder is rebuilt after ``configure`` replaces the settings. Under
    ``override_embedding_models`` every call builds a new one, so an override never
    outlives its block.

    :param use_case: the call site.
    :param dimensions: the vector length to ask for, as for ``build_embedder``.
    :returns: the embedder.
    :rtype: Embedder
    :raises ValueError: if ``build_embedder`` rejects the use case.
    """
    if _model_override.get() is not None:
        return build_embedder(use_case, dimensions=dimensions)
    settings = get_settings()
    cached = _shared_embedders.get((use_case, dimensions))
    if cached is None or cached[0] is not settings:
        embedder = build_embedder(use_case, dimensions=dimensions, settings=settings)
        cached = _shared_embedders[use_case, dimensions] = (settings, embedder)
    return cached[1]


def _start(
    use_case: UseCase,
    texts: str | Sequence[str],
    dimensions: int | None,
    attribution: Attribution,
) -> tuple[Embedder, TracedCall, list[str]]:
    listed = [texts] if isinstance(texts, str) else list(texts)
    if not listed:
        msg = "embed needs at least one text"
        raise ValueError(msg)
    embedder = shared_embedder(use_case, dimensions)
    model = cast("PolicyEmbeddingModel", embedder.model)
    call = TracedCall(
        use_case=use_case,
        attribution=attribution,
        fields=TracedCall.fields_for(
            use_case, attribution, model_tags=status_tags([model.catalog_entry])
        ),
        traced=enabled(),
    )
    return embedder, call, listed


def _finish(call: TracedCall, result: EmbeddingResult) -> EmbedResult:
    return EmbedResult(
        embeddings=result.embeddings,
        model=call.use_case.model,
        dimensions=len(result.embeddings[0]),
        usage=result.usage,
    )


def _user_settings(call: TracedCall) -> EmbeddingSettings | None:
    """Return the settings that send the user id to the provider (spec §4.1).

    Only the OpenAI and Azure embeddings APIs have a field for it, ``user``, which
    PydanticAI's OpenAI embedding model passes through ``extra_body``.
    """
    user_id = call.attribution.user_id
    provider = call.use_case.model.partition(":")[0]
    if user_id is None or provider not in _USER_FIELD_PROVIDERS:
        return None
    return EmbeddingSettings(extra_body={"user": user_id})


def _replacement(call: TracedCall, exc: Exception) -> Exception:
    """Return what to raise in place of ``exc``: its library error, or itself.

    A malformed response arrives from ``PolicyEmbeddingModel`` as
    ``UnexpectedModelBehavior`` and becomes ``LLMInvalidOutput``.
    """
    use_case = call.use_case
    error = classify_run_failure(exc, use_case=use_case.name, model=use_case.model)
    return exc if error is None else error


async def embed(
    use_case: UseCase,
    texts: str | Sequence[str],
    *,
    input_type: EmbedInputType,
    dimensions: int | None = None,
    attribution: Attribution = NO_ATTRIBUTION,
) -> EmbedResult:
    """Embed texts under a traced root.

    The use case's embedder is built on the first call and reused after that.

    :param use_case: the call site. Its model may not be retired or listed as a
        chat model, and it may declare no ``fallback_models``.
    :param texts: one text or several. Batching to the provider's limits is the
        caller's job.
    :param input_type: ``query`` for a search query, ``document`` for text to
        store. Required because Google and Cohere models embed the two
        differently, so a default would be silently wrong for one of them.
    :param dimensions: the vector length to ask for, for a model that can shorten
        its vectors. Omitted, the model returns its default length, which for a
        listed model is the catalog entry's ``dimensions``.
    :param attribution: the per-call attribution fields. The user id also reaches
        OpenAI and Azure as the request's ``user``; the other providers have no
        such field.
    :returns: the result.
    :rtype: EmbedResult
    :raises LLMError: on a model failure, or ``LLMInvalidOutput`` on a response
        that cannot be read or has a vector count that does not match the texts.
    :raises ValueError: if there are no texts, or ``build_embedder`` rejects the
        use case.
    """
    embedder, call, listed = _start(use_case, texts, dimensions, attribution)
    results: list[EmbedResult] = []

    async def traced(texts: list[str]) -> Mapping[str, object]:
        call.mark_trace()
        try:
            result = await embedder.embed(
                texts, input_type=input_type, settings=_user_settings(call)
            )
        except Exception as exc:
            if (error := _replacement(call, exc)) is exc:
                raise
            raise error from exc
        results.append(_finish(call, result))
        return _trace_output(results[0])

    await call.track(traced)(listed)
    return results[0]


def embed_sync(
    use_case: UseCase,
    texts: str | Sequence[str],
    *,
    input_type: EmbedInputType,
    dimensions: int | None = None,
    attribution: Attribution = NO_ATTRIBUTION,
) -> EmbedResult:
    """Embed texts synchronously, for Celery tasks and Dagster assets.

    Takes the same arguments as ``embed`` and behaves the same way. It cannot be
    called while an event loop is running in the thread; use ``embed`` there.

    :returns: the result.
    :rtype: EmbedResult
    """
    embedder, call, listed = _start(use_case, texts, dimensions, attribution)
    results: list[EmbedResult] = []

    def traced(texts: list[str]) -> Mapping[str, object]:
        call.mark_trace()
        try:
            result = embedder.embed_sync(
                texts, input_type=input_type, settings=_user_settings(call)
            )
        except Exception as exc:
            if (error := _replacement(call, exc)) is exc:
                raise
            raise error from exc
        results.append(_finish(call, result))
        return _trace_output(results[0])

    call.track(traced)(listed)
    return results[0]


@contextmanager
def override_embedding_models(model: EmbeddingModel) -> Generator[None]:
    """Make ``embed`` and ``embed_sync`` use the given model in place of a provider.

    The model is still wrapped in ``PolicyEmbeddingModel``, so retries and error
    classification behave as in production. Inside the block every call builds a
    new embedder rather than reusing a cached one.

    :param model: the model for every use case, e.g. PydanticAI's
        ``TestEmbeddingModel``.
    """
    token = _model_override.set(model)
    try:
        yield
    finally:
        _model_override.reset(token)
