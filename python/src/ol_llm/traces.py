"""Call spans for the entry points, and for embedding calls the library does not make.

``TracedCall`` opens the call span an entry point's call runs under, with the §4
attribution fields (spec §4) as its attributes, for runs and embeddings alike. The
span is OpenTelemetry's; each trace destination (``ol_llm.trace_destinations``)
takes part in the call while it runs and maps the span to its backend, e.g. Opik
makes it the call's own trace with its native environment.

``traced_embedding`` gives the same trace to an embedding request that a call site
built on LangChain or LangGraph makes with its own client (usage policies spec
§4.1, library spec §6.6). It runs the catalog checks ``embed`` runs and records a
span shaped like PydanticAI's embedding span, with the model and the input tokens
the caller reports, so every destination and the spend rules see the two paths
the same way. Retries, timeouts, and error classes stay with the caller's client.

Needs no extra.
"""

import inspect
import json
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

import pydantic_core
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.util.types import AttributeValue

from ol_llm import trace_attributes as attrs
from ol_llm.attribution import (
    NO_ATTRIBUTION,
    Attribution,
    DegradedReason,
    TraceFields,
    degraded_metadata,
    trace_fields,
)
from ol_llm.catalog import CachedCheck, embedding_entry, status_tags
from ol_llm.policy import UseCase, resolve_policy
from ol_llm.prompts import ManagedPrompt, attach_prompt
from ol_llm.settings import get_settings
from ol_llm.trace_destinations import CallTrace
from ol_llm.tracing import (
    active,
    destinations,
    enabled,
    include_content,
    tracer_provider,
)

_T = TypeVar("_T")

_current_call: ContextVar[CallTrace | None] = ContextVar(
    "ol_llm_current_call", default=None
)


def current_call() -> CallTrace | None:
    """Return the traced call the caller is inside, ``None`` outside one.

    ``None`` too inside a call that is not traced, even one inside a traced call.

    :returns: the call.
    :rtype: CallTrace | None
    """
    return _current_call.get()


def mark_degraded(reasons: Sequence[DegradedReason]) -> None:
    """Mark the current call's result degraded, on its span and for destinations.

    Does nothing outside a traced call or without a reason.

    :param reasons: why the result is degraded.
    """
    if not reasons or (call := _current_call.get()) is None:
        return
    call.degraded_reasons.extend(reasons)
    call.span.set_attributes(
        {
            attrs.attribution(key): value
            for key, value in degraded_metadata(call.degraded_reasons).items()
        }
    )


@dataclass
class TraceOutput:
    """What a call opened by ``TracedCall.trace_scope`` records as its output."""

    value: Any = None


@dataclass(frozen=True, kw_only=True)
class TracedCall:
    """The call span of one entry-point call, for runs and embeddings alike.

    :param use_case: the call site.
    :param attribution: the per-call fields.
    :param fields: what the call span is opened with.
    :param traced: whether tracing was on when the call started.
    :param prompts: the managed prompts the call uses, each linked to its trace.
        The first names the call in the trace metadata.
    """

    use_case: UseCase
    attribution: Attribution
    fields: TraceFields
    traced: bool
    prompts: tuple[ManagedPrompt[Any], ...] = ()

    @staticmethod
    def fields_for(
        use_case: UseCase,
        attribution: Attribution,
        prompt: ManagedPrompt[Any] | None = None,
        *,
        model_tags: Sequence[str] = (),
    ) -> TraceFields:
        """Return the call's attribution fields, with the policy it resolves to.

        :param use_case: the call site.
        :param attribution: the per-call fields.
        :param prompt: the managed prompt the call uses, if any.
        :param model_tags: the models' catalog-status tags, from ``status_tags``.
        :returns: the fields.
        :rtype: TraceFields
        """
        return trace_fields(
            use_case,
            environment=get_settings().environment,
            policy=resolve_policy(use_case),
            attribution=attribution,
            model_tags=model_tags,
            prompt_name=prompt.name if prompt else None,
            prompt_version=prompt.version if prompt else None,
        )

    def track_async(
        self, func: Callable[..., Awaitable[_T]]
    ) -> Callable[..., Awaitable[_T]]:
        """Wrap the coroutine function ``func`` in the call's span.

        The call's input is ``func``'s arguments and its output what it returns.
        """

        async def traced(*args: object) -> _T:
            with self.trace_scope(func, *args) as outcome:
                result = outcome.value = await func(*args)
            return result

        return traced

    @contextmanager
    def trace_scope(
        self, stand_in: Callable[..., object], *args: object
    ) -> Generator[TraceOutput]:
        """Hold the call's span current for the block, unless tracing is off.

        Each destination takes part in the call (``TraceDestination.call``) while
        the block runs, and the call's managed prompts are linked to it. The span
        records the block's exception, or else the output set on what this yields.
        Without content it records neither the input nor the output, and of an
        exception only its type.

        :param stand_in: a function whose parameters name the call's input.
        :param args: the input, in the order of ``stand_in``'s parameters.
        """
        outcome = TraceOutput()
        provider = tracer_provider(self.use_case.app) if self.traced else None
        if provider is None or not active():
            token = _current_call.set(None)
            try:
                yield outcome
            finally:
                _current_call.reset(token)
            return
        content = include_content()
        inputs = (
            dict(inspect.signature(stand_in).bind(*args).arguments) if content else None
        )
        with provider.get_tracer("ol_llm").start_as_current_span(
            self.use_case.name,
            attributes=_call_attributes(self.fields, inputs),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            call = CallTrace(
                name=self.use_case.name,
                fields=self.fields,
                span=span,
                content=content,
                input=inputs,
            )
            token = _current_call.set(call)
            try:
                with ExitStack() as stack:
                    for destination in destinations():
                        stack.enter_context(destination.call(call))
                    for prompt in self.prompts:
                        attach_prompt(prompt)
                    try:
                        yield outcome
                    except BaseException as exc:
                        # A cancelled block ends the call too, so the caller's
                        # context is left as it was.
                        call.error = exc
                        if isinstance(exc, Exception):
                            _record_error(span, exc, content=content)
                        raise
                    call.output = outcome.value
                    if content and outcome.value is not None:
                        span.set_attribute(attrs.OUTPUT, _json(outcome.value))
            finally:
                _current_call.reset(token)


def _call_attributes(
    fields: TraceFields, inputs: Mapping[str, Any] | None
) -> dict[str, AttributeValue]:
    """Return the call span's attributes: the §4 fields, the tags, the thread."""
    attributes: dict[str, AttributeValue] = {
        attrs.attribution(key): value for key, value in fields.metadata.items()
    }
    attributes[attrs.TAGS] = list(fields.tags)
    if fields.thread_id is not None:
        attributes[attrs.CONVERSATION_ID] = fields.thread_id
    if inputs is not None:
        attributes[attrs.INPUT] = _json(inputs)
    return attributes


def _json(value: object) -> str:
    """Return a value as JSON for a span attribute, ``str`` for what JSON lacks."""
    return pydantic_core.to_json(value, fallback=str).decode()


@CachedCheck
def _model_tags(use_case: UseCase) -> tuple[str, ...]:
    """Check a use case's embedding model once, as ``embed`` does per embedder.

    A rejected use case raises every time; an unlisted or deprecated one warns once,
    until its use case falls out of ``UseCaseCache``.
    """
    return status_tags([embedding_entry(use_case)])


def _embedding_input(texts: list[str]) -> None:
    """Stand in for a traced function, so the trace input matches ``embed``'s."""


@dataclass(frozen=True, kw_only=True)
class EmbeddingRequest:
    """What ``traced_embedding`` yields: where the caller reports the response.

    :param texts: the texts the request embeds.
    :param model: the catalog id the span names.
    :param span: the request's span, or ``None`` when tracing is off.
    :param output: what the call records as its output once ``record`` is
        called.
    """

    texts: Sequence[str]
    model: str
    span: Span | None
    output: TraceOutput

    def record(self, vectors: Sequence[Sequence[float]], *, input_tokens: int) -> None:
        """Record the response on the span and the trace. The vectors are not kept.

        :param vectors: the vectors the client returned.
        :param input_tokens: the input tokens the provider billed for this request.
            Report zero for a response served wholly from a cache: litellm fills
            in the usage of cached items, which were not billed. litellm also marks
            a partial hit as a cache hit and sums cached and billed usage, so for
            several texts count the misses' tokens or embed without its cache.
        """
        dimensions = len(vectors[0]) if vectors else 0
        if self.span is not None:
            usage = {attrs.USAGE_INPUT_TOKENS: input_tokens} if input_tokens else {}
            self.span.set_attributes(
                {
                    **usage,
                    attrs.RESPONSE_MODEL: self.model.partition(":")[2],
                    attrs.EMBEDDINGS_DIMENSION_COUNT: dimensions,
                }
            )
        self.output.value = {
            "model": self.model,
            "dimensions": dimensions,
            "embeddings": len(vectors),
            "input_tokens": input_tokens,
        }


@contextmanager
def traced_embedding(
    use_case: UseCase,
    texts: str | Sequence[str],
    *,
    attribution: Attribution = NO_ATTRIBUTION,
) -> Generator[EmbeddingRequest]:
    """Trace one embedding request the caller makes with its own client.

    Wrap each provider request, not a call that hedges or caches around several, so
    the trace counts every request that was billed. The client must call the use
    case's model: the span names it. The block's exception passes through
    unchanged; the spans record its message only when ``include_content`` is on,
    and its type either way. The texts are recorded only when ``include_content``
    is on, and the vectors never are.

    The call is a child of the current trace, or a new trace. A worker thread does
    not inherit the caller's context, so a request made on one is its own trace
    unless it is submitted with ``contextvars.copy_context().run``.

    For example, around litellm's embedding call::

        with ol_llm.traced_embedding(SEARCH_QUERY, [query]) as request:
            response = litellm.embedding(model=..., input=[query])
            cached = response._hidden_params.get("cache_hit") is True
            request.record(
                [item["embedding"] for item in response.data],
                input_tokens=0 if cached else response.usage.prompt_tokens,
            )

    :param use_case: the call site, naming an embedding model in ``provider:model``
        form. It may declare no ``fallback_models``.
    :param texts: one text or several, as sent to the provider.
    :param attribution: the per-call attribution fields.
    :returns: the request, on which the caller reports the response with
        ``record``.
    :rtype: Generator[EmbeddingRequest]
    :raises ValueError: if there are no texts, the catalog check rejects the use
        case, or the block ends without calling ``record``.
    """
    listed = [texts] if isinstance(texts, str) else list(texts)
    if not listed:
        msg = "traced_embedding needs at least one text"
        raise ValueError(msg)
    call = TracedCall(
        use_case=use_case,
        attribution=attribution,
        fields=TracedCall.fields_for(
            use_case, attribution, model_tags=_model_tags(use_case)
        ),
        traced=enabled(),
    )
    with call.trace_scope(_embedding_input, listed) as output:
        with _embedding_span(call, listed) as span:
            yield EmbeddingRequest(
                texts=listed, model=use_case.model, span=span, output=output
            )
        if output.value is None:
            msg = "traced_embedding's block ended without calling record"
            raise ValueError(msg)


@contextmanager
def _embedding_span(call: TracedCall, texts: list[str]) -> Generator[Span | None]:
    """Open a span with the attributes PydanticAI's embedding span carries.

    Yields ``None`` when the call is not traced.
    """
    provider = tracer_provider(call.use_case.app) if call.traced else None
    if provider is None:
        yield None
        return
    system, _, model_name = call.use_case.model.partition(":")
    attributes: dict[str, str | int] = {
        attrs.OPERATION_NAME: "embeddings",
        attrs.PROVIDER_NAME: system,
        attrs.REQUEST_MODEL: model_name,
        attrs.INPUTS_COUNT: len(texts),
    }
    content = include_content()
    if content:
        attributes[attrs.EMBEDDING_INPUTS] = json.dumps(texts)
    with provider.get_tracer("ol_llm").start_as_current_span(
        f"embeddings {model_name}",
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            _record_error(span, exc, content=content)
            raise


def _record_error(span: Span, error: Exception, *, content: bool) -> None:
    """Mark the span failed, as PydanticAI's ``record_uncaught_errors`` does.

    Without content only the exception's type is kept: a provider's error response
    can echo the request.
    """
    if not span.is_recording():
        return
    if content:
        span.record_exception(error, escaped=False)
        description: str | None = f"{type(error).__name__}: {error}"
    else:
        kind = type(error)
        name = (
            kind.__qualname__
            if kind.__module__ == "builtins"
            else f"{kind.__module__}.{kind.__qualname__}"
        )
        span.add_event(
            "exception",
            attributes={"exception.type": name, "exception.escaped": "False"},
        )
        description = None
    span.set_status(Status(StatusCode.ERROR, description=description))
