"""Root traces for the entry points, and for embedding calls the library does not make.

``TracedCall`` opens the Opik root trace an entry point's call runs under, with the
§4 attribution fields (spec §4), for runs and embeddings alike.

``traced_embedding`` gives the same trace to an embedding request that a call site
built on LangChain or LangGraph makes with its own client (usage policies spec
§4.1, library spec §6.6). It runs the catalog checks ``embed`` runs and records a
span shaped like PydanticAI's embedding span, with the model and the input tokens
the caller reports, so Opik and the spend rules see the two paths the same way.
Retries, timeouts, and error classes stay with the caller's client.

Needs no extra.
"""

import json
import traceback
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from typing import Any, TypeVar

import opik
from opentelemetry.trace import Span, Status, StatusCode
from opik import opik_context
from opik.api_objects.trace import TraceData
from opik.decorator import error_info_collector
from opik.decorator.arguments_helpers import TrackOptions
from opik.decorator.tracker import OpikTrackDecorator
from opik.types import ErrorInfoDict

from ol_llm.attribution import NO_ATTRIBUTION, Attribution, TraceFields, trace_fields
from ol_llm.catalog import embedding_entry, status_tags
from ol_llm.policy import UseCase, resolve_policy
from ol_llm.prompts import ManagedPrompt, attach_prompt
from ol_llm.settings import get_settings
from ol_llm.tracing import enabled, include_content, tracer_provider

_T = TypeVar("_T")

# Drives a root trace held open across a block with the halves of opik.track's own
# wrapper.
_TRACKER = OpikTrackDecorator()


@dataclass
class TraceOutput:
    """What a trace opened by ``TracedCall.trace_scope`` records as its output."""

    value: Any = None


@dataclass(frozen=True, kw_only=True)
class TracedCall:
    """The root trace of one entry-point call, for runs and embeddings alike.

    :param use_case: the call site.
    :param attribution: the per-call fields.
    :param fields: what the root trace is opened with.
    :param traced: whether tracing was on when the call started.
    :param prompt: the managed prompt the call uses, linked to its trace.
    """

    use_case: UseCase
    attribution: Attribution
    fields: TraceFields
    traced: bool
    prompt: ManagedPrompt[Any] | None = None

    @staticmethod
    def fields_for(
        use_case: UseCase,
        attribution: Attribution,
        prompt: ManagedPrompt[Any] | None = None,
        *,
        model_tags: Sequence[str] = (),
    ) -> TraceFields:
        """Return the root-trace fields for a call, with the policy it resolves to.

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

    def track(self, func: Callable[..., _T]) -> Callable[..., _T]:
        """Wrap ``func`` in the call's root trace, as ``trace_scope`` opens it.

        The trace's input is ``func``'s arguments and its output what it returns.
        """

        def traced(*args: object) -> _T:
            with self.trace_scope(func, *args) as outcome:
                result = outcome.value = func(*args)
            return result

        return traced

    def track_async(
        self, func: Callable[..., Awaitable[_T]]
    ) -> Callable[..., Awaitable[_T]]:
        """Wrap the coroutine function ``func`` in the call's root trace."""

        async def traced(*args: object) -> _T:
            with self.trace_scope(func, *args) as outcome:
                result = outcome.value = await func(*args)
            return result

        return traced

    def _current_trace(self) -> TraceData | None:
        """Return the trace this run marks, or ``None`` when it is not traced.

        Read from the context each time rather than trusted from the start of the
        run: tracing can be switched on at runtime after the run started without
        a trace, and marking a trace that is not there raises.
        """
        return opik_context.get_current_trace_data() if self.traced else None

    def mark_trace(self) -> None:
        """Put the use case's tags, the thread, and the prompt link on the trace.

        For a call the library opened the tags and prompt are already there; for
        one inside a caller's trace, whose own metadata lands on its span, this adds
        them there. A caller's trace that already has a thread keeps it.
        """
        if (trace := self._current_trace()) is not None:
            opik_context.update_current_trace(
                tags=list(self.fields.tags),
                metadata={
                    key: self.fields.metadata[key]
                    for key in ("prompt_name", "prompt_version")
                    if key in self.fields.metadata
                },
                thread_id=None if trace.thread_id else self.attribution.thread_id,
            )
            if self.prompt is not None:
                attach_prompt(self.prompt)

    @contextmanager
    def trace_scope(
        self, stand_in: Callable[..., object], *args: object
    ) -> Generator[TraceOutput]:
        """Hold the call's root trace current for the block, unless tracing is off.

        Every entry point's root trace is opened here, rather than by ``opik.track``,
        whose wrapper records the exception's message and source lines whatever
        ``include_content`` says. The trace records the block's exception, or else
        the output set on what this yields. Without content the exception's type and
        frames are recorded, not its message.

        :param stand_in: a function whose parameters name the trace's input.
        :param args: the input, in the order of ``stand_in``'s parameters.
        """
        outcome = TraceOutput()
        if not (self.traced and opik.is_tracing_active()):
            yield outcome
            return
        content = include_content()
        options = TrackOptions(
            name=self.use_case.name,
            type="general",
            tags=list(self.fields.tags),
            metadata=dict(self.fields.metadata),
            capture_input=content,
            ignore_arguments=None,
            capture_output=content,
            generations_aggregator=None,
            flush=False,
            project_name=self.fields.project_name,
            create_duplicate_root_span=True,
            source=None,
            environment=self.fields.environment,
        )
        # The two halves opik.track's wrapper runs around a call; the exact opik
        # pin keeps these private methods stable.
        span_started = _TRACKER._before_call(  # noqa: SLF001
            func=stand_in, track_options=options, args=args, kwargs={}
        )
        try:
            yield outcome
        except BaseException as exc:
            # A cancelled block closes the trace too, so the caller's context is
            # left as it was.
            _TRACKER._after_call(  # noqa: SLF001
                output=None,
                error_info=_error_info(exc, content=content)
                if isinstance(exc, Exception)
                else None,
                capture_output=content,
                should_process_span_data=span_started,
            )
            raise
        _TRACKER._after_call(  # noqa: SLF001
            output=outcome.value,
            error_info=None,
            capture_output=content,
            should_process_span_data=span_started,
        )


@cache
def _model_tags(use_case: UseCase) -> tuple[str, ...]:
    """Check a use case's embedding model once, as ``embed`` does per embedder.

    A rejected use case raises every time; an unlisted or deprecated one warns once.
    """
    return status_tags([embedding_entry(use_case)])


def _error_info(error: Exception, *, content: bool) -> ErrorInfoDict:
    """Return what the root trace records about an error.

    Without content the message is left out, and the traceback keeps only where
    each frame is: the formatted one ends with the message, repeats those of
    chained exceptions, and quotes source lines.
    """
    if content:
        return error_info_collector.collect(error)
    frames = "".join(
        f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n'
        for frame in traceback.extract_tb(error.__traceback__)
    )
    return {"exception_type": type(error).__name__, "traceback": frames}


def _embedding_input(texts: list[str]) -> None:
    """Stand in for a traced function, so the trace input matches ``embed``'s."""


@dataclass(frozen=True, kw_only=True)
class EmbeddingRequest:
    """What ``traced_embedding`` yields: where the caller reports the response.

    :param texts: the texts the request embeds.
    :param model: the catalog id the span names.
    :param span: the request's span, or ``None`` when tracing is off.
    :param output: what the root trace records as its output once ``record`` is
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
            usage = {"gen_ai.usage.input_tokens": input_tokens} if input_tokens else {}
            self.span.set_attributes(
                {
                    **usage,
                    "gen_ai.response.model": self.model.partition(":")[2],
                    "gen_ai.embeddings.dimension.count": dimensions,
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
    unchanged; the span and the root trace record its message only when
    ``include_content`` is on, and its type either way. The texts are recorded
    only when ``include_content`` is on, and the vectors never are.

    The trace is the current Opik trace's child, or a new root trace. A worker
    thread does not inherit the caller's context, so a request made on one is its
    own root trace unless it is submitted with ``contextvars.copy_context().run``.

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
        call.mark_trace()
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
        "gen_ai.operation.name": "embeddings",
        "gen_ai.provider.name": system,
        "gen_ai.request.model": model_name,
        "inputs_count": len(texts),
    }
    content = include_content()
    if content:
        attributes["inputs"] = json.dumps(texts)
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
