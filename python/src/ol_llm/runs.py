"""The entry points (spec §6.6): one Opik root trace per run.

``run`` and ``run_sync`` open the root trace with the §4 attribution fields, run the
agent under the use case's usage limits, and return an ``LLMResult``. The agent's
model spans land inside that trace through ``OpikSpanProcessor`` (see
``ol_llm.tracing``). Model failures reach the caller as ``LLMError`` subclasses,
and the trace records the same error.

A run inside a caller's own ``@opik.track`` becomes a span of the caller's trace,
in the caller's project: Opik files a span under its trace's project. The
attribution metadata goes on the run's span; the tags, ``thread_id``, and the
degraded marking go on the caller's trace, so filters and scoring rules scoped by
tag still find it. With tracing off, runs open no trace.

``run_stream`` holds the same root trace open while the caller consumes the stream
inside its own ``async with``. ``opik.track`` cannot do that: on an async generator
it makes the trace current only during each ``__anext__``. So the entry and exit
halves of the SDK's decorator run around the caller's block instead, which reaches
into opik internals that the exact opik pin and the smoke test keep honest.

``complete`` and ``complete_sync`` are the single-call path: a tool-less agent over
the use case's model, run by ``run``, so a call site that outgrows them builds the
same agent with ``build_agent`` and keeps everything else.
"""

from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Generator,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast, overload

import opik
from opik import opik_context
from opik.api_objects.trace import TraceData
from opik.decorator import error_info_collector
from opik.decorator.arguments_helpers import TrackOptions
from opik.decorator.tracker import OpikTrackDecorator
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

from ol_llm.attribution import (
    NO_ATTRIBUTION,
    Attribution,
    DegradedReason,
    TraceFields,
    degraded_metadata,
    trace_fields,
)
from ol_llm.errors import LLMError
from ol_llm.models import (
    agent_over,
    answering_model,
    attribution_model_settings,
    call_site,
    catalog_tags,
    classify_failure,
    classify_run_failure,
    recording_answers,
    shared_model,
    usage_limits,
)
from ol_llm.policy import UseCase, resolve_policy
from ol_llm.settings import get_settings
from ol_llm.tracing import enabled, include_content

OutputT = TypeVar("OutputT")
DepsT = TypeVar("DepsT")
_T = TypeVar("_T")

# Drives run_stream's root trace with the halves of opik.track's own wrapper.
_TRACKER = OpikTrackDecorator()


def _root_input(user_prompt: str | None) -> None:
    """Stand in for a traced function, so a stream's trace input matches ``run``'s."""


@dataclass
class _TraceOutput:
    """What a trace opened by ``_Run.trace_scope`` records as its output."""

    value: Any = None


@dataclass(frozen=True, kw_only=True)
class PostProcessed(Generic[OutputT]):
    """What a ``postprocess`` callable returns.

    :param output: the output to return in place of the model's.
    :param degraded_reasons: why the output is degraded, e.g. ``partial_output``.
    """

    output: OutputT
    degraded_reasons: tuple[DegradedReason, ...] = ()


@dataclass(frozen=True, kw_only=True)
class LLMResult(Generic[OutputT]):
    """The result of a run.

    :param output: the validated output, after ``postprocess``.
    :param model: the model id that gave the final response.
    :param usage: token and request usage for the run.
    :param degraded: whether any degraded reason applies.
    :param degraded_reasons: why the result is degraded.
    :param prompt_name: the managed prompt used, when there is one.
    :param prompt_version: that prompt's version.
    :param new_messages: the messages this run added, for the next turn's history.
    """

    output: OutputT
    model: str
    usage: RunUsage
    degraded: bool
    degraded_reasons: tuple[DegradedReason, ...]
    prompt_name: str | None
    prompt_version: str | None
    new_messages: Sequence[ModelMessage]


@dataclass(frozen=True)
class _Run(Generic[OutputT]):
    """The parts of a run shared by the async and sync entry points."""

    use_case: UseCase
    attribution: Attribution
    postprocess: Callable[[OutputT], PostProcessed[OutputT]] | None
    fields: TraceFields
    traced: bool

    def track(self, func: Callable[..., _T]) -> Callable[..., _T]:
        """Wrap ``func`` in the run's root trace, unless tracing is off."""
        if not self.traced:
            return func
        content = include_content()
        tracked: Callable[..., _T] = opik.track(
            name=self.use_case.name,
            project_name=self.fields.project_name,
            tags=list(self.fields.tags),
            metadata=dict(self.fields.metadata),
            environment=self.fields.environment,
            capture_input=content,
            capture_output=content,
        )(func)
        return tracked

    def _current_trace(self) -> TraceData | None:
        """Return the trace this run marks, or ``None`` when it is not traced.

        Read from the context each time rather than trusted from the start of the
        run: tracing can be switched on at runtime after the run started without
        a trace, and marking a trace that is not there raises.
        """
        return opik_context.get_current_trace_data() if self.traced else None

    def agent_arguments(self) -> dict[str, Any]:
        """Return what every run passes the agent, and mark the trace.

        For a run the library opened the tags are already on the trace; for one
        inside a caller's trace this adds them there. A caller's trace that already
        has a thread keeps it.
        """
        if (trace := self._current_trace()) is not None:
            opik_context.update_current_trace(
                tags=list(self.fields.tags),
                thread_id=None if trace.thread_id else self.attribution.thread_id,
            )
        return {
            "usage_limits": usage_limits(self.use_case),
            "model_settings": attribution_model_settings(self.attribution),
            "conversation_id": self.attribution.thread_id,
        }

    def replacement(self, exc: Exception, answers: Sequence[str]) -> LLMError | None:
        """Return the library error to raise in place of ``exc``, if any.

        It names the model that gave the last response, which for invalid or
        filtered output is the one that produced it.
        """
        model = answers[-1] if answers else self.use_case.model
        error = classify_run_failure(exc, use_case=self.use_case.name, model=model)
        return None if error is exc else error

    def finish(self, result: AgentRunResult[OutputT]) -> LLMResult[OutputT]:
        """Apply ``postprocess``, mark a degraded result, and build the result."""
        new_messages = result.new_messages()
        model = answering_model(new_messages) or self.use_case.model
        reasons: list[DegradedReason] = []
        if model != self.use_case.model:
            reasons.append("fallback_model")
        output = result.output
        if self.postprocess is not None:
            processed = self.postprocess(output)
            output = processed.output
            reasons.extend(processed.degraded_reasons)
        degraded_reasons = tuple(dict.fromkeys(reasons))
        if degraded_reasons and (trace := self._current_trace()) is not None:
            # Another run in the same trace may have marked it already.
            earlier = (trace.metadata or {}).get("degraded_reason")
            marked = cast("list[DegradedReason]", earlier.split(",") if earlier else [])
            opik_context.update_current_trace(
                metadata=dict(degraded_metadata([*marked, *degraded_reasons]))
            )
        return LLMResult(
            output=output,
            model=model,
            usage=result.usage,
            degraded=bool(degraded_reasons),
            degraded_reasons=degraded_reasons,
            prompt_name=None,
            prompt_version=None,
            new_messages=new_messages,
        )

    @contextmanager
    def trace_scope(self, user_prompt: str | None) -> Generator[_TraceOutput]:
        """Hold the run's root trace current for the block, unless tracing is off.

        The trace opened here matches the one ``track`` opens. It records the
        block's exception, or else the output set on what this yields.
        """
        outcome = _TraceOutput()
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
            func=_root_input, track_options=options, args=(user_prompt,), kwargs={}
        )
        try:
            yield outcome
        except BaseException as exc:
            # A cancelled block closes the trace too, so the caller's context is
            # left as it was.
            _TRACKER._after_call(  # noqa: SLF001
                output=None,
                error_info=error_info_collector.collect(exc)
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

    def stream_replacement(
        self, exc: Exception, answers: Sequence[str]
    ) -> LLMError | None:
        """Return the library error for a failure while a stream is read, if any.

        Unlike ``replacement`` it also classifies a provider failure, which reaches
        the reader of an open stream before ``PolicyModel`` sees it.
        """
        model = answers[-1] if answers else self.use_case.model
        name = self.use_case.name
        if (error := classify_failure(exc, use_case=name, model=model)) is not None:
            return error
        return self.replacement(exc, answers)


@dataclass
class LLMStream(Generic[OutputT]):
    """A streamed run, yielded by ``run_stream``.

    Read it with ``stream_text`` or ``stream_output``, then call ``result``. A model
    failure while reading raises an ``LLMError``.
    """

    _stream: StreamedRunResult[Any, OutputT]
    _run: _Run[OutputT]
    _answers: list[str]
    _result: LLMResult[OutputT] | None = field(default=None, init=False)

    @contextmanager
    def _library_errors(self) -> Generator[None]:
        try:
            yield
        except Exception as exc:
            if (error := self._run.stream_replacement(exc, self._answers)) is None:
                raise
            raise error from exc

    async def stream_text(self, *, delta: bool = True) -> AsyncIterator[str]:
        """Stream the response text.

        :param delta: yield each new piece of text; false yields the whole text so
            far each time.
        :returns: the text.
        :rtype: AsyncIterator[str]
        :raises LLMError: on a model failure after the stream opened.
        """
        with self._library_errors():
            async for text in self._stream.stream_text(delta=delta):
                yield text

    async def stream_output(self) -> AsyncIterator[OutputT]:
        """Stream the output, validated in partial mode as it arrives.

        :returns: each partial output, then the complete one.
        :rtype: AsyncIterator
        :raises LLMError: on a model failure after the stream opened.
        """
        with self._library_errors():
            async for output in self._stream.stream_output():
                yield output

    async def result(self) -> LLMResult[OutputT]:
        """Return the result, reading what is left of the stream first.

        Call it inside the ``async with``: it runs ``postprocess`` and marks the
        trace, which closes when the block ends.

        :returns: the result, with the same markers ``run`` sets.
        :rtype: LLMResult
        :raises LLMError: on a model failure or invalid output.
        """
        if self._result is None:
            if not self._stream.is_complete:
                with self._library_errors():
                    await self._stream.get_output()
            self._result = self._run.finish(self._stream.result)
        return self._result


def _start(
    agent: Agent[Any, OutputT],
    attribution: Attribution,
    postprocess: Callable[[OutputT], PostProcessed[OutputT]] | None,
) -> _Run[OutputT]:
    if not isinstance(agent.model, Model):
        msg = f"{agent!r} has no model from ol_llm.build_model"
        raise TypeError(msg)
    use_case = call_site(agent.model)
    fields = trace_fields(
        use_case,
        environment=get_settings().environment,
        policy=resolve_policy(use_case),
        attribution=attribution,
        model_tags=catalog_tags(agent.model),
    )
    return _Run(use_case, attribution, postprocess, fields, traced=enabled())


async def run(  # noqa: PLR0913
    agent: Agent[DepsT, OutputT],
    user_prompt: str | None = None,
    *,
    deps: DepsT = None,  # type: ignore[assignment]
    attribution: Attribution = NO_ATTRIBUTION,
    message_history: Sequence[ModelMessage] | None = None,
    postprocess: Callable[[OutputT], PostProcessed[OutputT]] | None = None,
) -> LLMResult[OutputT]:
    """Run an agent from ``build_agent`` under a traced root.

    :param agent: the agent.
    :param user_prompt: the user's message.
    :param deps: the run's dependencies.
    :param attribution: the per-call attribution fields.
    :param message_history: earlier messages in the conversation.
    :param postprocess: called on the validated output inside the trace; returns
        the output to use and any degraded reasons. An exception it raises reaches
        the caller unchanged.
    :returns: the result.
    :rtype: LLMResult
    :raises LLMError: on a model failure, invalid output, a content filter, or an
        exceeded usage limit.
    :raises TypeError: if the agent's model did not come from ``build_model``.
    """
    state = _start(agent, attribution, postprocess)
    results: list[LLMResult[OutputT]] = []

    async def traced(user_prompt: str | None) -> OutputT:
        arguments = state.agent_arguments()
        with recording_answers() as answers:
            try:
                result = await agent.run(
                    user_prompt, deps=deps, message_history=message_history, **arguments
                )
            except Exception as exc:
                if (error := state.replacement(exc, answers)) is None:
                    raise
                raise error from exc
        results.append(state.finish(result))
        return results[0].output

    await state.track(traced)(user_prompt)
    return results[0]


def run_sync(  # noqa: PLR0913
    agent: Agent[DepsT, OutputT],
    user_prompt: str | None = None,
    *,
    deps: DepsT = None,  # type: ignore[assignment]
    attribution: Attribution = NO_ATTRIBUTION,
    message_history: Sequence[ModelMessage] | None = None,
    postprocess: Callable[[OutputT], PostProcessed[OutputT]] | None = None,
) -> LLMResult[OutputT]:
    """Run an agent synchronously, for Celery tasks and Dagster assets.

    Takes the same arguments as ``run`` and behaves the same way. Like PydanticAI's
    ``run_sync``, it cannot be called while an event loop is running in the
    thread; use ``run`` there.

    :returns: the result.
    :rtype: LLMResult
    """
    state = _start(agent, attribution, postprocess)
    results: list[LLMResult[OutputT]] = []

    def traced(user_prompt: str | None) -> OutputT:
        arguments = state.agent_arguments()
        with recording_answers() as answers:
            try:
                result = agent.run_sync(
                    user_prompt, deps=deps, message_history=message_history, **arguments
                )
            except Exception as exc:
                if (error := state.replacement(exc, answers)) is None:
                    raise
                raise error from exc
        results.append(state.finish(result))
        return results[0].output

    state.track(traced)(user_prompt)
    return results[0]


@asynccontextmanager
async def run_stream(  # noqa: PLR0913
    agent: Agent[DepsT, OutputT],
    user_prompt: str | None = None,
    *,
    deps: DepsT = None,  # type: ignore[assignment]
    attribution: Attribution = NO_ATTRIBUTION,
    message_history: Sequence[ModelMessage] | None = None,
    postprocess: Callable[[OutputT], PostProcessed[OutputT]] | None = None,
) -> AsyncGenerator[LLMStream[OutputT]]:
    """Stream a run of an agent from ``build_agent`` under a traced root.

    For the request profile. Takes the same arguments as ``run``. The root trace
    stays current until the ``async with`` block ends, so the model spans land in
    it. A failure before the first chunk is retried and falls back as in ``run``
    and raises on entry; a failure after it reaches the caller. When the block
    ends with the stream read, the result is built if ``result`` was not called.

    :returns: the stream.
    :rtype: AbstractAsyncContextManager[LLMStream]
    :raises LLMError: on a model failure, invalid output, a content filter, or an
        exceeded usage limit.
    :raises TypeError: if the agent's model did not come from ``build_model``.
    """
    state = _start(agent, attribution, postprocess)
    with state.trace_scope(user_prompt) as outcome, recording_answers() as answers:
        arguments = state.agent_arguments()
        # What the caller's block or postprocess raised. LLMStream already
        # converts failures reading the stream; anything else is the caller's and
        # passes through unchanged, including past PolicyModel, which would
        # classify a provider error the caller's own code raised.
        callers: Exception | None = None
        try:
            async with agent.run_stream(
                user_prompt, deps=deps, message_history=message_history, **arguments
            ) as streamed:
                stream = LLMStream(streamed, state, answers)
                try:
                    yield stream
                    if streamed.is_complete:
                        outcome.value = (await stream.result()).output
                except Exception as exc:
                    callers = exc
                    raise
        except Exception as exc:
            if callers is None:
                if (error := state.replacement(exc, answers)) is None:
                    raise
                raise error from exc
        if callers is not None:
            raise callers


def _completion(
    use_case: UseCase,
    messages: Sequence[ModelMessage] | str,
    output_type: OutputSpec[Any],
    instructions: str | None,
) -> tuple[Agent[None, Any], str | None, Sequence[ModelMessage] | None]:
    """Return the agent, user prompt, and history a single call runs with."""
    agent = agent_over(
        shared_model(use_case), output_type=output_type, instructions=instructions
    )
    if isinstance(messages, str):
        return agent, messages, None
    # A history that ends in a request is sent as is, with no new user prompt.
    return agent, None, messages


@overload
async def complete(
    use_case: UseCase,
    messages: Sequence[ModelMessage] | str,
    *,
    instructions: str | None = None,
    attribution: Attribution = NO_ATTRIBUTION,
    postprocess: Callable[[str], PostProcessed[str]] | None = None,
) -> LLMResult[str]: ...
@overload
async def complete(
    use_case: UseCase,
    messages: Sequence[ModelMessage] | str,
    *,
    output_type: OutputSpec[OutputT],
    instructions: str | None = None,
    attribution: Attribution = NO_ATTRIBUTION,
    postprocess: Callable[[OutputT], PostProcessed[OutputT]] | None = None,
) -> LLMResult[OutputT]: ...
async def complete(  # noqa: PLR0913
    use_case: UseCase,
    messages: Sequence[ModelMessage] | str,
    *,
    output_type: OutputSpec[Any] = str,
    instructions: str | None = None,
    attribution: Attribution = NO_ATTRIBUTION,
    postprocess: Callable[[Any], PostProcessed[Any]] | None = None,
) -> LLMResult[Any]:
    """Make one model call for a use case, with no agent to build.

    The same as ``run`` over ``build_agent(use_case, output_type=...,
    instructions=...)``: the same model, policy, trace, and errors. The model is
    built once per use case and reused.

    :param use_case: the call site.
    :param messages: the user's message, or the conversation so far ending in the
        request to send.
    :param output_type: the output spec, ``str`` for free text; ``ToolOutput``
        names the tool the model is made to call.
    :param instructions: the instructions.
    :param attribution: the per-call attribution fields.
    :param postprocess: as for ``run``.
    :returns: the result.
    :rtype: LLMResult
    :raises LLMError: on a model failure, invalid output, a content filter, or an
        exceeded usage limit.
    :raises ValueError: if ``NativeOutput`` is asked of a model that lacks it, or
        ``build_model`` rejects the use case.
    """
    agent, user_prompt, history = _completion(
        use_case, messages, output_type, instructions
    )
    return await run(
        agent,
        user_prompt,
        attribution=attribution,
        message_history=history,
        postprocess=postprocess,
    )


@overload
def complete_sync(
    use_case: UseCase,
    messages: Sequence[ModelMessage] | str,
    *,
    instructions: str | None = None,
    attribution: Attribution = NO_ATTRIBUTION,
    postprocess: Callable[[str], PostProcessed[str]] | None = None,
) -> LLMResult[str]: ...
@overload
def complete_sync(
    use_case: UseCase,
    messages: Sequence[ModelMessage] | str,
    *,
    output_type: OutputSpec[OutputT],
    instructions: str | None = None,
    attribution: Attribution = NO_ATTRIBUTION,
    postprocess: Callable[[OutputT], PostProcessed[OutputT]] | None = None,
) -> LLMResult[OutputT]: ...
def complete_sync(  # noqa: PLR0913
    use_case: UseCase,
    messages: Sequence[ModelMessage] | str,
    *,
    output_type: OutputSpec[Any] = str,
    instructions: str | None = None,
    attribution: Attribution = NO_ATTRIBUTION,
    postprocess: Callable[[Any], PostProcessed[Any]] | None = None,
) -> LLMResult[Any]:
    """Make one model call synchronously, for Celery tasks and Dagster assets.

    Takes the same arguments as ``complete`` and behaves the same way; like
    ``run_sync``, it cannot be called while an event loop is running in the thread.

    :returns: the result.
    :rtype: LLMResult
    """
    agent, user_prompt, history = _completion(
        use_case, messages, output_type, instructions
    )
    return run_sync(
        agent,
        user_prompt,
        attribution=attribution,
        message_history=history,
        postprocess=postprocess,
    )
