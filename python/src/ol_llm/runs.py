"""The agent entry points (spec §6.6): one Opik root trace per run.

``run`` and ``run_sync`` open the root trace with the §4 attribution fields, run the
agent under the use case's usage limits, and return an ``LLMResult``. The agent's
model spans land inside that trace through ``OpikSpanProcessor`` (see
``ol_llm.tracing``). Model failures reach the caller as ``LLMError`` subclasses,
and the trace records the same error.

A run inside a caller's own ``@opik.track`` becomes a span of the caller's trace:
the attribution tags and metadata go on that span, and ``thread_id`` and the
degraded marking go on the caller's trace.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import opik
from opik import opik_context
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

from ol_llm.attribution import (
    NO_ATTRIBUTION,
    Attribution,
    DegradedReason,
    degraded_metadata,
    trace_fields,
)
from ol_llm.errors import LLMError
from ol_llm.models import (
    answering_model,
    attribution_model_settings,
    call_site,
    classify_run_failure,
    usage_limits,
)
from ol_llm.policy import UseCase, resolve_policy
from ol_llm.settings import get_settings
from ol_llm.tracing import include_content

OutputT = TypeVar("OutputT")
DepsT = TypeVar("DepsT")
_T = TypeVar("_T")


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

    def track(self, func: Callable[..., _T]) -> Callable[..., _T]:
        """Wrap ``func`` in the run's root trace."""
        fields = trace_fields(
            self.use_case,
            environment=get_settings().environment,
            policy=resolve_policy(self.use_case),
            attribution=self.attribution,
        )
        content = include_content()
        tracked: Callable[..., _T] = opik.track(
            name=self.use_case.name,
            project_name=fields.project_name,
            tags=list(fields.tags),
            metadata=dict(fields.metadata),
            environment=fields.environment,
            capture_input=content,
            capture_output=content,
        )(func)
        return tracked

    def agent_arguments(self) -> dict[str, Any]:
        """Return what every run passes the agent, and mark the trace's thread."""
        if self.attribution.thread_id is not None:
            opik_context.update_current_trace(thread_id=self.attribution.thread_id)
        return {
            "usage_limits": usage_limits(self.use_case),
            "model_settings": attribution_model_settings(self.attribution),
            "conversation_id": self.attribution.thread_id,
        }

    def replacement(self, exc: Exception) -> LLMError | None:
        """Return the library error to raise in place of ``exc``, if any."""
        error = classify_run_failure(
            exc, use_case=self.use_case.name, model=self.use_case.model
        )
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
        if degraded_reasons:
            opik_context.update_current_trace(
                metadata=dict(degraded_metadata(degraded_reasons))
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


def _start(
    agent: Agent[Any, OutputT],
    attribution: Attribution,
    postprocess: Callable[[OutputT], PostProcessed[OutputT]] | None,
) -> _Run[OutputT]:
    if not isinstance(agent.model, Model):
        msg = f"{agent!r} has no model from ol_llm.build_model"
        raise TypeError(msg)
    return _Run(call_site(agent.model), attribution, postprocess)


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
        try:
            result = await agent.run(
                user_prompt, deps=deps, message_history=message_history, **arguments
            )
        except Exception as exc:
            if (error := state.replacement(exc)) is None:
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

    Takes the same arguments as ``run`` and behaves the same way.

    :returns: the result.
    :rtype: LLMResult
    """
    state = _start(agent, attribution, postprocess)
    results: list[LLMResult[OutputT]] = []

    def traced(user_prompt: str | None) -> OutputT:
        arguments = state.agent_arguments()
        try:
            result = agent.run_sync(
                user_prompt, deps=deps, message_history=message_history, **arguments
            )
        except Exception as exc:
            if (error := state.replacement(exc)) is None:
                raise
            raise error from exc
        results.append(state.finish(result))
        return results[0].output

    state.track(traced)(user_prompt)
    return results[0]
