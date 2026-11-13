"""Opik's LangChain tracer for ``traced_chat``, pricing every chat span.

A LangChain run is not an OpenTelemetry span, so the Opik destination records a
chat turn's runs with Opik's own LangChain tracer, attached to the call's root
span. Every chat model span names the use case's model as the catalog's Opik price
table names it, with the token usage LangChain reports, so Opik prices it whatever
chat model class made the call (``ChatLiteLLM`` reports usage Opik's own LangChain
extractors do not read).

Needs the ``langchain`` extra.
"""

import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import opik
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables.graph import Graph
from langchain_core.tracers.schemas import Run
from opik import context_storage, opik_context
from opik.api_objects.span import SpanData
from opik.integrations.langchain import OpikTracer
from opik.types import DistributedTraceHeadersDict, LLMProvider

from ol_llm.catalog import CatalogEntry, load_catalog
from ol_llm.policy import UseCase
from ol_llm.trace_destinations.interface import CallTrace
from ol_llm.trace_destinations.opik_destination import error_info

GRAPH_METADATA_KEY = "_opik_graph_definition"
"""The trace metadata key Opik's "Show Agent Graph" view reads."""

CUT_OFF_METADATA_KEY = "cut_off"
"""Set on the span of a run still open when the ``traced_chat`` block exited."""


@contextmanager
def chat_tracer(
    call: CallTrace,
    *,
    use_case: UseCase,
    metadata: Mapping[str, Any],
    graph: Graph | None,
) -> Generator[Sequence[BaseCallbackHandler]]:
    """Yield a ``ChatTracer`` attached to the call's root span, and close it after.

    Yields no callbacks when the SDK opened no root span for the call.

    :param call: the ``traced_chat`` call, entered.
    :param use_case: the call site.
    :param metadata: call-site fields for the spans of root runs.
    :param graph: the LangGraph graph, for Opik's agent graph view.
    """
    root = opik_context.get_current_span_data()
    if root is None:
        yield ()
        return
    if graph is not None:
        opik_context.update_current_trace(
            metadata={
                GRAPH_METADATA_KEY: {"format": "mermaid", "data": graph.draw_mermaid()}
            }
        )
    tracer = ChatTracer(
        use_case,
        parent=DistributedTraceHeadersDict(
            opik_trace_id=root.trace_id, opik_parent_span_id=root.id
        ),
        project_name=root.project_name,
        environment=root.environment,
        metadata=dict(metadata),
        content=call.content,
    )
    try:
        yield (tracer,)
    finally:
        tracer.close()


class ChatTracer(OpikTracer):
    """Opik's LangChain tracer, attached to one root trace, pricing every chat span.

    Opik's tracer opens a trace of its own when it cannot see the caller's, e.g. on
    a worker thread; given the root span as distributed headers, every run attaches
    to it instead. Each span the tracer sends passes through ``_SpanSink``, which
    gives chat spans the catalog's model and provider and LangChain's token usage,
    puts the root trace's environment on every span, and leaves content out when
    the trace carries none. Each chat span names the use case's model, even when a
    fallback served the call.

    Opik's tracer pushes each run's span onto the Opik context of the task that
    started the run, which for a stream is the caller's, so a ``@track`` function in
    a node nests under it. ``close`` ends the block's spans and removes them from
    that context.

    Opik's tracer has no retriever hooks, so it records no retriever span and drops
    any run nested in one for want of a parent span. This tracer records retrievers
    as it does tools.
    """

    def __init__(  # noqa: PLR0913
        self,
        use_case: UseCase,
        *,
        parent: DistributedTraceHeadersDict,
        project_name: str | None,
        environment: str | None,
        metadata: dict[str, Any],
        content: bool,
    ) -> None:
        """Attach the runs to ``parent``.

        :param use_case: the call site; its model names the chat spans.
        :param parent: the root span's trace and span ids.
        :param project_name: the root span's project.
        :param environment: the root span's environment.
        :param metadata: metadata for the spans of root runs.
        :param content: whether spans keep their input, output, and error
            messages.
        """
        super().__init__(
            metadata=metadata, project_name=project_name, distributed_headers=parent
        )
        self._parent = parent
        # Not _model and _provider: OpikTracer reads its own _provider as a
        # provider override for every span it ends.
        self._chat_model, self._chat_provider = _opik_names(use_case.model)
        self._sink = _SpanSink(environment=environment, content=content, pending={})
        self._started: list[SpanData] = []
        self._closed = False
        # Held while close scans the started spans and while a run's span joins
        # them, so a run admitted on another thread just before close is either
        # ended by close or ends itself.
        self._lock = threading.Lock()

    def close(self) -> None:
        """End the runs the block left open and stop recording runs.

        Called from the task that entered the block, before the root trace ends. A
        block that exits mid-stream leaves the stream's runs open: LangChain ends
        them when the abandoned stream is closed, if ever, after the root trace
        has ended. Even then Opik's tracer drops a run's span once the run's root
        run ends, so a cancelled stream's chat model run, which ends after the
        graph's, never sends its span ended. So every span the tracer started and
        did not end ends here, marked ``CUT_OFF_METADATA_KEY``, and the tracer
        ignores later callbacks, which would send a span a second time. The spans
        record no error: what ended the block, e.g. the caller failing to send a
        chunk, is the root trace's, not the runs'. The spans the runs' start
        callbacks pushed onto this context are removed, leaving the root span on
        top for the call to end.
        """
        with self._lock:
            self._closed = True
            for span in self._started:
                if span.end_time is None:
                    self._cut_off(span)
        context_storage.trim_span_data_stack_to_certain_span(
            self._parent["opik_parent_span_id"]
        )

    def _cut_off(self, span: SpanData) -> None:
        span.init_end_time().update(metadata={CUT_OFF_METADATA_KEY: True})
        self._sink.__internal_api__span__(**span.as_parameters)

    def _skip_tracking(self) -> bool:
        return self._closed or super()._skip_tracking()

    def _emit_start_span(self, span_data: SpanData) -> None:
        # Named from the start, so a chat span the block cuts off, which never
        # sees on_llm_end, still names its model.
        if span_data.type == "llm":
            span_data.update(model=self._chat_model, provider=self._chat_provider)
        with self._lock:
            if self._closed:
                # Admitted before close, started after: its end callbacks are
                # skipped, so it ends here.
                self._cut_off(span_data)
                return
            self._started.append(span_data)
            super()._emit_start_span(span_data)

    def _persist_run(self, run: Run) -> None:
        if not self._closed:
            super()._persist_run(run)

    @property
    def _opik_client(self) -> opik.Opik:
        return self._sink  # type: ignore[return-value]

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Run:  # noqa: ANN401
        """Name the use case's model and the reported usage on the chat span."""
        fields: dict[str, Any] = {
            "model": self._chat_model,
            "provider": self._chat_provider,
        }
        if (usage := _usage(response, self._chat_provider)) is not None:
            fields["usage"] = usage
        self._pend(run_id, fields)
        return super().on_llm_end(response, run_id=run_id, **kwargs)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401
    ) -> Run:
        """Record the error as the entry points do."""
        self._pend_error(run_id, error)
        return super().on_llm_error(error, run_id=run_id, **kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401
    ) -> Run:
        """Record the error as the entry points do."""
        self._pend_error(run_id, error)
        return super().on_chain_error(error, run_id=run_id, **kwargs)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401
    ) -> Run:
        """Record the error as the entry points do."""
        self._pend_error(run_id, error)
        return super().on_tool_error(error, run_id=run_id, **kwargs)

    def _on_retriever_start(self, run: Run) -> None:
        if not self._skip_tracking():
            self._process_start_span(run, allow_duplicating_root_span=True)

    def _on_retriever_end(self, run: Run) -> None:
        if not self._skip_tracking():
            self._process_end_span(run)

    def _on_retriever_error(self, run: Run) -> None:
        if not self._skip_tracking():
            self._process_end_span_with_error(run)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401
    ) -> Run:
        """Record the error as the entry points do."""
        self._pend_error(run_id, error)
        return super().on_retriever_error(error, run_id=run_id, **kwargs)

    def _pend_error(self, run_id: UUID, error: BaseException) -> None:
        # Opik's tracer records every error, a cancellation included, as type
        # "Exception" with LangChain's formatted traceback, which ends with the
        # message and repeats those of chained exceptions.
        self._pend(
            run_id, {"error_info": error_info(error, content=self._sink.content)}
        )

    def _pend(self, run_id: UUID, fields: dict[str, Any]) -> None:
        """Hold fields for the run's span until the tracer sends it, ended."""
        if (span := self.get_current_span_data_for_run(run_id)) is not None:
            self._sink.pending.setdefault(span.id, {}).update(fields)


@dataclass(frozen=True, kw_only=True)
class _SpanSink:
    """The SDK client as ``ChatTracer`` sees it, finishing each span it sends.

    :param environment: the root trace's environment, for every span.
    :param content: whether spans keep their input, output, and error messages.
    :param pending: fields to set on a span when it is sent ended, by span id.
    """

    environment: str | None
    content: bool
    pending: dict[str, dict[str, Any]]

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(opik.get_global_client(), name)

    def __internal_api__span__(self, **span: Any) -> None:  # noqa: ANN401
        """Finish the span's fields, then send it with the SDK's client."""
        if self.environment is not None:
            span["environment"] = self.environment
        if "end_time" in span:
            fields = self.pending.pop(span["id"], {})
            if span.get("error_info") is None:
                fields.pop("error_info", None)
            span.update(fields)
        if not self.content:
            for key in ("input", "output"):
                span.pop(key, None)
        opik.get_global_client().__internal_api__span__(**span)


def _opik_names(model_id: str) -> tuple[str, str]:
    """Return the model and provider names Opik's price table holds a model under.

    The catalog's when it lists them, else the halves of the id.
    """
    entry: CatalogEntry | None = load_catalog().get(model_id)
    if entry is not None and entry.opik_model and entry.opik_provider:
        return entry.opik_model, entry.opik_provider
    provider, _, name = model_id.partition(":")
    return name, provider


def _usage(response: LLMResult, provider: str) -> dict[str, Any] | None:
    """Return the run's token usage in the shape Opik prices ``provider`` from.

    Each prompt's generations are the choices of one response, and a client that
    asks for several (``n``) puts the whole response's usage on every choice, so
    only the first counts.
    """
    total: UsageMetadata | None = None
    for generations in response.generations:
        usages = (
            generation.message.usage_metadata
            for generation in generations
            if isinstance(generation, ChatGeneration)
            and isinstance(generation.message, AIMessage)
            and generation.message.usage_metadata
        )
        if (first := next(usages, None)) is not None:
            total = add_usage(total, first)
    if total is None:
        return None
    return _provider_usage(total, provider)


def _provider_usage(total: UsageMetadata, provider: str) -> dict[str, Any]:
    """Return LangChain's usage in the provider's own shape.

    Opik's cost calculator for Anthropic and Bedrock reads cache reads and writes
    from their own keys and bills the input key as uncached; given another shape
    it bills every input token at the full rate. Its Google calculator reads
    cached tokens from Google's key. LangChain's ``input_tokens`` is the sum of
    every input token type, cache reads and writes included, as OpenAI's
    ``prompt_tokens`` and Google's ``prompt_token_count`` are.
    """
    inputs = total.get("input_token_details", {})
    cache_read = inputs.get("cache_read", 0)
    cache_write = inputs.get("cache_creation", 0)
    reasoning = total.get("output_token_details", {}).get("reasoning", 0)
    uncached = total["input_tokens"] - cache_read - cache_write
    if provider in {LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC_VERTEXAI}:
        return {
            "input_tokens": uncached,
            "output_tokens": total["output_tokens"],
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        }
    if provider == LLMProvider.BEDROCK:
        return {
            "inputTokens": uncached,
            "outputTokens": total["output_tokens"],
            "totalTokens": total["total_tokens"],
            "cacheReadInputTokens": cache_read,
            "cacheWriteInputTokens": cache_write,
        }
    if provider in {LLMProvider.GOOGLE_AI, LLMProvider.GOOGLE_VERTEXAI}:
        # Google counts thinking apart from the reply; LangChain counts both in
        # output_tokens.
        return {
            "prompt_token_count": total["input_tokens"],
            "candidates_token_count": total["output_tokens"] - reasoning,
            "thoughts_token_count": reasoning,
            "total_token_count": total["total_tokens"],
            "cached_content_token_count": cache_read,
        }
    usage: dict[str, Any] = {
        "prompt_tokens": total["input_tokens"],
        "completion_tokens": total["output_tokens"],
        "total_tokens": total["total_tokens"],
    }
    if cache_read:
        usage["prompt_tokens_details"] = {"cached_tokens": cache_read}
    if reasoning:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return usage
