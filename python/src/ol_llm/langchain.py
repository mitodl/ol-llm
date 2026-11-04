"""Root traces for chat calls that a LangChain or LangGraph call site makes itself.

``traced_chat`` gives a LangChain chat model's calls the root trace the entry
points open (usage policies spec §4.1, library spec §6.6): the §4 attribution
fields, the catalog checks ``build_model`` runs, and a ``model_status`` tag for a
deprecated or unlisted model. It yields the LangChain callbacks that record the
call site's runs as spans of that trace. Every chat model span names the use case's
model as the catalog's Opik price table names it, with the token usage LangChain
reports, so Opik prices it whatever chat model class made the call (``ChatLiteLLM``
reports usage Opik's own LangChain extractors do not read). Retries, timeouts
(from ``resolve_policy``), and errors stay with the caller's client.

Needs the ``langchain`` extra.
"""

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
from opik import opik_context
from opik.integrations.langchain import OpikTracer
from opik.types import DistributedTraceHeadersDict

from ol_llm.attribution import NO_ATTRIBUTION, Attribution
from ol_llm.catalog import CatalogEntry, check_chat_model, load_catalog
from ol_llm.policy import UseCase
from ol_llm.traces import TracedCall, TraceOutput, error_info
from ol_llm.tracing import enabled, include_content

GRAPH_METADATA_KEY = "_opik_graph_definition"
"""The trace metadata key Opik's "Show Agent Graph" view reads."""


def _root_input(user_prompt: str | None) -> None:
    """Stand in for a traced function, so the trace input matches ``run``'s."""


@dataclass(frozen=True, kw_only=True)
class ChatTrace:
    """What ``traced_chat`` yields: the callbacks for the call site's runs.

    :param callbacks: pass these to the graph or chat model, e.g. in the
        ``config`` of ``astream``. Empty when the call is not traced.
    :param output: what the root trace records as its output once ``record`` is
        called.
    """

    callbacks: Sequence[BaseCallbackHandler]
    output: TraceOutput

    def record(self, output: object) -> None:
        """Record the reply the caller assembled as the root trace's output.

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
    """Trace the LangChain runs of one chat turn under the entry points' root trace.

    Every chat model run the callbacks see must call the use case's model: its span
    names that model. The block's exception passes through unchanged and the trace
    records it. Inputs, outputs, and error messages are recorded only when
    ``include_content`` is on; without it an error keeps its type and frames.

    The trace is the current Opik trace's child, or a new root trace. The runs
    attach to it whatever thread or task LangChain calls the callbacks from.

    For example, around a LangGraph agent's stream::

        with ol_llm.traced_chat(TUTOR, message, attribution=attribution) as chat:
            config = {"callbacks": [*chat.callbacks]}
            reply = "".join(
                [chunk async for chunk in stream(agent, message, config)]
            )
            chat.record(reply)

    :param use_case: the call site, naming a chat model in ``provider:model`` form.
    :param user_prompt: the user's message, recorded as the trace input.
    :param attribution: the per-call attribution fields.
    :param metadata: call-site fields for the spans of the runs, e.g. the course a
        tutor is answering about. The root trace keeps the §4 fields only.
    :param graph: the LangGraph graph, from ``get_graph(xray=True)``, for Opik's
        agent graph view.
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
        call.mark_trace()
        opened = call.traced and opik.is_tracing_active()
        root = opik_context.get_current_span_data() if opened else None
        if root is None:
            yield ChatTrace(callbacks=(), output=output)
            return
        if graph is not None:
            opik_context.update_current_trace(
                metadata={
                    GRAPH_METADATA_KEY: {
                        "format": "mermaid",
                        "data": graph.draw_mermaid(),
                    }
                }
            )
        tracer = ChatTracer(
            use_case,
            parent=DistributedTraceHeadersDict(
                opik_trace_id=root.trace_id, opik_parent_span_id=root.id
            ),
            project_name=root.project_name,
            environment=root.environment,
            metadata=dict(metadata or {}),
        )
        yield ChatTrace(callbacks=(tracer,), output=output)


class ChatTracer(OpikTracer):
    """Opik's LangChain tracer, attached to one root trace, pricing every chat span.

    Opik's tracer opens a trace of its own when it cannot see the caller's, e.g. on
    a worker thread; given the root span as distributed headers, every run attaches
    to it instead. Each span the tracer sends passes through ``_SpanSink``, which
    gives chat spans the catalog's model and provider and LangChain's token usage,
    puts the root trace's environment on every span, and leaves content out when
    ``include_content`` is off.
    """

    def __init__(
        self,
        use_case: UseCase,
        *,
        parent: DistributedTraceHeadersDict,
        project_name: str | None,
        environment: str | None,
        metadata: dict[str, Any],
    ) -> None:
        """Attach the runs to ``parent``.

        :param use_case: the call site; its model names the chat spans.
        :param parent: the root span's trace and span ids.
        :param project_name: the root span's project.
        :param environment: the root span's environment.
        :param metadata: metadata for the spans of root runs.
        """
        super().__init__(
            metadata=metadata, project_name=project_name, distributed_headers=parent
        )
        self._model, self._provider = _opik_names(use_case.model)
        self._sink = _SpanSink(
            environment=environment, content=include_content(), pending={}
        )

    @property
    def _opik_client(self) -> opik.Opik:
        return self._sink  # type: ignore[return-value]

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Run:  # noqa: ANN401
        """Name the use case's model and the reported usage on the chat span."""
        fields: dict[str, Any] = {"model": self._model, "provider": self._provider}
        if (usage := _usage(response)) is not None:
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
        # Opik's tracer records every error as type "Exception" with LangChain's
        # formatted traceback, which ends with the message.
        if isinstance(error, Exception):
            info = error_info(error, content=self._sink.content)
            self._pend(run_id, {"error_info": info})

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


def _usage(response: LLMResult) -> dict[str, Any] | None:
    """Return the run's token usage in OpenAI's shape.

    The SDK reads that shape for any provider. LangChain's ``usage_metadata``
    counts cache reads in ``input_tokens``, as OpenAI's ``prompt_tokens`` does.
    """
    total: UsageMetadata | None = None
    for generations in response.generations:
        for generation in generations:
            if not isinstance(generation, ChatGeneration):
                continue
            message = generation.message
            if isinstance(message, AIMessage) and message.usage_metadata:
                total = add_usage(total, message.usage_metadata)
    if total is None:
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": total["input_tokens"],
        "completion_tokens": total["output_tokens"],
        "total_tokens": total["total_tokens"],
    }
    if cached := total.get("input_token_details", {}).get("cache_read"):
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    if reasoning := total.get("output_token_details", {}).get("reasoning"):
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return usage
