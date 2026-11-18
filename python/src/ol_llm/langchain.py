"""Call spans for chat calls that a LangChain or LangGraph call site makes itself.

``traced_chat`` gives a LangChain chat model's calls the call span the entry
points open (usage policies spec §4.1, library spec §6.6): the §4 attribution
fields, the catalog checks ``build_model`` runs, and a ``model_status`` tag for a
deprecated or unlisted model. It yields the LangChain callbacks the trace
destinations give for recording the call site's runs; for Opik, its LangChain
tracer, which names and prices every chat model span by the catalog
(``ol_llm.trace_destinations.opik_langchain``). Retries, timeouts, and errors stay
with the caller's client; ``chat_litellm_kwargs`` turns the use case's policy into
settings for a langchain-litellm ``ChatLiteLLM``.

Needs the ``langchain`` extra.
"""

from collections.abc import Generator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.graph import Graph

from ol_llm.attribution import NO_ATTRIBUTION, Attribution
from ol_llm.catalog import check_chat_model
from ol_llm.policy import UseCase, resolve_policy
from ol_llm.traces import TracedCall, TraceOutput, current_call
from ol_llm.tracing import destinations, enabled


def chat_litellm_kwargs(
    use_case: UseCase, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Return ``ChatLiteLLM`` settings that apply the use case's call policy.

    The timeout becomes ``request_timeout``. litellm hands it to the provider's
    OpenAI-SDK client as the httpx timeout, which bounds each read, so on a stream
    it bounds the wait for the response and the gap between chunks, never the
    whole reply (checked in litellm 1.105.7 and openai 2.54.0).

    The retries go to that client (``model_kwargs["max_retries"]``), which retries
    timeouts, connection errors, 408, 409, 429 and 5xx before it returns the
    response, so a stream is never retried once its response has started.
    ``ChatLiteLLM.max_retries`` counts attempts, not retries, and its retry wraps
    only the start of a stream too; it is set to 1 so the two layers don't
    multiply. Do not set litellm's ``num_retries``, which retries the whole call
    again on top.

    Where this departs from the entry points' policy:

    - The SDK retries until the response headers arrive, not until the first
      chunk, so a stream that stalls after its headers fails without a retry.
    - The SDK's waits are not held to ``max_backoff_seconds``. It waits as long as
      a Retry-After on any retryable status asks, up to 120 s, and does not retry
      past that; otherwise it backs off 0.5 s doubling to at most 8 s per wait
      (1.5 s over the default 2 retries, about 15.5 s over 5).
    - The server can force or suppress a retry with ``x-should-retry``.
    - On ``azure/``, litellm caches the client without keying on ``max_retries``,
      so the first use case to build an Azure client sets the retries for every
      use case sharing it; the timeout still applies per request.
    - Only providers litellm calls through the OpenAI SDK (``openai/``,
      ``azure/``, OpenAI-compatible proxies) read ``max_retries``; others drop
      it.
    - ``fallback_models`` is not applied.

    For example, with the caller's own ``model_kwargs`` merged in::

        policy = ol_llm.chat_litellm_kwargs(TUTOR)
        ChatLiteLLM(
            model=model,
            streaming=True,
            **{**policy, "model_kwargs": {**policy["model_kwargs"], **model_kwargs}},
        )

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
        "model_kwargs": {"max_retries": policy.max_retries},
    }


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
