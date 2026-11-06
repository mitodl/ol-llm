"""Shared library for LLM calls at MIT Open Learning.

The Opik SDK starts its own Sentry client on import, reporting to Comet's DSN, and
that replaces whatever Sentry client the host application already configured.
Setting OPIK_SENTRY_ENABLE before anything imports opik stops it. The SDK also
reports which of its features get used to Comet's stats collector, which
OPIK_ANALYTICS_ENABLE=false stops. This module sets both first, so every consumer
gets them by importing ol_llm before opik. An explicit value in the environment
wins over either default.

The model factory, the agent builder, and the run and embedding entry points
need the ``pydantic-ai`` extra, and ``traced_chat`` the ``langchain`` extra. They
are imported on first use, so the base package imports without either.
"""

import os
from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

os.environ.setdefault("OPIK_SENTRY_ENABLE", "false")
os.environ.setdefault("OPIK_ANALYTICS_ENABLE", "false")

from ol_llm.attribution import (
    Attribution,
    DegradedReason,
    TraceFields,
    degraded_metadata,
    trace_fields,
)
from ol_llm.batch import (
    BatchFailed,
    BatchKeys,
    BatchReport,
    Chunk,
    CircuitBreaker,
    Provenance,
    RowFailure,
    celery_rate_limit,
    dagster_pool,
    process_batch,
)
from ol_llm.catalog import (
    CatalogEntry,
    check_chat_model,
    check_model_ids,
    load_catalog,
)
from ol_llm.errors import (
    LLMBudgetExceeded,
    LLMContentFiltered,
    LLMError,
    LLMInvalidOutput,
    LLMRateLimited,
    LLMRejected,
    LLMUnavailable,
)
from ol_llm.loops import close_thread_loop, run_coroutine
from ol_llm.policy import (
    CallPolicy,
    Profile,
    UseCase,
    resolve_policy,
    worst_case_seconds,
)
from ol_llm.prompts import (
    ChatTemplate,
    ManagedPrompt,
    TemplateType,
    TextTemplate,
    create_prompt,
    fetch_prompt,
    get_prompt,
    promote_prompt,
)
from ol_llm.settings import (
    GatewayRoute,
    Settings,
    configure,
    get_settings,
)
from ol_llm.traces import EmbeddingRequest, traced_embedding
from ol_llm.tracing import (
    KeycloakClient,
    TracingConfig,
    configure_tracing,
)

if TYPE_CHECKING:
    from ol_llm.embeddings import EmbedResult, embed, embed_sync
    from ol_llm.langchain import ChatTrace, traced_chat
    from ol_llm.models import agent_over, build_agent, build_model, usage_limits
    from ol_llm.runs import (
        LLMResult,
        LLMStream,
        PostProcessed,
        complete,
        complete_sync,
        run,
        run_stream,
        run_sync,
    )

# The version lives only in python/pyproject.toml, which is what a release bumps.
__version__ = version("ol-llm")

_LAZY_EXPORTS = {
    "ChatTrace": "ol_llm.langchain",
    "traced_chat": "ol_llm.langchain",
    "EmbedResult": "ol_llm.embeddings",
    "embed": "ol_llm.embeddings",
    "embed_sync": "ol_llm.embeddings",
    "agent_over": "ol_llm.models",
    "build_agent": "ol_llm.models",
    "build_model": "ol_llm.models",
    "usage_limits": "ol_llm.models",
    "LLMResult": "ol_llm.runs",
    "LLMStream": "ol_llm.runs",
    "PostProcessed": "ol_llm.runs",
    "complete": "ol_llm.runs",
    "complete_sync": "ol_llm.runs",
    "run": "ol_llm.runs",
    "run_stream": "ol_llm.runs",
    "run_sync": "ol_llm.runs",
}
# What each extra installs that the lazily imported modules import directly.
_EXTRA_MODULES = {
    "pydantic_ai": "pydantic-ai",
    "httpx2": "pydantic-ai",
    "langchain_core": "langchain",
}

__all__ = [
    "Attribution",
    "BatchFailed",
    "BatchKeys",
    "BatchReport",
    "CallPolicy",
    "CatalogEntry",
    "ChatTemplate",
    "ChatTrace",
    "Chunk",
    "CircuitBreaker",
    "DegradedReason",
    "EmbedResult",
    "EmbeddingRequest",
    "GatewayRoute",
    "KeycloakClient",
    "LLMBudgetExceeded",
    "LLMContentFiltered",
    "LLMError",
    "LLMInvalidOutput",
    "LLMRateLimited",
    "LLMRejected",
    "LLMResult",
    "LLMStream",
    "LLMUnavailable",
    "ManagedPrompt",
    "PostProcessed",
    "Profile",
    "Provenance",
    "RowFailure",
    "Settings",
    "TemplateType",
    "TextTemplate",
    "TraceFields",
    "TracingConfig",
    "UseCase",
    "agent_over",
    "build_agent",
    "build_model",
    "celery_rate_limit",
    "check_chat_model",
    "check_model_ids",
    "close_thread_loop",
    "complete",
    "complete_sync",
    "configure",
    "configure_tracing",
    "create_prompt",
    "dagster_pool",
    "degraded_metadata",
    "embed",
    "embed_sync",
    "fetch_prompt",
    "get_prompt",
    "get_settings",
    "load_catalog",
    "process_batch",
    "promote_prompt",
    "resolve_policy",
    "run",
    "run_coroutine",
    "run_stream",
    "run_sync",
    "trace_fields",
    "traced_chat",
    "traced_embedding",
    "usage_limits",
    "worst_case_seconds",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Import the API that needs an extra on first use."""
    if (module_name := _LAZY_EXPORTS.get(name)) is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only a missing extra gets the hint; any other import failure is a bug
        # and should surface as itself.
        if (extra := _EXTRA_MODULES.get((exc.name or "").partition(".")[0])) is None:
            raise
        msg = f"ol_llm.{name} needs the ol-llm[{extra}] extra: {exc}"
        raise ImportError(msg) from exc
    return getattr(module, name)
