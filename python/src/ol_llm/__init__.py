"""Shared library for LLM calls at MIT Open Learning.

The Opik SDK starts its own Sentry client on import, reporting to Comet's DSN, and
that replaces whatever Sentry client the host application already configured.
Setting OPIK_SENTRY_ENABLE before anything imports opik stops it. The SDK also
reports which of its features get used to Comet's stats collector, which
OPIK_ANALYTICS_ENABLE=false stops. This module sets both first, so every consumer
gets them by importing ol_llm before opik. An explicit value in the environment
wins over either default.

The model factory, the agent builder, and the run and embedding entry points
need the ``pydantic-ai`` extra. They are imported on first use, so the base
package imports without it.
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
from ol_llm.catalog import CatalogEntry, check_model_ids, load_catalog
from ol_llm.errors import (
    LLMBudgetExceeded,
    LLMContentFiltered,
    LLMError,
    LLMInvalidOutput,
    LLMRateLimited,
    LLMRejected,
    LLMUnavailable,
)
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
    from ol_llm.models import build_agent, build_model, usage_limits
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

_PYDANTIC_AI_EXPORTS = {
    "EmbedResult": "ol_llm.embeddings",
    "embed": "ol_llm.embeddings",
    "embed_sync": "ol_llm.embeddings",
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
# What the pydantic-ai extra installs that ol_llm.models imports directly.
_PYDANTIC_AI_MODULES = frozenset({"pydantic_ai", "httpx2"})

__all__ = [
    "Attribution",
    "BatchFailed",
    "BatchKeys",
    "BatchReport",
    "CallPolicy",
    "CatalogEntry",
    "ChatTemplate",
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
    "build_agent",
    "build_model",
    "celery_rate_limit",
    "check_model_ids",
    "complete",
    "complete_sync",
    "configure",
    "configure_tracing",
    "create_prompt",
    "dagster_pool",
    "degraded_metadata",
    "embed",
    "embed_sync",
    "get_prompt",
    "get_settings",
    "load_catalog",
    "process_batch",
    "promote_prompt",
    "resolve_policy",
    "run",
    "run_stream",
    "run_sync",
    "trace_fields",
    "traced_embedding",
    "usage_limits",
    "worst_case_seconds",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Import the PydanticAI-backed API on first use."""
    if (module_name := _PYDANTIC_AI_EXPORTS.get(name)) is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only a missing extra gets the hint; any other import failure is a bug
        # and should surface as itself.
        if (exc.name or "").partition(".")[0] not in _PYDANTIC_AI_MODULES:
            raise
        msg = f"ol_llm.{name} needs the ol-llm[pydantic-ai] extra: {exc}"
        raise ImportError(msg) from exc
    return getattr(module, name)
