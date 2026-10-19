"""Shared library for LLM calls at MIT Open Learning.

The Opik SDK starts its own Sentry client on import, reporting to Comet's DSN, and
that replaces whatever Sentry client the host application already configured.
Setting OPIK_SENTRY_ENABLE before anything imports opik stops it. This module
does that first, so every consumer gets it by importing ol_llm before opik.

The model factory and agent builder need the ``pydantic-ai`` extra. They are
imported on first use, so the base package imports without it.
"""

import os
from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

os.environ.setdefault("OPIK_SENTRY_ENABLE", "false")

from ol_llm.attribution import (
    Attribution,
    DegradedReason,
    TraceFields,
    degraded_metadata,
    trace_fields,
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
from ol_llm.policy import CallPolicy, Profile, UseCase, resolve_policy
from ol_llm.settings import (
    GatewayRoute,
    Settings,
    configure,
    get_settings,
)

if TYPE_CHECKING:
    from ol_llm.models import build_agent, build_model, usage_limits

# The version lives only in python/pyproject.toml, which is what a release bumps.
__version__ = version("ol-llm")

_PYDANTIC_AI_EXPORTS = frozenset({"build_agent", "build_model", "usage_limits"})
# What the pydantic-ai extra installs that ol_llm.models imports directly.
_PYDANTIC_AI_MODULES = frozenset({"pydantic_ai", "httpx2"})

__all__ = [
    "Attribution",
    "CallPolicy",
    "CatalogEntry",
    "DegradedReason",
    "GatewayRoute",
    "LLMBudgetExceeded",
    "LLMContentFiltered",
    "LLMError",
    "LLMInvalidOutput",
    "LLMRateLimited",
    "LLMRejected",
    "LLMUnavailable",
    "Profile",
    "Settings",
    "TraceFields",
    "UseCase",
    "build_agent",
    "build_model",
    "check_model_ids",
    "configure",
    "degraded_metadata",
    "get_settings",
    "load_catalog",
    "resolve_policy",
    "trace_fields",
    "usage_limits",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Import the PydanticAI-backed API on first use."""
    if name not in _PYDANTIC_AI_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    try:
        module = import_module("ol_llm.models")
    except ModuleNotFoundError as exc:
        # Only a missing extra gets the hint; any other import failure is a bug
        # and should surface as itself.
        if (exc.name or "").partition(".")[0] not in _PYDANTIC_AI_MODULES:
            raise
        msg = f"ol_llm.{name} needs the ol-llm[pydantic-ai] extra: {exc}"
        raise ImportError(msg) from exc
    return getattr(module, name)
