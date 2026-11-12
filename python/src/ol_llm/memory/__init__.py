"""Learner memory and general memory behind one narrow interface (spec §5.3, §5.5).

An app builds one ``Memory`` from configuration at startup and, at the start of
each request, asks it for the learner's memory with the learner's session token::

    memory = ol_llm.memory.build_memory()  # OL_LLM_MEMORY_* settings

    notes = await memory.learner(session_token, use_case="tutor-recall")
    await notes.put("pace", "Has 30 minutes per weekday.")
    found = await notes.search("how much time does the learner have?")

Sync code wraps it in ``SyncMemory``, which runs each call on the thread's ol-llm
event loop (``ol_llm.run_coroutine``).

The backend (``hindsight`` for the deployed service, ``in-memory`` for tests and
local development, or one another package registers) is chosen by configuration,
so it can change without touching call sites. The ``hindsight`` backend needs the
``memory`` extra; the rest of the interface needs none.
"""

from typing import TYPE_CHECKING, Any

from ol_llm.memory.config import (
    ENTRY_POINT_GROUP,
    BackendFactory,
    MemoryConfig,
    backend_factory,
    build_memory,
    hindsight_backend,
    register_backend,
)
from ol_llm.memory.errors import (
    MemoryFenced,
    MemoryForbidden,
    MemoryNotFound,
    MemoryRejected,
    MemoryStoreError,
    MemoryTimeout,
    MemoryUnauthenticated,
    MemoryUnavailable,
    MemoryUnsupported,
)
from ol_llm.memory.in_memory import InMemoryBackend, InMemoryStore
from ol_llm.memory.interface import (
    ALL_CAPABILITIES,
    Capability,
    ForgetStore,
    LearnerCredential,
    MemoryBackend,
    Note,
    NoteStore,
    ServiceCredential,
    TokenSource,
)
from ol_llm.memory.service import (
    AdminMemory,
    GeneralMemory,
    LearnerMemory,
    Memory,
    SyncAdminMemory,
    SyncGeneralMemory,
    SyncLearnerMemory,
    SyncMemory,
)
from ol_llm.memory.tokens import (
    ClientCredentialsTokens,
    KeycloakExchangeClient,
    KeycloakTokenExchange,
    TokenExchange,
)

if TYPE_CHECKING:
    from ol_llm.memory.hindsight import HindsightBackend


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Import the Hindsight backend, which needs the ``memory`` extra, on first use."""
    if name != "HindsightBackend":
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return hindsight_backend()


__all__ = [
    "ALL_CAPABILITIES",
    "ENTRY_POINT_GROUP",
    "AdminMemory",
    "BackendFactory",
    "Capability",
    "ClientCredentialsTokens",
    "ForgetStore",
    "GeneralMemory",
    "HindsightBackend",
    "InMemoryBackend",
    "InMemoryStore",
    "KeycloakExchangeClient",
    "KeycloakTokenExchange",
    "LearnerCredential",
    "LearnerMemory",
    "Memory",
    "MemoryBackend",
    "MemoryConfig",
    "MemoryFenced",
    "MemoryForbidden",
    "MemoryNotFound",
    "MemoryRejected",
    "MemoryStoreError",
    "MemoryTimeout",
    "MemoryUnauthenticated",
    "MemoryUnavailable",
    "MemoryUnsupported",
    "Note",
    "NoteStore",
    "ServiceCredential",
    "SyncAdminMemory",
    "SyncGeneralMemory",
    "SyncLearnerMemory",
    "SyncMemory",
    "TokenExchange",
    "TokenSource",
    "backend_factory",
    "build_memory",
    "hindsight_backend",
    "register_backend",
]
