### Added

- `ol_llm.memory`: one interface to learner memory and general memory, with the backend chosen by configuration (`OL_LLM_MEMORY_BACKEND`, or a `MemoryConfig` passed to `build_memory`). `Memory.learner(session_token, use_case=...)` exchanges the learner's session token and returns `put`, `get`, `search`, `delete` and `list` for that learner; the learner always comes from the token. `Memory.general(name, ...)` writes general memory under the general grant, and `Memory.admin(...).forget(learner_id)` deletes everything for one learner and fences later writes.
- `SyncMemory`, the same operations for sync code, run on the thread's ol-llm event loop (`run_coroutine`).
- The `hindsight` backend for the deployed Hindsight service, on `hindsight-client`'s generated API (the `memory` extra), and an `in-memory` backend for tests and local development that enforces the same rules. Other packages add backends with `register_backend` or an `ol_llm.memory_backends` entry point.
- `KeycloakTokenExchange` (RFC 8693 standard token exchange of a session token for the memory audience) and `ClientCredentialsTokens` for general writes and forget. A backend without Keycloak takes neither.
- Typed memory errors: `MemoryNotFound`, `MemoryForbidden`, `MemoryFenced`, `MemoryUnauthenticated`, `MemoryUnavailable` (with `MemoryTimeout`, both retryable), `MemoryRejected`, and `MemoryUnsupported` for a capability the backend lacks.
- `ol_llm.memory.conformance`, the checks every memory backend must pass. Each memory operation is a span in the calling trace with the §4 attribution fields; note text and queries are on it only when `OL_LLM_TRACE_CONTENT` is on.

### Changed

- The `memory` extra now installs `hindsight-client` 1.2.x, matching the deployed Hindsight server (1.2.0), in place of 0.10.x.
- ol-llm now requires `httpx2`, the maintained continuation of `httpx` that the openai, anthropic, mistralai and pydantic-ai-slim SDKs already require. The memory package sends with it: `HindsightBackend(http=...)` and `KeycloakTokenExchange(http=...)` take an `httpx2.AsyncClient`. `ClientCredentialsTokens(http=...)` still takes an `httpx.Client`, because it shares the token cache the Opik SDK's `httpx` hooks use.
