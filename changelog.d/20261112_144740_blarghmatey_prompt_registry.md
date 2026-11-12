### Added

- `ol_llm.prompt_registry`: the prompt registry behind `get_prompt`, `fetch_prompt`, `create_prompt`, `promote_prompt` and `ol-llm-prompts`, chosen by name like memory backends. `OL_LLM_PROMPT_REGISTRY` is `opik` by default, or `file` with `OL_LLM_PROMPT_REGISTRY_PATH` set to an exports directory, which serves the committed exports at their versions for tests and local work without Opik. The file registry serves them under `production`, or under the labels in `OL_LLM_PROMPT_REGISTRY_LABELS` (e.g. `production,ci`). `configure_prompt_registry` sets one in code, and the `ol_llm.prompt_registries` entry-point group adds others.

### Changed

- ol-llm renders prompt templates itself instead of through Opik's template classes. Text prompts render exactly as before. A mustache template given the wrong variables raises `ol_llm.PromptVariablesError` (a `ValueError`) instead of Opik's `PromptPlaceholdersDontMatchFormatArguments`. A chat template now keeps every message key, messages without a `role`, a `content` of `None`, and content parts Opik's renderer did not know; an `audio_url` part stays a part where Opik flattened the message to text; and a jinja2 error in a chat message raises where Opik sent the raw template. A media part whose URL renders empty is still left out.
- `get_prompt` caches fetched versions itself for five minutes (`ol_llm.prompts.PROMPT_CACHE_TTL_SECONDS`, at most 128 of them). The first call after that fetches again and keeps the cached version if the fetch fails, finds nothing, or misses the two-second deadline. `OPIK_PROMPT_CACHE_TTL_SECONDS` no longer applies.
- A default of the other structure (text where the registry holds chat, or the reverse) raises `TypeError` from `get_prompt` and `create_prompt`, instead of Opik's `PromptTemplateStructureMismatch`. `promote_prompt` raises `LookupError` for a missing prompt or version.
- `fetch_prompt`, and so `ol-llm-prompts` and the evaluation gate, raise `ol_llm.prompt_registry.RegistryUnavailableError` when Opik cannot be reached or answers 5xx, chained to the transport error or `ApiError` they raised before.
- `ManagedPrompt.registry_ref` is typed `object | None` instead of Opik's `BasePrompt | None`.
- `promote_prompt` and `ol-llm-prompts promote` register a label Opik does not have as an environment before moving it, instead of failing.
- jinja2 is a direct dependency (it was already installed through opik).
