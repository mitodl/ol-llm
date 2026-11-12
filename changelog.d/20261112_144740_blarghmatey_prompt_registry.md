### Added

- `ol_llm.prompt_registry`: the prompt registry behind `get_prompt`, `fetch_prompt`, `create_prompt`, `promote_prompt` and `ol-llm-prompts`, chosen by name like memory backends. `OL_LLM_PROMPT_REGISTRY` is `opik` by default, or `file` with `OL_LLM_PROMPT_REGISTRY_PATH` set to an exports directory, which serves the committed exports at their versions and under the `production` label for tests and local work without Opik. `configure_prompt_registry` sets one in code, and the `ol_llm.prompt_registries` entry-point group adds others.

### Changed

- ol-llm renders prompt templates itself instead of through Opik's template classes. Mustache output is the same. A mustache template given the wrong variables raises `ol_llm.PromptVariablesError` (a `ValueError`) instead of Opik's `PromptPlaceholdersDontMatchFormatArguments`. A chat template keeps every message key and content part, where Opik's renderer kept only `role` and `content` and dropped parts it did not know.
- `get_prompt` caches fetched versions itself for five minutes (`ol_llm.prompts.PROMPT_CACHE_TTL_SECONDS`), then serves the cached version while it refreshes in the background. `OPIK_PROMPT_CACHE_TTL_SECONDS` no longer applies.
- A default of the other structure (text where the registry holds chat, or the reverse) raises `TypeError` from `get_prompt` and `create_prompt`, instead of Opik's `PromptTemplateStructureMismatch`. `promote_prompt` raises `LookupError` for a missing prompt or version.
- `promote_prompt` and `ol-llm-prompts promote` register a label Opik does not have as an environment before moving it, instead of failing.
- jinja2 is a direct dependency (it was already installed through opik).
