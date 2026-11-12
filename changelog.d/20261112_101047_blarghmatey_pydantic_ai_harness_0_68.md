### Changed

- The `harness` extra moves to `pydantic-ai-harness` 0.68, which pins PydanticAI 2.68. 0.68 renames several capability parameters and event fields (e.g. `num_results` to `max_results`, `max_duration_secs` to `max_duration_seconds`); the old names still work and emit a deprecation warning. The PydanticAI 2.67 and 2.68 release notes list the other compatibility changes.
- PydanticAI 2.68 caps `google-genai<3` (pydantic/pydantic-ai#12700). An install with the `harness` and `google` extras was held on PydanticAI 2.66 and could resolve google-genai 3, which sends Gemini Developer API requests to `v1`; it now resolves google-genai 2.x and sends them to `v1beta` again.
