### Changed

- On models without mid-conversation system messages (in the catalog: every Bedrock and Google model, and `anthropic:claude-sonnet-5`), every request with a system prompt or instructions now carries a fixed sentence explaining PydanticAI's `<system>` tag in its system field. It goes after any `system_prompt=` text and ahead of the instructions, so for an agent built with `instructions=` alone, as ol-llm's entry points are, it is the first system block. PydanticAI 2.63 added it (pydantic/pydantic-ai#11506) and offers no setting to turn it off. Each cached prompt prefix takes one cache miss after the upgrade, and tests that assert the first system block is the app's prompt need updating. OpenAI, Azure, Mistral and `anthropic:claude-haiku-5-5` requests are unchanged.

### Security

- Every PydanticAI extra now requires `pydantic-ai-slim>=2.66`, which fixes eight advisories published on 2026-11-04 that affect 2.65 and earlier, among them GHSA-m5vj-8mcw-82qx (high) and GHSA-jp94-mmr3-2pvv.
- The `harness` extra moves to `pydantic-ai-harness` 0.66, which fixes exponential backtracking in the `private_key` secret-redaction pattern (GHSA-mfx5-pc32-32m8). The extra's old `<0.61` cap held consumers on affected releases. 0.66 pins PydanticAI 2.66.
