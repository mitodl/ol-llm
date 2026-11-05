### Security

- The `harness` extra moves to `pydantic-ai-harness` 0.66, which fixes exponential backtracking in the `private_key` secret-redaction pattern (GHSA-mfx5-pc32-32m8). The extra's old `<0.61` cap held consumers on affected releases. 0.66 pins PydanticAI 2.66.
- Every PydanticAI extra now requires `pydantic-ai-slim>=2.66`, which fixes eight advisories that affect 2.60, among them GHSA-m5vj-8mcw-82qx (high, web fetch memory blow-up) and GHSA-jp94-mmr3-2pvv (embedding spans recording `extra_headers` and `extra_body`). Google models now ask Google for a server deadline of at least 20 minutes; the local timeout is still the policy's.
