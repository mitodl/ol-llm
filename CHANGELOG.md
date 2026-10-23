# Changelog

All notable changes to ol-llm are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries are collected from
`changelog.d/` at release; see `changelog.d/README.md` to add one.

<!-- scriv-insert-here -->

<a id='changelog-0.0.5'></a>
## [0.0.5] - 2026-10-23

### Added

- `model_settings=` and `output_validators=` on `complete`, `complete_sync`, `build_agent`, and `agent_over`. `model_settings` refuses `timeout`, `openai_user`, and `anthropic_metadata` with `ValueError`, since the library sets those (#24).
- `prompt=` takes a sequence of managed prompts on `run`, `run_sync`, `run_stream`, and `complete` (#24).
- `agent_over` is exported from `ol_llm` (#24).
- `ol_llm.evaluation`: golden datasets seeded from files, experiments linked to the prompt versions under test, judge models from the catalog, and advisory or blocking thresholds, plus the `ol-llm-eval` command. An `LLMError` from the task or a judge scores its item as a failure instead of aborting the run. Run it where the Opik SDK is installed whole, not in a slimmed runtime image (#25).

### Changed

- Every provider extra requires `pydantic-ai-slim>=2.60` (was 2.52), and the `harness` extra requires `pydantic-ai-harness>=0.60,<0.61`. A consumer locked to an older pydantic-ai-slim has to move up (#26, #28).
- On Bedrock, pydantic-ai-slim 2.60 sends a Claude request that sets no `max_tokens` with the model's maximum output (64,000 for Claude Haiku 4.5, 128,000 for Sonnet 5) through `ConverseStream`. Bedrock reserves that much tokens-per-minute quota when the request starts, so set `max_tokens` through `model_settings`. The role needs `bedrock:InvokeModelWithResponseStream`, and the policy timeout bounds the gap between chunks (#28).
- A failure after a stream's first event keeps its own error class instead of becoming `LLMUnavailable`, so a Bedrock throttle in the middle of a stream raises `LLMRateLimited` (#26).
- A 424 that Bedrock raises inside a stream is retried and falls back while it is the stream's first event, per spec §6.3. A 424 outside a stream is still `LLMRejected` (#26).

### Fixed

- The `anthropic` extra works with anthropic 2.x. Under 0.0.4 an install that paired anthropic 2 with pydantic-ai-slim older than 2.60 failed every Anthropic request with `TypeError`, and 0.0.5 requires 2.60, which supports it (#28).
- A finished worker thread's event loop is released again under pydantic-ai-slim 2.59 (#26).
- A model that lists `timeout` in `unsupported_settings` (Bedrock) is no longer sent one, so pydantic-ai-slim 2.58 and later do not raise `UnsupportedSettingWarning` on every Bedrock request. Bedrock's timeout is its boto3 client's, as before (#26).

### Security

- The `harness` extra moves off pydantic-ai-harness 0.52, whose pin held pydantic-ai-slim at 2.52.0, to a release whose pydantic-ai-slim carries the fix for GHSA-6fqq-452j-qhrp (#23, #26).

<a id='changelog-0.0.4'></a>
## [0.0.4] - 2026-10-21

### Changed

- With content capture off, no entry point's root trace carries an exception message or source line, only the exception type and bare frames (#18).
- `import ol_llm` defaults `OPIK_ANALYTICS_ENABLE` to false, so the Opik SDK sends no usage events to Comet (#19).
- Requires `pydantic>=2.14`. This makes the base install unresolvable on edx-platform `release/verawood`, which pins pydantic 2.13.3 (#20).

### Fixed

- A cancelled `run`, or one a caller's `asyncio.wait_for` times out, closes its root trace and clears the Opik context (#18).
- Concurrent first OpenAI responses in worker threads no longer fail on pydantic's deferred-model rebuild race (#20).

<a id='changelog-0.0.3'></a>
## [0.0.3] - 2026-10-21

### Added

- `complete` and `complete_sync` for single calls (#10).
- `run_stream` under the run's root trace (#11).
- Managed prompts fetched from Opik, with an embedded fallback marked degraded when Opik is unreachable (#12).
- `ol_llm.batch`: batch helpers for Dagster assets and Celery, with row-level failure records and a circuit breaker (#13).
- `embed` and `embed_sync` under the call policy (#14).
- `traced_embedding` for embedding clients the library does not build, e.g. litellm in mit-learn. It needs no extra (#17).

### Changed

- opik is pinned to 2.2.94 (#7).

### Fixed

- An agent built once works from any event loop: `PolicyModel` builds its provider model for each loop, so `run_sync` from a second thread or a second `asyncio.run` no longer fails. `Settings.azure_ad_token_provider` must be synchronous (#16).

<a id='changelog-0.0.2'></a>
## [0.0.2] - 2026-10-19

### Added

- The model catalog (`catalog.toml`): each model's kind, status, route, and list prices, with `check_model_ids` and the `ol-llm-catalog` command for app-side CI and Prometheus price rules (#5).
- Opik tracing: `run` and `run_sync` open the root trace with the attribution fields, model spans attach to it, and Opik behind Keycloak is reached with a client-credentials token (#8).

### Changed

- `build_model` builds an id the catalog does not list, with a warning and a `model_status:uncataloged` trace tag. A retired id raises, and a deprecated one is tagged `model_status:deprecated` (#9).

<a id='changelog-0.0.1'></a>
## [0.0.1] - 2026-10-15

### Added

- The library core: use cases with `request` and `batch` profiles, the call policy (one retry layer and a timeout on every request), `build_model` and `build_agent`, declared fallbacks, the `LLMError` classes, attribution fields, and `ol_llm.testing`. PydanticAI, the harness, and memory are opt-in extras (#4).
