# Changelog

All notable changes to ol-llm are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries are collected from
`changelog.d/` at release; see `changelog.d/README.md` to add one.

<!-- scriv-insert-here -->

<a id='changelog-0.0.12'></a>
## [0.0.12] - 2026-11-10

### Added

- The prompt pointer file (`ol_llm.pointers.PromptPointers`): an app's Opik project, the directory of its committed embedded defaults, and, per prompt, the version production runs. `ol-llm-prompts export` writes each pointed version's text into that directory for the pull request to commit; `check` refuses a pointer to a version Opik does not have or a committed export that is not its pointed version and, with `--environment`, reports a label that is not on its pointed version; `promote` moves the `production` label to the pointed versions after resolving all of them. See `docs/prompt-promotion.md`. (#64)
- `ol_llm.load_default(name, directory=..., template=...)` reads a committed export as the embedded default for `get_prompt`. (#64)
- The reusable workflow `.github/workflows/prompt-promotion.yml`: `check` on a pull request, the drift report on a schedule, and `promote` on a push or manual run, which reads the pointer file from the default branch's head, runs one promotion at a time per pointer file, and takes the promotion client from a GitHub environment so no pull request job holds it. The check runs ol-llm at the workflow's own commit and reads only the pointer file and exports from the pull request, so the pull request's code never runs next to the read client. Any other event fails. (#64)

### Changed

- An evaluation gate declaration takes its prompt versions from the pointer file: replace the `[prompts]` table with `pointers = "<path from the declaration>"` and `prompts = ["<name>", ...]`, and add the pointer file (its `project` must equal the gate's `app`). A declaration with a `[prompts]` table can no longer run. (#64)

<a id='changelog-0.0.11'></a>
## [0.0.11] - 2026-11-10

### Added

- `ol_llm.memory`: one interface to learner memory and general memory, with the backend chosen by configuration (`OL_LLM_MEMORY_BACKEND`, or a `MemoryConfig` passed to `build_memory`). `Memory.learner(session_token, use_case=...)` exchanges the learner's session token and returns `put`, `get`, `search`, `delete` and `list` for that learner; the learner always comes from the token. `Memory.general(name, ...)` writes general memory under the general grant, and `Memory.admin(...).forget(learner_id)` deletes everything for one learner and fences later writes.
- `SyncMemory`, the same operations for sync code, run on the thread's ol-llm event loop (`run_coroutine`).
- The `hindsight` backend for the deployed Hindsight service, on `hindsight-client`'s generated API (the `memory` extra), and an `in-memory` backend for tests and local development that enforces the same rules. Other packages add backends with `register_backend` or an `ol_llm.memory_backends` entry point.
- `KeycloakTokenExchange` (RFC 8693 standard token exchange of a session token for the memory audience) and `ClientCredentialsTokens` for general writes and forget. A backend without Keycloak takes neither.
- Typed memory errors: `MemoryNotFound`, `MemoryForbidden`, `MemoryFenced`, `MemoryUnauthenticated`, `MemoryUnavailable` (with `MemoryTimeout`, both retryable), `MemoryRejected`, and `MemoryUnsupported` for a capability the backend lacks.
- `ol_llm.memory.conformance`, the checks every memory backend must pass. Each memory operation is a span in the calling trace with the §4 attribution fields; note text and queries are on it only when `OL_LLM_TRACE_CONTENT` is on.
- `Settings.together_api_key`, so a host that configures with `ol_llm.configure()` can supply Together's key. Unset, the key is still read from `TOGETHER_API_KEY`.

### Changed

- The `memory` extra now installs `hindsight-client` 1.2.x, matching the deployed Hindsight server (1.2.0), in place of 0.10.x.
- ol-llm now requires `httpx2`, the maintained continuation of `httpx` that the openai, anthropic, mistralai and pydantic-ai-slim SDKs already require. The memory package sends with it: `HindsightBackend(http=...)` and `KeycloakTokenExchange(http=...)` take an `httpx2.AsyncClient`. `ClientCredentialsTokens(http=...)` still takes an `httpx.Client`, because it shares the token cache the Opik SDK's `httpx` hooks use.

### Fixed

- When every model in a fallback chain fails on a request that continues a suspended response (OpenAI background mode), the caller gets the primary's error. It got the error of the member holding the suspended job, which is tried first.
- On a retry that starts at a fallback model, the request's span names the model that answered instead of the whole chain.
- With `UsageLimits(count_tokens_before_request=True)` on a use case with no fallback models, the token count request gets the use case's timeout, retries, and error classes. It was sent with the caller's timeout in place of the policy's, never retried, and a provider failure reached the caller as PydanticAI's `ModelHTTPError`.
- Together requests go to the base URL of the installed `TogetherProvider` instead of a copy of it.

<a id='changelog-0.0.10'></a>
## [0.0.10] - 2026-11-10

### Added

- The README documents how uv consumers override the version caps (`name<=X`) that opik 2.2.139 and later put on each of its dependencies, which an application inherits through ol-llm's exact opik pin. Without the overrides, uv fails to resolve an application that needs a newer version than a cap (e.g. `openai==3.46.0` next to opik 2.2.143's `openai<=3.42.1`), or installs the capped version. The opik release ol-llm pins now (2.2.94) has no caps, so nothing needs to change yet. pip has no equivalent. ol-llm now bumps opik once a month to keep the caps close to current.

### Changed

- The reusable evaluation gate workflow (`.github/workflows/evaluation-gate.yml`) installs and runs the calling project with uv 0.13.7 instead of 0.12.20. uv 0.13 makes Python 3.15 the default it downloads: a calling project that pins no Python version (no `.python-version`) and whose `requires-python` allows 3.15 now gets 3.15 when the runner has no matching interpreter, where it got 3.14 before. Pin the version with `uv python pin 3.14` or cap `requires-python` below 3.15 to keep 3.14. uv 0.13 also extracts source distributions with a stricter tar reader that rejects archives containing hard links or unsupported tar extensions; a locked dependency installed from such an sdist fails the gate's install step. The lockfile format is unchanged, so a lock written by uv 0.12 still passes `uv sync --frozen`.

### Fixed

- `ol_llm.evaluation.evaluate` puts `Settings.environment` on each item's trace and on the task's and judges' calls under it. `opik.evaluate` opens those traces with no environment and Opik gives nested spans their parent's, so experiment task and judge spend sat in Opik's default environment instead of the app's, and Opik logged a "nested span under the environment" warning for every call.

- With tracing off (`OPIK_TRACK_DISABLE=true` or `TracingConfig(enabled=False)`), Opik SDK calls made after tracing is configured (`fetch_prompt`, `create_prompt`, `promote_prompt`, datasets, experiments) now send the Keycloak token when the `OPIK_KEYCLOAK_*` client is configured. Before, ol-llm registered the SDK's auth hook only with tracing on, so these calls reached Opik without credentials and got a 401. Tracing off still opens no root trace and exports no spans, and `get_prompt` still returns the embedded default without asking Opik.

### Security

- `check_chat_model`, the embedding catalog check, and the models and embedders `complete` and `embed` (and their `_sync` forms) run on are no longer kept for every use case for the life of the process. An application that builds a use case from a request's model id kept one of each, provider clients included, for every distinct id a caller sent. A use case whose models the catalog all lists stays cached, up to 1,024 of them (`CHECKED_USE_CASES_MAX`). A use case naming any model the catalog does not list, and an embedder asked for a `dimensions` other than the catalog's, is kept only among the 64 most recently used (`UNCATALOGED_USE_CASES_MAX`), so a caller cycling ids can't evict the application's cataloged use cases. Such a use case builds new clients, and its unlisted-model warning is logged again, once it falls out of those 64. An application's own use case on an unlisted model shares those 64; list its model in the catalog to keep it protected. Only a use case's `model` may come from a request: `fallback_models` and `embed`'s `dimensions` must be defined by the application.
- `UseCase` raises `ValueError` when `model` or any of `fallback_models` is longer than 512 characters (`ol_llm.policy.MODEL_ID_MAX_CHARS`), so an application that builds a use case from a request's model id refuses an oversized id before it reaches a cache, a log line or a provider. The error gives the id's length, not the id.
- `UseCase` raises `ValueError` when `fallback_models` names more than 8 models (`ol_llm.policy.MAX_FALLBACK_MODELS`), so a use case built from a request can't make a call build and try an arbitrarily long chain. The error gives the count, not the ids.
- The catalog's unlisted-model and deprecation warnings write the model id with `repr`, so an id carrying a newline can't start a forged log line.

<a id='changelog-0.0.9'></a>
## [0.0.9] - 2026-11-09

### Added

- `ol_llm.run_coroutine(coro)` runs a coroutine on one long-lived event loop per thread that ol-llm owns, so the provider clients built for that loop are reused by every later call in the thread. Sync code that awaits ol-llm should use it in place of `asyncio.run` or `async_to_sync`, which start a loop per call and leave a client behind per call. It refuses to run inside a running loop, and an interrupted coroutine is cancelled and cleaned up before the exception propagates.
- `ol_llm.close_thread_loop()` closes the calling thread's loop and the provider clients built for it. The main thread's loop closes at interpreter exit.

### Changed

- `run_sync`, `complete_sync`, and `embed_sync` run on that per-thread loop instead of PydanticAI's `run_sync`, so a Celery task or Dagster asset reuses one set of provider clients per worker thread.
- On models without mid-conversation system messages (in the catalog: every Bedrock and Google model, and `anthropic:claude-sonnet-5`), every request with a system prompt or instructions now carries a fixed sentence explaining PydanticAI's `<system>` tag in its system field. It goes after any `system_prompt=` text and ahead of the instructions, so for an agent built with `instructions=` alone, as ol-llm's entry points are, it is the first system block. PydanticAI 2.63 added it (pydantic/pydantic-ai#11506) and offers no setting to turn it off. Each cached prompt prefix takes one cache miss after the upgrade, and tests that assert the first system block is the app's prompt need updating. OpenAI, Azure, Mistral and `anthropic:claude-haiku-5-5` requests are unchanged.

### Fixed

- A process forked after ol-llm was used (a Celery prefork worker whose app configured tracing at startup) no longer shares the parent's sockets or waits on its dead threads. The child keeps every parent thread's event loop alive without closing it, drops the parent's provider clients without closing them, builds its own Bedrock client, gets new HTTP clients for the Keycloak token endpoint and the span exporter, and a new Opik SDK client, whose sender threads did not survive the fork. It also starts with fresh module locks (shared models, embedders, settings, tracing) and no in-flight prompt fetches, so a lock or fetch a parent thread held at the fork can't block it. Before, `ol_llm.tracing.flush()` in the child hung waiting on the parent's sender threads.
- Each Bedrock client is built from a boto3 session of its own instead of boto3's default session, which isn't thread-safe, so first calls from several models on different threads no longer race. Threads using one model share one client.

### Security

- Every PydanticAI extra now requires `pydantic-ai-slim>=2.66`, which fixes eight advisories published on 2026-11-04 that affect 2.65 and earlier, among them GHSA-m5vj-8mcw-82qx (high) and GHSA-jp94-mmr3-2pvv.
- The `harness` extra moves to `pydantic-ai-harness` 0.66, which fixes exponential backtracking in the `private_key` secret-redaction pattern (GHSA-mfx5-pc32-32m8). The extra's old `<0.61` cap held consumers on affected releases. 0.66 pins PydanticAI 2.66.

<a id='changelog-0.0.8'></a>
## [0.0.8] - 2026-11-05

### Added

- `ol_llm.traced_chat` traces one chat turn of a call site that keeps its LangChain or LangGraph chat model, under the root trace `run` opens: the §4 attribution fields, the catalog check, and a `model_status:deprecated` or `model_status:uncataloged` tag for a deprecated or unlisted model. It yields LangChain callbacks that record the graph's runs as spans of that trace from any thread or task, retrievers included (Opik's own tracer records no retriever span and drops the runs nested in one). Each chat model span names the use case's model as the catalog's Opik price table names it, with the token usage LangChain reports in the shape Opik prices that provider from (Anthropic's, Bedrock's and Google's own, cache reads and writes included; OpenAI's otherwise), so Opik prices `ChatLiteLLM` calls too. A block left mid-stream (a `break`, an exception, a cancelled task) ends the runs it left open, marked `cut_off` rather than failed, and leaves the caller's Opik context as it found it. `graph` puts the agent graph in Opik's graph view and `metadata` goes on the runs' spans. Without `OL_LLM_TRACE_CONTENT` the spans keep no input or output and an error keeps its type and frames. Needs the new `langchain` extra (#44).
- `ol_llm.check_chat_model(use_case)` runs `build_model`'s catalog checks for a chat model the call site builds itself: a retired or embedding model raises, and an unlisted or deprecated one warns once per process. It returns the `model_status` tags (#44).

<a id='changelog-0.0.7'></a>
## [0.0.7] - 2026-10-29

### Added

- Catalog entries for `anthropic:claude-sonnet-5`, `anthropic:claude-haiku-5-5`, and `bedrock:us.anthropic.claude-haiku-5-5`, so they build without the `model_status:uncataloged` tag and have a price in the spend rules. Claude Haiku 5.5 is priced by prompt length, and the catalog records the price for prompts up to 100,000 tokens. The Haiku 5.5 entries name no Opik model, since the price table of the pinned Opik release (2.2.94) does not list it (#38).
- `ol_llm.evaluation.judge_pairwise` judges a candidate experiment's answers against a reference experiment's, paired by dataset item id on the same dataset version. The judge (`eval-judge`) is asked once with each answer first, and a disagreement between the two orders is a tie. Each verdict is logged on the candidate's experiment item as a `win_rate_vs_baseline` score (1 for a win, 0.5 for a tie, 0 for a loss) with the judge's reasons, and the returned `PairwiseComparison` gives the win rate, counting a tie as half, and the items that were not paired or not judged (#40).
- A gate declaration takes `[pairwise]` with the use case's `criteria` and optional `answer_fields` (default `["output"]`), which judges each answer against the baseline's. It needs `judge_model` and a `minimum` or `maximum` on `win_rate_vs_baseline` (not a `tolerance`), and the report includes the win rate. With no baseline it is `not compared`. The gate exits 2 when the baseline's answers cannot be paired with the run's (a different dataset version, more than one trial per item, or a candidate answer without one of `answer_fields`) or no paired item got a verdict because the candidate's task or the judge raised on each (#40).
- `MetricOutcome` takes `compared` and `detail` (#40).

### Changed

- A gate declaration with a threshold on `win_rate_vs_baseline` and no `[pairwise]` raises `ValueError` (#40).

<a id='changelog-0.0.6'></a>
## [0.0.6] - 2026-10-26

### Added

- `ol_llm.gate` and `ol-llm-eval gate`: run a use case's evaluation gate as its repository declares it in TOML (pinned prompt and dataset versions, judge model, thresholds, advisory or blocking). It exits 2 in either mode when the gate cannot run (a missing dataset, dataset version, or prompt version, a pinned dataset version without every item in the declared golden file, an unreadable golden file, an uncataloged judge) or ran but evaluated nothing (every task raised an `LLMError`, or a thresholded metric scored no item), with what to change. Tolerances are measured from the newest default-branch run on the production prompt version (#30).
- `.github/workflows/evaluation-gate.yml`, a reusable workflow that runs a gate in the caller's uv environment and posts the report on the pull request; `docs/evaluation-gate.md` has a caller to copy (#30).
- `fetch_prompt` fetches one prompt version or label from Opik, text or chat, with no embedded default (#30).
- `Threshold` takes a `tolerance` and a `direction`, required with a tolerance and no bound. `MetricOutcome.status` is `pass`, `fail`, or `not compared` (a tolerance-only threshold with no baseline, which is not a miss), and a tolerance fails a mean that is not a number, as a bound already did. `check_thresholds` and the new `score_report` (which does not raise or log) take the baseline's means as `baselines`; `evaluate` takes extra `experiment_tags` (#30).

### Changed

- `Threshold` raises `ValueError` for a bound or tolerance that is not finite, and so does `ol-llm-eval check --threshold 'x>=nan'`. In 0.0.5 such a threshold failed or passed every mean (#30).

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

- Requires `certifi>=2026.10.19`. Earlier releases bundle emSign roots whose intermediate CA keys are compromised (GHSA-6q48-6c7w-9gw2), so requests and httpx could trust a forged certificate. ol-llm does not import certifi; the floor reaches every app that installs it (#29).
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
