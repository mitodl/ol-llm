# Error contract

How every model call is bounded, retried, and failed. Decided 2026-09-11 and specified as
types on 2026-10-01; the reasoning and the measurements behind each rule are in
[`shared-llm-library-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/shared-llm-library-spec.md)
§3 and §6. Timeouts and retries are mandatory; fallback is opt-in per use case.

## Use cases and profiles

Every call site declares a use case once: the app, a name, a profile, and the model id in
`provider:model` form. The profile sets the defaults.

| Profile | For | `TIMEOUT_SECONDS` | `MAX_RETRIES` | `MAX_BACKOFF_SECONDS` |
| --- | --- | --- | --- | --- |
| `request` | a user is waiting | 60 | 2 | 10, total across the call |
| `batch` | pipelines, background jobs | 300 | 5 | 300, per wait |

The values resolve in this order, later wins:

1. the profile default;
2. a value on the use case, which must come with a written `policy_reason`;
3. `OL_LLM_<PROFILE>_<SETTING>`, e.g. `OL_LLM_BATCH_TIMEOUT_SECONDS`;
4. `OL_LLM_<USE_CASE>_<SETTING>`, with the name upper-cased and anything outside `A-Z0-9`
   replaced by `_` (`course-translation` reads `OL_LLM_COURSE_TRANSLATION_MAX_RETRIES`).

A use case may not be named `request` or `batch`. A value that does not parse, or is out of
range, fails when the model is built. The resolved values go on the trace as
`policy_timeout_seconds`, `policy_max_retries`, and `policy_max_backoff_seconds`.

`TIMEOUT_SECONDS` bounds one attempt; on a stream it bounds the gap between chunks. The worst
case for one call is `(MAX_RETRIES + 1) × TIMEOUT_SECONDS` plus backoff, and that again for
each fallback model, so worker time limits (Celery, arq, Dagster) have to sit above it.

## One retry layer

The library is the only thing that retries. Every provider SDK's own retries are off (OpenAI,
Azure OpenAI, Anthropic, and Bedrock retry by default; Google and Mistral make one attempt).
A gateway route used by the library must also be configured with no retries of its own.

Failures are classified by type and status, never by message text:

| Raised by the provider path | Error | Retried | Falls back |
| --- | --- | --- | --- |
| HTTP 429 | `LLMRateLimited` (carries `retry_after`) | yes | yes |
| HTTP 408, 500, 502, 503, 504, 529 | `LLMUnavailable` | yes | yes |
| a timeout, a dropped connection, a broken response | `LLMUnavailable` | yes | yes |
| any other HTTP status; any other transport error | `LLMRejected` (carries `status_code`, none for a transport error) | no | no |
| the provider's content filter | `LLMContentFiltered` | no | no |
| invalid output once output retries are spent | `LLMInvalidOutput` | no | no |
| a run's usage limit | `LLMBudgetExceeded` | no | no |

Every error carries the use case name, the model id, whether it is retryable, and the
provider's exception as its cause. Provider messages never reach users; request-time surfaces
map the error class to their own text.

Backoff is exponential with jitter, starting at 0.5 s. On the request profile the backoff
budget is a total, and a `Retry-After` longer than what remains ends the call at once with
`LLMRateLimited`. On the batch profile each wait is capped, and a longer `Retry-After` waits
the cap and tries again. A stream is retried while it opens; a failure after the first chunk
reaches the caller as `LLMUnavailable`.

Invalid output is retried on the model that produced it, `output_retries` times (default 1).
The retry falls back only if that model is rate limited or unavailable, and then only to the
models after it in the chain. A streamed run (`run_stream`) is the exception: it keeps the
first output that matches the output type, so invalid output raises `LLMInvalidOutput` with
no retry, after the caller may already have read part of it.

## Fallback

A use case that lists `fallback_models` tries them in order once the model before has spent
its own retries on a rate limit or an outage. A rejected request does not fall back. When every
model fails, the caller gets the primary model's error (on a retry of invalid output, the
error of the model that produced it), with the whole group of failures as its cause. A model
whose gateway route already falls back takes no library fallback.

## Degraded results

A result produced by a fallback model (`fallback_model`), from an embedded prompt default
because Opik was unreachable (`embedded_prompt`), or knowingly incomplete (`partial_output`)
is marked degraded, on the returned value and on the trace as `degraded` and
`degraded_reason`.

## Batch rows

A row failure records its class and does not fail the chunk. `LLMRejected` opens the circuit at
once, since retrying a configuration bug burns budget. `LLMRateLimited` and `LLMUnavailable`
count toward the circuit's threshold. The other errors fail only their row.

## Python status

Implemented: use cases, policy resolution, the error classes, classification of model
requests, the retry wrapper, fallback composition, the content-filter and output-retry
settings on agents, and the entry points `run`, `run_sync`, `run_stream`, `complete`, and
`complete_sync`.

Not yet: the batch helpers. A bare `agent.run` outside the entry points raises
PydanticAI's own `ContentFilterError`, `UnexpectedModelBehavior`, and `UsageLimitExceeded`
rather than `LLMContentFiltered`, `LLMInvalidOutput`, and `LLMBudgetExceeded`, does not apply
`usage_limits`, does not pass the user id to the provider, and never sets `degraded`.
