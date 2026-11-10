### Added

- `Settings.together_api_key`, so a host that configures with `ol_llm.configure()` can supply Together's key. Unset, the key is still read from `TOGETHER_API_KEY`.

### Fixed

- When every model in a fallback chain fails on a request that continues a suspended response (OpenAI background mode), the caller gets the primary's error. It got the error of the member holding the suspended job, which is tried first.
- On a retry that starts at a fallback model, the request's span names the model that answered instead of the whole chain.
- With `UsageLimits(count_tokens_before_request=True)` on a use case with no fallback models, the token count request gets the use case's timeout, retries, and error classes. It was sent with the caller's timeout in place of the policy's, never retried, and a provider failure reached the caller as PydanticAI's `ModelHTTPError`.
- Together requests go to the base URL of the installed `TogetherProvider` instead of a copy of it.
