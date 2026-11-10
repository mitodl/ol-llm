### Fixed

- When every model in a fallback chain fails on a request that continues a suspended response (OpenAI background mode), the caller gets the primary's error. It got the error of the member holding the suspended job, which is tried first.
- On a retry that starts at a fallback model, the request's span names the model that answered instead of the whole chain.
- With `UsageLimits(count_tokens_before_request=True)`, the token count request gets the use case's timeout, retries, and error classes. It had none of them, so a provider failure reached the caller as PydanticAI's `ModelHTTPError`.
