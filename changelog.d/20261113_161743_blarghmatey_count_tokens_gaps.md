### Fixed

- A use case with fallback models can count tokens before each request (`UsageLimits(count_tokens_before_request=True)`), which raised `NotImplementedError` before. The count starts where the request will and falls back on rate limits and outages, as a request does; when every model fails, or the chain reaches a fallback that cannot count (Mistral, OpenAI chat completions), the caller gets the primary's error. The model the count starts at (the primary, or on a retry the model being retried) still raises `NotImplementedError` if it cannot count.
- `count_tokens` on an embedder's model is retried and classified under the call policy, as `embed` is, instead of reaching the caller as the provider's raw exception.
- A request that continues a suspended response (OpenAI background mode) and fails on every model no longer has the agent send a second cancel for the job the fallback chain already cancelled.
