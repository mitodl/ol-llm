### Added

- `ol_llm.with_call_policy(model, use_case)` wraps a LangChain chat model so its calls retry
  under the use case's policy, with the entry points' backoff and `max_backoff_seconds`
  budget, and a stream only until its first chunk. (#93)
- `ol_llm.chat_litellm_kwargs(use_case)` returns `ChatLiteLLM` settings for the use case's
  timeout, which bounds each read (the gap between chunks on a stream), with the client's own
  retries off. (#93)
