### Added

- `ol_llm.chat_litellm_kwargs(use_case)` returns `ChatLiteLLM` settings that apply the use
  case's timeout and retries: the timeout bounds each read (the gap between chunks on a
  stream), and the provider's OpenAI-SDK client retries before a response starts, never
  after a stream's first chunk.
