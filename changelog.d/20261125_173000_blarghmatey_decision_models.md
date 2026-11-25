### Added

- Decision ("system-one") models: `build_model`, `complete` and `run` build PydanticAI's
  decision models, so an agent's output fields are answered as typed questions with
  probabilities. Providers: `azure-decision:` (Microsoft-Decision on an Azure Foundry
  account, `Settings.azure_decision_endpoint` or `OL_LLM_AZURE_DECISION_ENDPOINT`, with
  `Settings.azure_ad_token_provider`), `system-one:` (any System One server,
  `OL_LLM_SYSTEMONE_BASE_URL` and `OL_LLM_SYSTEMONE_API_KEY`), `typesafe:`
  (`TYPESAFE_API_KEY`), `openai-decisions:` and `google-decisions:`. They run under the use
  case's policy, error classes and tracing, and can fall back to a chat model on an outage.
- `LLMResult.provider_details` holds the final response's provider details: a decision
  model's confidence and probabilities per output field.
- The model catalog has a `decision` kind and lists `azure-decision:Microsoft-Decision-1.1`.

### Changed

- `ol_llm.langchain.classify_chat_failure` is now `ol_llm.retries.classify_http_failure`.
