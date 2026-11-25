### Added

- `ol_llm.decide(use_case, state, questions)` and `decide_sync` call a system-one decision
  model: a text or JSON state and named `YesNo`, `Choice` and `Score` questions, answered with
  probabilities in a `Decision`. Requests use the System One API that Microsoft-Decision,
  TypeSafe's Jev, Laya and Strands Decider share, chosen by the model id's provider: `azure:`
  (an Azure Foundry account, `Settings.azure_decision_endpoint` or
  `OL_LLM_AZURE_DECISION_ENDPOINT`, with `Settings.azure_ad_token_provider`), `typesafe:`
  (`TYPESAFE_API_KEY`), or `systemone:` for a self-hosted server (`OL_LLM_SYSTEMONE_BASE_URL`,
  `OL_LLM_SYSTEMONE_API_KEY`). Calls run under the use case's policy and error classes and are
  traced with token usage. Needs no extra.
- The model catalog has a `decision` kind and lists `azure:Microsoft-Decision-1.1`.
  `ol_llm.check_decision_model` checks a decision use case as `decide` does.

### Changed

- `ol_llm.langchain.classify_chat_failure` is now `ol_llm.retries.classify_http_failure`.
