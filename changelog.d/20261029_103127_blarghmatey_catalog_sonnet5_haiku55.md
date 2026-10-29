### Added

- Catalog entries for `anthropic:claude-sonnet-5`, `anthropic:claude-haiku-5-5`, and `bedrock:us.anthropic.claude-haiku-5-5`, so they build without the `model_status:uncataloged` tag and have a price in the spend rules. Claude Haiku 5.5 is priced by prompt length, and the catalog records the price for prompts up to 100,000 tokens. The Haiku 5.5 entries name no Opik model, since the price table of the pinned Opik release (2.2.94) does not list it (#38).
