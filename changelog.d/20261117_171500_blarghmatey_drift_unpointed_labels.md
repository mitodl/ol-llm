### Changed

- `ol-llm-prompts check --environment` also reports a prompt in the app's namespace that carries the label while no pointer names it (`<name>: production is on v2, and no pointer names it`) and exits 1 for it, as for a drifted pointer. Only the promotion job is meant to move `production`, so such a label was moved some other way. An app whose namespace holds labeled prompts it does not point at yet (e.g. ones it still fetches outside ol-llm) sees its scheduled drift report fail until it adds pointers for them.
- `PromptRegistry` has a new abstract method, `prompt_names(app=...)`, which lists the prompts in an app's namespace. A registry registered through the `ol_llm.prompt_registries` entry points has to implement it.
