### Added

- The prompt pointer file (`ol_llm.pointers.PromptPointers`): an app's Opik project and, per prompt, the version production runs. `ol-llm-prompts check` refuses a pointer to a version Opik does not have and, with `--environment`, reports a label that is not on its pointed version; `promote` moves the `production` label to the pointed versions after resolving all of them; `export` writes each pointed template for the image. See `docs/prompt-promotion.md`.
- `ol_llm.load_default(name, directory=..., template=...)` reads an exported template as the embedded default for `get_prompt`.
- The reusable workflow `.github/workflows/prompt-promotion.yml`: `check` on a pull request, the drift report on a schedule, and `promote` on a push or manual run, which reads the pointer file from the default branch's head, runs one promotion at a time per pointer file, and takes the promotion client from a GitHub environment so pull request code never holds it. Any other event fails.

### Changed

- An evaluation gate declaration takes its prompt versions from the pointer file: replace the `[prompts]` table with `pointers = "<path from the declaration>"` and `prompts = ["<name>", ...]`, and add the pointer file (its `project` must equal the gate's `app`). A declaration with a `[prompts]` table can no longer run.
