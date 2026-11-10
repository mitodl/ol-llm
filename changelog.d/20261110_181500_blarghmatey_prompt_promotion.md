### Added

- The prompt pointer file (`ol_llm.pointers.PromptPointers`): an app's Opik project, the directory of its committed embedded defaults, and, per prompt, the version production runs. `ol-llm-prompts export` writes each pointed version's text into that directory for the pull request to commit; `check` refuses a pointer to a version Opik does not have or a committed export that is not its pointed version and, with `--environment`, reports a label that is not on its pointed version; `promote` moves the `production` label to the pointed versions after resolving all of them. See `docs/prompt-promotion.md`. (#64)
- `ol_llm.load_default(name, directory=..., template=...)` reads a committed export as the embedded default for `get_prompt`. (#64)
- The reusable workflow `.github/workflows/prompt-promotion.yml`: `check` on a pull request, the drift report on a schedule, and `promote` on a push or manual run, which reads the pointer file from the default branch's head, runs one promotion at a time per pointer file, and takes the promotion client from a GitHub environment so no pull request job holds it. The check runs ol-llm at the workflow's own commit and reads only the pointer file and exports from the pull request, so the pull request's code never runs next to the read client. Any other event fails. (#64)

### Changed

- An evaluation gate declaration takes its prompt versions from the pointer file: replace the `[prompts]` table with `pointers = "<path from the declaration>"` and `prompts = ["<name>", ...]`, and add the pointer file (its `project` must equal the gate's `app`). A declaration with a `[prompts]` table can no longer run. (#64)
