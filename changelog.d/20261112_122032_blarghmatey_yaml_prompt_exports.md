### Changed

- `ol-llm-prompts export` writes `<exports>/<name>.prompt.yaml` instead of `<name>.json`, so a pull request that moves a pointer shows the new prompt as readable text: a multi-line template is a YAML literal block. Text a literal block cannot hold exactly is written double-quoted and escaped, so no line of an export ends in whitespace, and every export reads back as exactly the version's text or `export` refuses to write it. `load_default` and `check` read the new files. Re-export after upgrading. PyYAML is now a dependency.
- The pointer file names its app with `app` instead of `project`, matching the gate declaration's `app`.
- `get_prompt`, `fetch_prompt`, `create_prompt` and `promote_prompt` take `app=` instead of `project_name=`.
- `ManagedPrompt.source` is `"registry"` for a fetched prompt (was `"opik"`), and `ManagedPrompt.opik_prompt` is renamed `registry_ref`. Code that compares `source` with `"opik"` has to change.
