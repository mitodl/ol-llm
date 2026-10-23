### Added

- `ol_llm.gate` and `ol-llm-eval gate`: run a use case's evaluation gate as its repository declares it in TOML (pinned prompt and dataset versions, judge model, thresholds, advisory or blocking). A missing dataset, dataset version, or prompt version fails the gate with what to change, in either mode. Tolerances are measured from the newest default-branch run on the production prompt version (#PR).
- `.github/workflows/evaluation-gate.yml`, a reusable workflow that runs a gate in the caller's uv environment and posts the report on the pull request; `docs/evaluation-gate.md` has a caller to copy (#PR).
- `fetch_prompt` fetches one prompt version or label from Opik, text or chat, with no embedded default (#PR).
- `Threshold` takes a `tolerance` and a `direction`, and `check_thresholds` and the new `score_report` (which does not raise or log) take the baseline's means as `baselines`; `evaluate` takes extra `experiment_tags` (#PR).
