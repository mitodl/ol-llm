### Added

- `ol_llm.experiments`: the experiment backend behind `ol_llm.evaluation` and the evaluation gate, chosen by name like memory backends and prompt registries. `OL_LLM_EXPERIMENT_BACKEND` is `opik` by default, or `memory`, which keeps datasets and experiments in the process for tests. `configure_experiment_backend` sets one in code, and the `ol_llm.experiment_backends` entry-point group adds others.
- `ol_llm.evaluation.hallucination`, `answer_relevance`, `context_precision` and `context_recall`: Opik's judge metrics as ol-llm metrics on a `judge_model`, with Opik's score names (`hallucination_metric` and so on), so a suite imports no Opik. Replace `Hallucination(model=judge)` with `hallucination(judge)`.
- `ol_llm.evaluation.Metric` and `Score`: a metric is a name and a `score(**inputs)` returning a `Score` (`name`, `value`, `reason`, `failed`) or a list of them. Subclass `Metric` for an app's own scoring function.

### Changed

- Metrics passed to `evaluate` and in a gate `Suite` are `ol_llm.evaluation.Metric`s, not Opik `BaseMetric`s, and `evaluate` raises `TypeError` for anything else. `ValidatorMetric` is a `Metric` and its `score` returns a `Score`. A metric that reads Opik's `task_span` is no longer supported.
- A metric input missing from the item and the output raises `ol_llm.evaluation.MissingMetricInputError` (a `ValueError`) from `evaluate` and aborts the run, on every backend and for `ValidatorMetric` as for the judge metrics; the judge metrics used to raise Opik's `ScoreMethodMissingArguments`, and a `ValidatorMetric` recorded a failed score on every item. A gate whose metric is missing an input cannot run (`ol-llm-eval gate` exits 2 with the metric and the missing field names).
- `MAX_EXPERIMENT_ITEMS` moved from `ol_llm.evaluation` to `ol_llm.experiments`.
- `evaluate` returns `ol_llm.evaluation.ExperimentResult` (the experiment's id, name, URL, and each item with its output and scores) instead of Opik's `EvaluationResult`, and `check_thresholds`, `score_report` and `judge_pairwise` take one.
- `JudgeModel` is ol-llm's own class instead of an Opik `OpikBaseModel`; pass it to the judge metrics above. Its `generate_provider_response` is now `respond` (and `arespond`).
- `evaluate`, `seed_dataset` and `judge_pairwise` take `app=` instead of `project_name=`, and `ol-llm-eval seed` takes `--app` instead of `--project`.
- `seed_dataset` returns the dataset's latest version name instead of the Opik dataset.
- `ol_llm.gate.find_baseline` no longer takes `dataset_id`.
