### Added

- `ol_llm.evaluation.judge_pairwise` judges a candidate experiment's answers against a reference experiment's, paired by dataset item id on the same dataset version. The judge (`eval-judge`) is asked once with each answer first, and a disagreement between the two orders is a tie. Each verdict is logged on the candidate's experiment item as a `win_rate_vs_baseline` score (1 for a win, 0.5 for a tie, 0 for a loss) with the judge's reasons, and the returned `PairwiseComparison` gives the win rate, counting a tie as half, and the items that were not paired or not judged (#40).
- A gate declaration takes `[pairwise]` with the use case's `criteria`, which judges each answer against the baseline's; a threshold on `win_rate_vs_baseline` bounds the win rate and the report includes it. With no baseline it is `not compared` (#40).
- `MetricOutcome` takes `compared` and `detail` (#40).
