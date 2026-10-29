# Evaluation gates

A use case whose output a learner or instructor sees runs an evaluation on every pull request
that changes its prompt version, model, fallback list, or retrieval and tool configuration
([`llm-opik-lifecycle-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/llm-opik-lifecycle-spec.md)
§5.1). Each one declares a gate in its repository; `ol-llm-eval gate` runs it, and the reusable
workflow [`evaluation-gate.yml`](../.github/workflows/evaluation-gate.yml) runs that on pull
requests and posts the result.

## The declaration

A TOML file next to the prompt, owned by the app team:

```toml
app = "dagster-ml"                       # the app's Opik project
use_case = "feedback-summary"
mode = "advisory"                        # every gate starts advisory
suite = "feedback_summary_eval:suite"    # module:attribute
judge_model = "bedrock:us.anthropic.claude-sonnet-5"

[prompts]                                # the version under test, per prompt
feedback-summary = "v4"

[dataset]
name = "feedback-summary-golden"
version = "v2"                           # pinned; changing it is a reviewed change
file = "feedback_summary_golden.json"    # optional; seeded before each run

[[thresholds]]                           # a deterministic validator passes on every item
metric = "placeholders"
minimum = 1.0

[[thresholds]]                           # a judged metric may drop 0.05 below the baseline
metric = "answer_relevance_metric"
tolerance = 0.05
direction = "higher"                     # required with a tolerance and no bound

[[thresholds]]                           # lower is better, so a maximum and a rise
metric = "hallucination_metric"
maximum = 0.3
tolerance = 0.05
```

`[prompts]` holds the version production runs. A pull request that promotes a prompt changes
that number, so its gate runs the new version; a model or configuration change runs the
production version unchanged.

A threshold takes a `minimum`, a `maximum`, a `tolerance`, or a combination. `tolerance` is
measured from the baseline in the direction that is worse: down for a metric where higher is
better, up for one where lower is better. A threshold with a `minimum` is higher-is-better and
one with only a `maximum` is lower-is-better; a tolerance with no bound must say which with
`direction = "higher"` or `"lower"`, since guessing wrong would pass a regression. Any item
left unscored fails its metric, and so does a mean that is not a number.

Until there is a baseline, a tolerance-only threshold is reported as `not compared`, which is
neither a pass nor a miss.

`[prompts]` duplicates the pointer file the promotion job reads (lifecycle spec §2.3 item 2),
and ol-llm does not define that file's format yet. Until it does, a pull request that promotes
a prompt changes both, and nothing checks that they agree.

## The suite

`suite` names a callable that takes a `GateContext` and returns a `Suite`. The module is
imported with the declaration's directory first on `sys.path`, so it can sit beside it.

```python
from ol_llm import UseCase, complete_sync
from ol_llm.evaluation import ValidatorMetric
from ol_llm.gate import GateContext, Suite
from opik.evaluation.metrics import AnswerRelevance

SUMMARY = UseCase(
    app="dagster-ml",
    name="feedback-summary",
    profile="batch",
    model="bedrock:global.anthropic.claude-haiku-4-5-20251001-v1:0",
)


def placeholders_kept(text: str, output: str) -> bool: ...


def suite(context: GateContext) -> Suite:
    prompt = context.prompts["feedback-summary"]

    def task(item: dict) -> dict:
        result = complete_sync(SUMMARY, prompt.format(text=item["text"]), prompt=prompt)
        return {"input": item["text"], "output": result.output}

    return Suite(
        task=task,
        metrics=[
            ValidatorMetric("placeholders", placeholders_kept),
            AnswerRelevance(model=context.require_judge()),
        ],
        experiment_config={"model": SUMMARY.model},
    )
```

`context.require_judge()` returns the declared judge, built with `judge_model`, so its calls
are routed, traced, and costed under `eval-judge` like any other call. It should not be the
model under test.

## Running it

```sh
ol-llm-eval gate evaluation/feedback-summary.gate.toml --change pr-123 --report report.md
```

It fetches the pinned prompt versions, seeds `file` when there is one, and checks the dataset
has the pinned version and that the version holds every item in `file`. It then runs one
experiment tagged `gate:<use_case>` and `change:<change>`, compares it with the thresholds and
the baseline, and writes the report. It exits 0 on a pass or an advisory miss, 1 on a blocking
miss, and 2 when the gate cannot run, in either mode, saying what to change: a missing dataset,
dataset version, or prompt version, a pinned dataset version without every item in `file`, a
golden file that cannot be read or seeded, a judge model the catalog does not list, a
declaration that does not parse, a suite that does not import. It also exits 2 when the
experiment ran but evaluated nothing: the dataset version has no items, the task raised an
`LLMError` on every item (credentials, model access, a gateway route), or a thresholded metric
scored no item.

## Baselines

The baseline is the newest run of the gate made with `--record-baseline`, on the pinned
dataset version, that ran every prompt's `production`-labeled version. The workflow records
one on a push, a manual run, or a scheduled run on the default branch, so a use case gets its
first baseline from a manual run of the workflow on the default branch. Until then, and after
the dataset pin changes, only the fixed bounds apply and the report says there is no
baseline.

The spec defines the baseline as the last experiment for the production version. Pull
requests also run that version (any model or configuration change does), so taking any such
experiment would measure a change against an unmerged one.

## Pairwise comparison with the baseline

Thresholds bound each metric's mean, so they can say an answer got worse on a scale, but not
whether the judge would rather have the baseline's answer to the same question. A gate gets
that by declaring `[pairwise]` with the criteria for its use case, and a threshold on
`win_rate_vs_baseline`, added to a declaration like the one above:

```toml
[pairwise]
criteria = """
Choose the response that best serves the learner who asked. Prefer responses that are
accurate and supported by the retrieval context, that address what was asked including
qualifiers such as "basic" or "free", and that are clear and appropriately concise.
"""
answer_fields = ["output"]               # the task output fields the judge compares

[[thresholds]]
metric = "win_rate_vs_baseline"
minimum = 0.45
```

Once the experiment has run, `ol_llm.evaluation.judge_pairwise` pairs its items with the
baseline's by dataset item id (the baseline is on the same pinned dataset version, so every
item should pair). For each pair, the declared `judge_model` sees the dataset item and the two
answers' `answer_fields`, and says which better meets the criteria, or that neither does. It
is asked twice, once with each answer first. When the two orders agree that is the verdict,
and when they do not the item is a tie, so a judge that favors whichever answer comes first
cannot decide an item. The judge's calls go through ol-llm under the `eval-judge` use case, so
each is a trace in the app's project like any other judge call.

Each verdict is logged on the run's experiment item as a `win_rate_vs_baseline` score: 1 for a
win, 0.5 for a tie, 0 for a loss, with the judge's reason for each order and the verdict as the
score's category. A tie counts as half a win, so the win rate is the mean of those scores, the
same average Opik shows on the experiment, and 0.5 means the change did as well as the
baseline. The report also counts wins, ties, and losses, and how many ties came from the two
orders disagreeing. `minimum = 0.45` lets a change lose slightly more items than it wins, since
a judge is noisy.

An item in only one of the two experiments, or one where the baseline has no answer (its task
raised an `LLMError`, or its output lacks one of `answer_fields`), has no pair and does not
count either way; the report says how many there were and why. When no item has a pair, e.g.
right after a change to `answer_fields`, the win rate is `not compared` until the next
default-branch run records a baseline with those fields. An item the run's task raised on, or
the judge raised on, counts as unscored, which fails the threshold as it does for any other
metric, and a run where the judge raised on every pair cannot run (exit 2). A pairwise
comparison takes one trial per item, and needs every one of the run's answers to have the
`answer_fields`; the gate cannot run if either is not so.

With no baseline, `win_rate_vs_baseline` is `not compared`, as a tolerance-only threshold is,
and the judge is not called. The threshold takes a `minimum` or `maximum` and no `tolerance`,
since the win rate is already relative to the baseline. The criteria above are a start for
the chat bots, from learn-ai's `rag_evaluation --pairwise`; each use case should write its own.

## The workflow

```yaml
name: evaluation gate

on:
  pull_request:
    paths:
      - "dg_projects/ml/ml/lib/summarize.py"
      - "dg_projects/ml/evaluation/**"
  push:
    branches: [main]
    paths:
      - "dg_projects/ml/ml/lib/summarize.py"
      - "dg_projects/ml/evaluation/**"
  workflow_dispatch:

permissions: {}

jobs:
  feedback-summary:
    uses: mitodl/ol-llm/.github/workflows/evaluation-gate.yml@<commit> # ol-llm-v<version>
    permissions:
      contents: read
      pull-requests: write
      id-token: write
    with:
      gate: evaluation/feedback-summary.gate.toml
      working-directory: dg_projects/ml
      opik-url: ${{ vars.OPIK_URL }}
      keycloak-token-url: ${{ vars.OPIK_KEYCLOAK_TOKEN_URL }}
      aws-role-arn: ${{ vars.EVALUATION_AWS_ROLE_ARN }}
    secrets:
      OPIK_KEYCLOAK_CLIENT_ID: ${{ secrets.OPIK_EVALUATION_CLIENT_ID }}
      OPIK_KEYCLOAK_CLIENT_SECRET: ${{ secrets.OPIK_EVALUATION_CLIENT_SECRET }}
```

The `paths` filter is what decides which changes require an evaluation (spec §5.1 item 1).
The job installs the calling project with `uv sync --frozen`, so ol-llm, its provider extras,
and the suite's code come from the caller's lockfile; opik must be installed whole there, not
with the slimming overrides (see `ol_llm.evaluation`). Experiments are recorded with
`environment=ci`. Pull requests from forks get no secrets, so the job fails for them rather
than passing unevaluated. A job that fails before the gate reports (a failed install, missing
credentials) replaces the pull request comment with a link to its log, so an earlier run's
result does not stay up; a run cancelled by a newer push posts nothing.

The job needs Opik reachable from GitHub's runners and a Keycloak client of its own
(spec §1.2 item 3).
