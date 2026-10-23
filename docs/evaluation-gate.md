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
better, up for one where lower is better. A threshold with only a `maximum` is lower-is-better;
set `direction = "lower"` or `"higher"` to say so explicitly. Any item left unscored fails its
metric.

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

It fetches the pinned prompt versions and checks the dataset holds the pinned version
(seeding `file` first), runs one experiment tagged `gate:<use_case>` and `change:<change>`,
compares it with the thresholds and the baseline, and writes the report. It exits 0 on a pass
or an advisory miss, 1 on a blocking miss, and 2 when the gate cannot run: a missing dataset,
dataset version, or prompt version, a declaration that does not parse, a suite that does not
import. A gate that cannot run fails in either mode and says what to change.

## Baselines

The baseline is the newest run of the gate made with `--record-baseline`, on the pinned
dataset version, that ran every prompt's `production`-labeled version. The workflow records
one on every run on the default branch outside a pull request (a push or a manual run), so a
use case gets its first baseline from a manual run of the workflow on the default branch.
Until then, and after the dataset pin changes, only the fixed bounds apply and the report says
there is no baseline.

The spec defines the baseline as the last experiment for the production version. Pull
requests also run that version (any model or configuration change does), so taking any such
experiment would measure a change against an unmerged one.

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
      OPIK_CLIENT_ID: ${{ secrets.OPIK_EVALUATION_CLIENT_ID }}
      OPIK_CLIENT_SECRET: ${{ secrets.OPIK_EVALUATION_CLIENT_SECRET }}
```

The `paths` filter is what decides which changes require an evaluation (spec §5.1 item 1).
The job installs the calling project with `uv sync --frozen`, so ol-llm, its provider extras,
and the suite's code come from the caller's lockfile; opik must be installed whole there, not
with the slimming overrides (see `ol_llm.evaluation`). Experiments are recorded with
`environment=ci`. Pull requests from forks get no secrets, so the job fails for them rather
than passing unevaluated.

The job needs Opik reachable from GitHub's runners and a Keycloak client of its own
(spec §1.2 item 3).
