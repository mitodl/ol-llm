# Prompt promotion

Prompts are authored in the app's Opik project, where anyone admitted can create versions and
move the `ci` and `qa` labels. The `production` label moves only by pull request
([`llm-opik-lifecycle-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/llm-opik-lifecycle-spec.md)
§2.3): the app's repository holds a pointer file naming the version production runs, a pull
request changes a number, and merging it moves the label. Rollback is a revert of the pointer
change.

## The pointer file

One per app repository:

```toml
project = "dagster-ml"                   # the app's Opik project

[prompts]                                # the version production runs, per prompt
feedback-summary = "v4"
feedback-category-proposal = "v2"
```

Every managed prompt the app fetches has a pointer, whether or not a gate evaluates it. A
prompt's [evaluation gate](evaluation-gate.md) names this file and reads its version from it,
so a pull request that moves a pointer runs the gate on the version merging will promote.

## `ol-llm-prompts`

```sh
ol-llm-prompts check prompts.toml                            # every pointed version exists
ol-llm-prompts check prompts.toml --environment production   # and the label is where it points
ol-llm-prompts promote prompts.toml                          # move production to the pointers
ol-llm-prompts export prompts.toml --output build/prompts    # write the embedded defaults
```

`check` exits 2 when a pointed prompt or version does not exist in Opik. With `--environment`
it also prints each prompt whose label is on another version, or on none, and exits 1 if there
is one: that is the drift report (spec §1.2 item 5), for a label someone moved in the UI.

`promote` fetches every pointed version before it moves any label, so a pointer to a missing
version moves nothing. It leaves a label that is already in place and keeps each version's
other labels. If it stops partway (Opik going away, say), running it again finishes the rest.
It exits 2 when the label is not a registered environment in the workspace; Opik seeds only
`development`, `staging`, and `production`.

`export` writes `<name>.json` for each prompt: the pointed version's template (or messages, for
a chat prompt), its type, version, and template hash. The app passes it to `get_prompt` as the
embedded default:

```python
PROMPTS_DIR = Path(__file__).parent / "prompts"

SUMMARY_PROMPT = ol_llm.load_default(
    "feedback-summary", directory=PROMPTS_DIR, template=ol_llm.TextTemplate
)
```

The default hashes the same as the version it came from, so a batch run that fell back to it
gets the same idempotency keys as one that fetched the version.

A promotion that changes only a pointer moves the label without rebuilding the image, so the
embedded default stays one version behind until the next build. During an Opik outage at
cold start the app then runs the previous production prompt, marked degraded.

## The workflow

```yaml
name: prompt promotion

on:
  pull_request:
    paths: ["dg_projects/ml/prompts.toml"]
  push:
    branches: [main]
    paths: ["dg_projects/ml/prompts.toml"]
  workflow_dispatch:

permissions: {}

jobs:
  pointers:
    uses: mitodl/ol-llm/.github/workflows/prompt-promotion.yml@<commit> # ol-llm-v<version>
    permissions:
      contents: read
    with:
      pointers: prompts.toml
      working-directory: dg_projects/ml
      opik-url: ${{ vars.OPIK_URL }}
      keycloak-token-url: ${{ vars.OPIK_KEYCLOAK_TOKEN_URL }}
    secrets:
      OPIK_KEYCLOAK_CLIENT_ID: ${{ secrets.OPIK_PROMOTION_CLIENT_ID }}
      OPIK_KEYCLOAK_CLIENT_SECRET: ${{ secrets.OPIK_PROMOTION_CLIENT_SECRET }}
```

On a pull request the job runs `check`. On a push or a manual run it runs `promote` with the
pointer file from the default branch's head as it is when the job starts, not from the commit
that triggered it, and runs one promotion per repository and pointer file at a time. Two merges
close together therefore cannot finish out of order and move the label back. A manual run is
how to restore the label after someone moved it in the UI.

The job authenticates with a Keycloak client of its own, the only one meant to move
`production`. Opik cannot enforce that, which is what `check --environment production` is for.
The evaluation gate runs separately, on the same pull request, when its `paths` include the
pointer file.
