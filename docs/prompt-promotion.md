# Prompt promotion

Prompts are authored in the app's Opik project, where anyone admitted can create versions and
move the `ci` and `qa` labels. The `production` label moves only by pull request
([`llm-opik-lifecycle-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/llm-opik-lifecycle-spec.md)
§2.3): the app's repository holds a pointer file naming the version production runs, a pull
request changes a number and commits that version's text, and merging it moves the label.
Rollback is a revert of the pull request.

## The pointer file

One per app repository:

```toml
app = "dagster-ml"                       # the app's namespace (its Opik project)
exports = "prompts"                      # the embedded defaults, from this file

[prompts]                                # the version production runs, per prompt
feedback-summary = "v4"
feedback-category-proposal = "v2"
"tutor.system" = "v7"                    # quote a name with a dot
```

A `<use_case>.<part>` name must be quoted: TOML reads an unquoted dotted key as a nested
table, and the file is then refused.

Every managed prompt the app fetches has a pointer, whether or not a gate evaluates it. A
prompt's [evaluation gate](evaluation-gate.md) names this file and reads its version from it,
so a pull request that moves a pointer runs the gate on the version merging will promote.

## Moving production

1. Create the version in Opik (the UI or the Playground), and try it under the `ci` or `qa`
   label if you like.
2. Change its number in the pointer file.
3. Run `ol-llm-prompts export prompts.toml` and commit what it writes with the pointer change.
4. Open the pull request. Its diff shows the new prompt text, `check` confirms the committed
   text is the pointed version's, and the evaluation gate runs on that version.
5. Merging moves `production` to it.

The committed text is the embedded default: what `get_prompt` returns when Opik cannot be
reached at cold start. Committing it in the pull request (rather than exporting at image build)
puts the fallback in front of the reviewer and keeps Opik out of every app's build.

## `ol-llm-prompts`

```sh
ol-llm-prompts export prompts.toml                           # write the embedded defaults
ol-llm-prompts check prompts.toml                            # versions exist, exports match
ol-llm-prompts check prompts.toml --environment production   # and the label is where it points
ol-llm-prompts promote prompts.toml                          # move production to the pointers
```

`export` writes `<exports>/<name>.prompt.yaml` for each prompt: its name, the app, the pointed
version, the template's type (`mustache` or `jinja2`), its template hash, and the template (or
the messages, for a chat prompt). A multi-line template is written as a YAML literal block, so
the pull request shows the prompt as it reads:

```yaml
name: feedback-summary
app: dagster-ml
version: v4
type: mustache
template_hash: 8e31a7bc6baa57c09d90bd7ee44ca9801432b95dae933ebfbcb23130ce14c223
template: |-
  You are analyzing feedback about MIT Learn, ...

  <learner_text>
  {{conversation_text}}
  </learner_text>
```

Text a literal block cannot hold exactly (a line ending in spaces, U+0085) is written
double-quoted and escaped instead, so no line of an export ends in whitespace and hooks such as
`trailing-whitespace` leave it alone. Every export reads back as exactly the version's text, or
`export` refuses to write it. An export is generated, so exclude `\.prompt\.yaml$` from YAML
formatters (yamlfmt, prettier): `export` rewrites whatever they change, and `check` fails if a
reformat ever changes the text. detect-secrets flags the `template_hash` line as a high-entropy
string; exclude it with `--exclude-lines '^\s*template_hash:\s*["'']?[0-9a-f]{64}["'']?\s*$'`
(detect-secrets reads YAML values back quoted).

`export` also removes its own earlier export of a prompt the file no longer names and leaves
every other file. `exports` must be a directory beneath the pointer file's. The app reads an
export with `load_default`, which works offline, and passes it to `get_prompt`:

```python
PROMPTS_DIR = Path(__file__).parent / "prompts"

SUMMARY_PROMPT = ol_llm.load_default(
    "feedback-summary", directory=PROMPTS_DIR, template=ol_llm.TextTemplate
)
```

The default hashes the same as the version it came from, so a batch run that fell back to it
gets the same idempotency keys as one that fetched the version.

`check` exits 2 when a pointed prompt or version does not exist in Opik, or when a committed
export is missing, is for another version, has text that is not its version's (edited by
hand), or is for a prompt the file no longer names. The message says to run `export` and
commit. With `--environment` it also prints each prompt whose label is on another version, or
on none, and exits 1 if there is one: that is the drift report (spec §1.2 item 5), for a label
someone moved in the UI.

`promote` fetches every pointed version before it moves any label, so a pointer to a missing
version moves nothing. It leaves a label that is already in place and keeps each version's
other labels. If it stops partway (Opik going away, say), running it again finishes the rest.
It exits 2 when the label is not a registered environment in the workspace; Opik seeds only
`development`, `staging`, and `production`.

## The workflow

```yaml
name: prompt promotion

on:
  pull_request:
    paths: ["dg_projects/ml/prompts.toml"]
  push:
    branches: [main]
    paths: ["dg_projects/ml/prompts.toml"]
  schedule:
    - cron: "17 13 * * 1-5"
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
      OPIK_KEYCLOAK_CLIENT_ID: ${{ secrets.OPIK_EVALUATION_CLIENT_ID }}
      OPIK_KEYCLOAK_CLIENT_SECRET: ${{ secrets.OPIK_EVALUATION_CLIENT_SECRET }}
      OPIK_PROMOTION_CLIENT_ID: ${{ secrets.OPIK_PROMOTION_CLIENT_ID }}
      OPIK_PROMOTION_CLIENT_SECRET: ${{ secrets.OPIK_PROMOTION_CLIENT_SECRET }}
```

Each event does one thing:

- `pull_request` runs `check`.
- `schedule` runs `check --environment production`, the drift report, which fails when a label
  is not where the pointer file puts it.
- `push` and `workflow_dispatch` run `promote`. It reads the pointer file from the default
  branch's head as it is when the job starts, not from the commit that triggered it, and runs
  one promotion per repository and pointer file at a time, so two merges close together cannot
  finish out of order and move the label back. A manual run restores a label someone moved in
  the UI.
- Any other event (e.g. `merge_group`) fails the job rather than promoting or passing
  unchecked.

The check and the drift report read Opik with `OPIK_KEYCLOAK_CLIENT_*`, the client the
evaluation gate already uses on pull requests. They never install or run the pull request's
code: the job checks out ol-llm at the reusable workflow's own commit (`job.workflow_sha`),
installs it from ol-llm's lockfile, and reads only the pointer file and the exports from the
pull request. A compromised dependency in a pull request (a Renovate bump, say) therefore never
runs next to that secret.

Promotion runs in the GitHub environment `prompt-promotion` (the `promotion-environment` input)
and takes `OPIK_PROMOTION_CLIENT_*` from that environment's secrets, so no pull request job
holds the client meant to move `production`. Create the environment with its deployment
branches limited to the default branch, and set the two secrets on it rather than on the
repository. The caller still passes them by name; GitHub substitutes the environment's values
in the job that names the environment.

Opik admits any authenticated client as a workspace admin, so the read client could move a
label too, and the evaluation gate has to run the pull request's suite code with it. Someone
who can push a branch to the repository can also edit the calling workflow. The drift report
is what catches a label moved any of those ways.

Every prompt the app fetches should have a gate whose `paths` include the pointer file, so a
pull request that moves a pointer also evaluates the new version. Nothing here enforces that,
and promotion does not wait on the gate's result.
