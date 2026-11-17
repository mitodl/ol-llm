# Prompt promotion

Prompts are authored in the app's Opik project, where anyone admitted can create versions and
move the `ci` and `qa` labels. The `production` label moves only by pull request
([`llm-opik-lifecycle-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/llm-opik-lifecycle-spec.md)
§2.3): the app's repository holds a pointer file naming the version production runs, a pull
request changes a number and commits that version's text, and merging it moves the label.
Rollback is a revert of the pull request.

This is also the runbook: [moving production](#moving-production),
[what a running app sees](#what-a-running-app-sees), [rolling back](#rolling-back),
[when Opik is down](#when-opik-is-down), [overrides](#overrides), and
[after a promotion](#after-a-promotion).

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
A label Opik does not have yet is registered as an environment in the workspace first (Opik
seeds only `development`, `staging`, and `production`). Against the read-only `file` registry
it exits 2.

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

## What a running app sees

Deployed code calls `get_prompt(name, app=..., default=...)` per run, which resolves the
label named by the deployment's environment (`ci`, `qa`, `production`). Each process caches
what it fetched for `PROMPT_CACHE_TTL_SECONDS` (300 s):

- A label move reaches every process within five minutes. There is no call that clears the
  cache; restarting the process is the only way to apply a move sooner.
- When a refresh fails, is slow, or finds nothing, the process keeps serving the version it
  has and tries again after another TTL. A label removed by mistake therefore does not break
  a warm process, but every process that starts afterwards raises `LookupError`.
- With nothing cached, `get_prompt` waits `COLD_FETCH_DEADLINE_SECONDS` (2 s). If Opik has not
  answered or cannot be reached, it returns the embedded default (the committed export) with
  `degraded_reasons=("embedded_prompt",)`. A `run` given that prompt marks its trace
  `degraded` with that reason and returns it in the result's `degraded_reasons`.
- A batch run resolves the label once at the start and uses that version throughout. Key
  batch work on the prompt's `template_hash`, not its version: the export hashes the same as
  its version, so a run that fell back to it does not redo work.

The export and the label can disagree for a while. The label moves when the promotion job
runs; the export changes when the image built from that merge deploys. Until then a cold
start during an Opik outage runs the previous production text, marked degraded.

## Rolling back

Revert the pull request that moved the pointer. The revert puts back the old number and the
old export, its check runs, and merging it runs `promote`, which moves `production` back.
Nothing new is created in Opik: a label can point at any existing version, so rolling back
never needs a restored copy of an old one.

Rolling back faster than a pull request can merge: move `production` to the old version in
the Opik UI, then revert the pull request as above. Until the revert merges, the scheduled
drift report fails, which is expected, and a manual run of the workflow (`workflow_dispatch`)
would move the label forward again, so do not run it. Running processes pick up either move
within the cache TTL.

After a move to another prompt registry backend, an old version has a new id there, so
rolling back to a version from before the move is a forward pointer change naming that id,
not a revert.

## When Opik is down

- Running apps keep serving what they cached, and processes that start fetch nothing and run
  on the embedded default, marked degraded (above). Nothing fails because Opik is down.
- Do not promote. The promotion job fails on a merge while Opik is unreachable; once Opik is
  back, run the workflow manually (`workflow_dispatch`). `promote` is idempotent and finishes
  whatever a failed run left.
- Pull request checks and evaluation gates fail, since both read Opik. Re-run them when it is
  back rather than merging past them.
- Do not move labels in the UI to work around it. The drift report would not see the move
  until Opik is reachable again.

## Overrides

Data passed into a prompt (a glossary, retrieved context, course facts) is a template
variable, not an override: the trace still names the version, and the variables are part of
the run's input.

A per-request override that replaces the prompt text (learn-ai's staff-only `instructions`
field, for example) is a debugging tool. It never creates a version or moves a label. Lifecycle
spec §3 item 7 requires its trace to carry `prompt_override=true` and the name and version it
replaced, and keeps overridden traces out of datasets and online scoring. ol-llm does not set
that metadata itself; an app that allows an override adds it.

Inside Opik's local runner, an active prompt mask makes a fetch return the masked version
instead of the label's. Masks live only in that runner's context, never in a deployed process.

## After a promotion

`get_prompt` links the fetched version to the trace it is called in, so each production trace in
the app's Opik project names the prompt version it ran. Read the first production traces after
the label moves:

- They name the new version. A trace still on the old one more than five minutes after the
  move comes from a process whose refresh is failing.
- None is marked degraded with `embedded_prompt` (or names no version, for a call site that
  does not pass the prompt to `run`). If some are, those processes cannot reach Opik.
- Online scoring rules, where an app has them, run on sampled production traces in the Opik
  backend. They select production by the `env:production` tag, since a rule cannot filter on
  the trace's environment field. Compare their scores, and any feedback scores, before and
  after the move. A drop is a reason to roll back.
