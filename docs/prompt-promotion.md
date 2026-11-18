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
on none, and each prompt in the app's Opik project that carries the label while no pointer
names it, and exits 1 if there is either: that is the drift report (spec §1.2 item 5), for a
label someone moved in the UI.

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
  is not where the pointer file puts it, or is on a prompt the pointer file does not name. Retiring
  a prompt therefore means removing its pointer and then its `production` label in the Opik UI
  (`promote` never removes a label), once the app no longer fetches it.
- `push` and `workflow_dispatch` run `promote`. It reads the pointer file from the default
  branch's head as it is when the job starts, not from the commit that triggered it, and runs
  one promotion per repository and pointer file at a time, so two merges close together cannot
  finish out of order and move the label back. A manual run restores a label someone moved in
  the UI on a pointed prompt.
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

Deployed code calls `get_prompt(name, app=..., default=...)` per run. It resolves the label
named by the deployment's environment (`OL_LLM_ENVIRONMENT`: `ci`, `qa`, or `production`). With
the registry disabled (`OPIK_TRACK_DISABLE=true`, i.e. local development) it fetches nothing and
returns the embedded default, unmarked. Each process caches what it fetched for
`PROMPT_CACHE_TTL_SECONDS` (300 s):

- A label move reaches every process within five minutes. Restarting the app is the way to apply
  it sooner. A worker forked from a parent that had already fetched the prompt starts with the
  parent's cache, so restart the whole app, not one worker.
- When a refresh fails, is slow, or finds nothing, the process keeps serving the version it has
  and tries again after another TTL. A label removed by mistake therefore keeps serving in a warm
  process, and every process that starts afterwards raises `LookupError`.
- With nothing cached, `get_prompt` waits `COLD_FETCH_DEADLINE_SECONDS` (2 s). If Opik has not
  answered by then, or the fetch failed in transport, timed out, or got a 5xx, it returns the
  embedded default (the committed export) with `degraded_reasons=("embedded_prompt",)`. A
  successful `run` given that prompt marks its trace `degraded` with that reason (when tracing
  is on) and returns it in the result's `degraded_reasons`.
- A batch run should resolve the label once at the start and use that version throughout, and
  key its rows on the prompt's `template_hash`, not its version (`ol_llm.batch.BatchKeys` does
  both). The export hashes the same as its version, so a run that fell back to it does not redo
  rows.

The export and the label can disagree for a while. The label moves when the promotion job runs;
the export changes when the image built from that merge deploys. Until then a cold start during
an Opik outage runs the previous production text, marked degraded.

## Rolling back

Revert the pull request that moved the pointer. The revert puts back the old number and the old
export, its check runs, and merging it runs `promote`, which moves `production` back. Nothing new
is created in Opik: a label can point at any existing version, so rolling back never needs a
restored copy of an old one. Until the image built from the revert deploys, the export is still
the text being rolled back, so a cold start during an Opik outage runs it.

Rolling back faster than a pull request can merge: move `production` to the old version in the
Opik UI, then revert the pull request as above. Until the revert merges, the default branch still
points at the version you moved away from, and every `promote` run moves the label back to it. That
is not only a manual run (`workflow_dispatch`): any merge that touches the pointer file, e.g. a
teammate's pointer change for another prompt, and any re-run of an earlier promotion job does it
too. Hold all pointer-file merges and re-runs until the revert merges. The scheduled drift report
fails in the meantime, which is expected.

After a move to another prompt registry backend, an old version has a new id there, so rolling
back to a version from before the move is a forward pointer change naming that id, not a revert.

## When Opik is down

- What falls back: a fetch that fails in transport, times out, misses the 2 s deadline, or gets a
  5xx, or finds no Opik configured (`OPIK_URL_OVERRIDE` unset). Running processes keep serving what they cached. A process that starts fetches nothing and
  runs on the embedded default, marked degraded, and every `get_prompt` call in it can wait up to
  2 s until Opik answers, since the fallback is not cached.
- What does not fall back: a 4xx, e.g. from APISIX or Keycloak in front of Opik, raises out of
  `get_prompt`, and a 404 reads as "no such label" and raises `LookupError`. A misconfigured or
  half-up gateway fails cold starts rather than degrading them.
- Do not merge pointer pull requests. Merging is promoting: the promotion job fails, and the new
  export still ships with the next image, so cold starts would fall back to text the label was
  never moved to. If one was merged, run the workflow manually (`workflow_dispatch`) once Opik is
  back. `promote` is idempotent and finishes whatever a failed run left.
- `ol-llm-prompts check` and `promote` exit 1 with a traceback when Opik cannot be reached. 1 is
  also the drift report's status, so read the log of a failed scheduled report before treating it
  as drift. Pull request checks and evaluation gates fail the same way; re-run them when Opik is
  back rather than merging past them.

## Overrides

Data passed into a prompt (a glossary, retrieved context, course facts) is a template variable,
not an override, and does not change which version the prompt is.

A per-request override that replaces the prompt text (learn-ai's staff-only `instructions` field,
for example) is a debugging tool. It never creates a version or moves a label. Lifecycle spec §3
item 7 allows one only if its trace carries `prompt_override=true` and the name and version it
replaced, and if overridden traces are kept out of datasets and online scoring. ol-llm has no way
yet for an app to add that to a run's trace (`Attribution` carries only `user_id`, `thread_id`,
and `run_id`), so an app cannot meet the condition through ol-llm today.

Inside Opik's local runner, an active prompt mask makes a fetch return the masked version instead
of the label's. Masks live only in that runner's context, never in a deployed process.

## After a promotion

A run's trace names the prompt version when the call site passes the prompt to `run`
(`prompt=`); call sites that do not are invisible to these checks. In the app's Opik project,
read the production traces after the label moves:

- New request-time traces name the new version within five minutes of the move. A trace on the
  old version after that comes from a batch run that started before the move, a pinned
  `version=` call, a process whose refresh is failing, or a process running on the export (whose
  version is the old one until the image built from the merge deploys). Only the export is marked
  `degraded` with `embedded_prompt`. A failing refresh keeps serving the cached registry version
  unmarked, so find it by the warning it logs ("Could not refresh prompt ...; serving the cached
  version").
- A trace marked `embedded_prompt` means that process's cold fetch missed the deadline or failed.
  A process that cannot reach Opik at all usually cannot deliver its trace there either, so look
  for the warning `get_prompt` logs ("The prompt registry did not return prompt ...") in the
  app's logs as well.
- Online scoring rules, where an app has them, run on sampled production traces in the Opik
  backend and select production by the `env:production` tag. Compare their scores, and any
  feedback scores, before and after the move. A drop is a reason to roll back.
