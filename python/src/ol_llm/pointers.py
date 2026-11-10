"""The prompt pointer file and the jobs that act on it (lifecycle spec §2.3).

Prompts are authored in Opik. Each app repository holds one pointer file naming,
for every managed prompt, the version production runs::

    project = "dagster-ml"              # the app's Opik project

    [prompts]
    feedback-summary = "v4"

Changing production is a pull request that changes a number. Three jobs read the
file, through ``ol-llm-prompts``:

- ``check`` on the pull request: every pointed version exists. With
  ``--environment`` it also reports each label that is not on its pointed version,
  which is the drift report (spec §1.2 item 5).
- ``promote`` on merge: moves the ``production`` label to each pointed version.
  It resolves every pointer before moving any label, so a pointer to a missing
  version moves nothing. Promotion jobs for a repository run one at a time and read
  the file from the default branch's head (the reusable workflow does both), so two
  merges close together cannot finish out of order and move a label back.
- ``export`` at build: writes each pointed version's template into the image, where
  ``load_default`` reads it as the embedded default ``get_prompt`` falls back to.

An evaluation gate names its prompts and reads their versions from this file
(``ol_llm.gate``), so one file says which version is under test and which version
merging promotes.
"""

import argparse
import json
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import pydantic
from opik.exceptions import EnvironmentNotFoundError

from ol_llm.prompts import (
    ChatTemplate,
    ManagedPrompt,
    TemplateT,
    TextTemplate,
    fetch_prompt,
    promote_prompt,
)

PRODUCTION = "production"
"""The label the pointer file controls."""

POINTER_ERROR_STATUS = 2
"""What ``ol-llm-prompts`` exits with when the pointer file cannot be acted on."""

DRIFT_STATUS = 1
"""What ``ol-llm-prompts check --environment`` exits with when a label drifted."""

VersionName = Annotated[str, pydantic.StringConstraints(pattern=r"^v[1-9][0-9]*$")]
"""An Opik prompt or dataset version, e.g. ``v3``."""

PromptName = Annotated[
    str, pydantic.StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
]
"""A prompt name, ``<use_case>`` or ``<use_case>.<part>`` (spec §3 item 11). It is
also the exported file's name, so it cannot hold a path separator."""


class PointerError(Exception):
    """The pointer file cannot be acted on; the message says what to change."""


class PromptPointers(pydantic.BaseModel):
    """An app's pointer file: the version production runs, per prompt.

    :param project: the app's Opik project, which holds the prompts.
    :param prompts: prompt name to the version production runs.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    project: str
    prompts: Annotated[dict[PromptName, VersionName], pydantic.Field(min_length=1)]

    @classmethod
    def load(cls, path: Path) -> "PromptPointers":
        """Read a pointer file.

        :param path: the file.
        :returns: the pointers.
        :rtype: PromptPointers
        :raises PointerError: if the file cannot be read or is not a pointer file.
        """
        try:
            return cls.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
        except (OSError, tomllib.TOMLDecodeError, pydantic.ValidationError) as exc:
            msg = f"{path} is not a prompt pointer file: {exc}"
            raise PointerError(msg) from exc


def resolve(pointers: PromptPointers) -> dict[str, ManagedPrompt[Any]]:
    """Fetch every pointed version from Opik.

    :param pointers: the pointer file.
    :returns: each prompt at its pointed version.
    :rtype: dict[str, ManagedPrompt]
    :raises PointerError: naming every pointer whose prompt or version Opik does
        not have.
    """
    found: dict[str, ManagedPrompt[Any]] = {}
    missing: list[str] = []
    for name, version in pointers.prompts.items():
        prompt = fetch_prompt(name, project_name=pointers.project, version=version)
        if prompt is None:
            missing.append(f"{name} {version}")
        else:
            found[name] = prompt
    if missing:
        msg = (
            f"Opik project {pointers.project!r} has no {', '.join(missing)}. Point at "
            "a version that exists in Opik."
        )
        raise PointerError(msg)
    return found


@dataclass(frozen=True, kw_only=True)
class LabelState:
    """Where a label is, against where the pointer file puts it.

    :param name: the prompt.
    :param pointed: the version the pointer file names.
    :param labeled: the version carrying the label, or ``None`` when none does.
    """

    name: str
    pointed: str
    labeled: str | None

    @property
    def drifted(self) -> bool:
        """Whether the label is anywhere but the pointed version."""
        return self.labeled != self.pointed


def label_states(
    pointers: PromptPointers, *, environment: str = PRODUCTION
) -> list[LabelState]:
    """Compare each prompt's label with its pointer.

    :param pointers: the pointer file.
    :param environment: the label to compare.
    :returns: one state per prompt, in the file's order.
    :rtype: list[LabelState]
    """
    states = []
    for name, version in pointers.prompts.items():
        labeled = fetch_prompt(
            name, project_name=pointers.project, environment=environment
        )
        states.append(
            LabelState(
                name=name,
                pointed=version,
                labeled=None if labeled is None else labeled.version,
            )
        )
    return states


def promote(
    pointers: PromptPointers, *, environment: str = PRODUCTION
) -> list[LabelState]:
    """Move a label to every pointed version that does not carry it yet.

    Every pointer is resolved first, so a pointer to a missing version moves no
    label. A failure partway (e.g. Opik going away) can leave some labels moved;
    running it again finishes the rest, since a label already in place is left.

    :param pointers: the pointer file, read from the default branch's head.
    :param environment: the label to move.
    :returns: the state of each prompt before the move; ``drifted`` ones moved.
    :rtype: list[LabelState]
    :raises PointerError: if a pointed version does not exist, or the label is
        not a registered environment in the workspace.
    """
    resolve(pointers)
    states = label_states(pointers, environment=environment)
    for state in states:
        if not state.drifted:
            continue
        try:
            promote_prompt(
                state.name,
                version=state.pointed,
                environment=environment,
                project_name=pointers.project,
            )
        except EnvironmentNotFoundError as exc:
            msg = (
                f"{environment!r} is not a registered environment in this Opik "
                "workspace; create it once with Opik's create_environment"
            )
            raise PointerError(msg) from exc
    return states


def export(pointers: PromptPointers, directory: Path) -> list[Path]:
    """Write each pointed version's template to ``<directory>/<name>.json``.

    Run at image build, so the embedded default is the reviewed version.

    :param pointers: the pointer file.
    :param directory: where to write; created if missing.
    :returns: the files written.
    :rtype: list[Path]
    :raises PointerError: if a pointed version does not exist.
    """
    prompts = resolve(pointers)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, prompt in prompts.items():
        template = prompt.template
        content: dict[str, Any] = (
            {"messages": [dict(m) for m in template.messages]}
            if isinstance(template, ChatTemplate)
            else {"template": template.template}
        )
        path = directory / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "name": name,
                    "project": pointers.project,
                    "version": prompt.version,
                    "type": template.type,
                    "template_hash": template.template_hash,
                    **content,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def load_default(name: str, *, directory: Path, template: type[TemplateT]) -> TemplateT:
    """Read the embedded default ``export`` wrote, to pass to ``get_prompt``.

    :param name: the prompt.
    :param directory: where ``export`` wrote it.
    :param template: ``TextTemplate`` or ``ChatTemplate``, the prompt's structure.
    :returns: the template, with the exported version.
    :rtype: TextTemplate | ChatTemplate
    :raises FileNotFoundError: if nothing was exported for the prompt.
    :raises ValueError: if the export is of the other structure.
    """
    exported: Mapping[str, Any] = json.loads(
        (directory / f"{name}.json").read_text(encoding="utf-8")
    )
    chat = "messages" in exported
    if chat != (template is ChatTemplate):
        structure = "chat" if chat else "text"
        msg = f"{name!r} was exported as a {structure} prompt, not {template.__name__}"
        raise ValueError(msg)
    loaded: TextTemplate | ChatTemplate = (
        ChatTemplate(
            messages=exported["messages"],
            type=exported["type"],
            version=exported["version"],
        )
        if chat
        else TextTemplate(
            template=exported["template"],
            type=exported["type"],
            version=exported["version"],
        )
    )
    return cast("TemplateT", loaded)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``ol-llm-prompts`` command.

    ``check`` exits 2 if a pointed version does not exist and, with
    ``--environment``, 1 if a label is not on its pointed version. ``promote``
    moves the label and prints what moved. ``export`` writes the embedded defaults.
    Each exits 2 when the pointer file cannot be read or acted on.

    :param argv: the arguments; ``sys.argv[1:]`` when omitted.
    :returns: the exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        prog="ol-llm-prompts", description="Act on an app's prompt pointer file."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="every pointed version exists")
    check.add_argument("pointers", type=Path, help="the pointer file")
    check.add_argument(
        "--environment", help="also report where this label is against the pointers"
    )
    move = commands.add_parser("promote", help="move a label to the pointed versions")
    move.add_argument("pointers", type=Path, help="the pointer file")
    move.add_argument("--environment", default=PRODUCTION, help="the label to move")
    dump = commands.add_parser("export", help="write the pointed templates")
    dump.add_argument("pointers", type=Path, help="the pointer file")
    dump.add_argument("--output", type=Path, required=True, help="the directory")
    args = parser.parse_args(argv)
    try:
        pointers = PromptPointers.load(args.pointers)
        if args.command == "export":
            for path in export(pointers, args.output):
                sys.stdout.write(f"wrote {path}\n")
            return 0
        if args.command == "promote":
            for state in promote(pointers, environment=args.environment):
                moved = f" (was {state.labeled or 'on no version'})"
                sys.stdout.write(
                    f"{state.name}: {args.environment} "
                    f"{'moved to' if state.drifted else 'already on'} {state.pointed}"
                    f"{moved if state.drifted else ''}\n"
                )
            return 0
        resolve(pointers)
    except PointerError as exc:
        sys.stderr.write(f"{exc}\n")
        return POINTER_ERROR_STATUS
    sys.stdout.write(f"{args.pointers}: every pointed version exists\n")
    if args.environment is None:
        return 0
    drifted = [
        s for s in label_states(pointers, environment=args.environment) if s.drifted
    ]
    for state in drifted:
        sys.stdout.write(
            f"{state.name}: {args.environment} is on {state.labeled or 'no version'}, "
            f"the pointer says {state.pointed}\n"
        )
    return DRIFT_STATUS if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
