"""The prompt pointer file and the jobs that act on it (lifecycle spec §2.3).

Prompts are authored in Opik. Each app repository holds one pointer file naming,
for every managed prompt, the version production runs::

    app = "dagster-ml"                  # the app's namespace in the registry
    exports = "prompts"                 # the committed embedded defaults

    [prompts]
    feedback-summary = "v4"

Changing production is a pull request that changes a number and commits the new
version's text, which ``ol-llm-prompts export`` writes into ``exports``. That text
is what ``load_default`` reads as the embedded default ``get_prompt`` falls back
to, so the fallback is reviewed in the pull request and the image build needs no
Opik. ``ol-llm-prompts`` also runs:

- ``check`` on the pull request: every pointed version exists and the committed
  export for each is that version's text. With ``--environment`` it also reports
  each label that is not on its pointed version, which is the drift report (spec
  §1.2 item 5).
- ``promote`` on merge: moves the ``production`` label to each pointed version.
  It resolves every pointer before moving any label, so a pointer to a missing
  version moves nothing. Promotion jobs for a repository run one at a time and read
  the file from the default branch's head (the reusable workflow does both), so two
  merges close together cannot finish out of order and move a label back.

An evaluation gate names its prompts and reads their versions from this file
(``ol_llm.gate``), so one file says which version is under test and which version
merging promotes.
"""

import argparse
import re
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pydantic
import yaml
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

EXPORT_SUFFIX = ".prompt.yaml"
"""What ``export`` names a prompt's file: ``<name>.prompt.yaml``."""

_NO_WRAP = 1_000_000_000
"""A line width no export reaches, so YAML never folds a long line."""

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

    :param app: the app, whose namespace in the registry (its Opik project) holds
        the prompts.
    :param exports: the directory of committed embedded defaults; ``load``
        resolves it against the pointer file's directory, which it must stay
        beneath.
    :param prompts: prompt name to the version production runs.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    app: str
    exports: Path
    prompts: Annotated[dict[PromptName, VersionName], pydantic.Field(min_length=1)]

    @classmethod
    def load(cls, path: Path) -> "PromptPointers":
        """Read a pointer file.

        :param path: the file.
        :returns: the pointers, with ``exports`` resolved against the file's
            directory.
        :rtype: PromptPointers
        :raises PointerError: if the file cannot be read, is not a pointer file,
            or its ``exports`` is not a directory beneath the file's own.
        """
        try:
            pointers = cls.model_validate(
                tomllib.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, tomllib.TOMLDecodeError, pydantic.ValidationError) as exc:
            msg = f"{path} is not a prompt pointer file: {exc}"
            raise PointerError(msg) from exc
        exports = path.parent / pointers.exports
        # resolve() follows a symlink, so a link out of the tree is refused too.
        base = path.parent.resolve()
        if exports.resolve() == base or not exports.resolve().is_relative_to(base):
            msg = (
                f"{path}: exports must be a directory beneath the pointer file's, "
                f"not {str(pointers.exports)!r}"
            )
            raise PointerError(msg)
        return pointers.model_copy(update={"exports": exports})


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
        prompt = fetch_prompt(name, app=pointers.app, version=version)
        if prompt is None:
            missing.append(f"{name} {version}")
        else:
            found[name] = prompt
    if missing:
        msg = (
            f"The registry has no {', '.join(missing)} for app {pointers.app!r}. "
            "Point at a version that exists there."
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
        labeled = fetch_prompt(name, app=pointers.app, environment=environment)
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
) -> Iterator[LabelState]:
    """Move a label to every pointed version that does not carry it yet.

    Every pointer is resolved before the first state is yielded, so a pointer to
    a missing version moves no label. Each state is yielded once its move is done,
    so a caller can report what moved before a failure partway (e.g. Opik going
    away). Running it again finishes the rest, since a label in place is left.

    :param pointers: the pointer file, read from the default branch's head.
    :param environment: the label to move.
    :returns: the state of each prompt before its move; ``drifted`` ones moved.
    :rtype: Iterator[LabelState]
    :raises PointerError: if a pointed version does not exist, or the label is
        not a registered environment in the workspace.
    """
    resolve(pointers)
    for state in label_states(pointers, environment=environment):
        if state.drifted:
            try:
                promote_prompt(
                    state.name,
                    version=state.pointed,
                    environment=environment,
                    app=pointers.app,
                )
            except EnvironmentNotFoundError as exc:
                msg = (
                    f"{environment!r} is not a registered environment in this Opik "
                    "workspace; create it once with Opik's create_environment"
                )
                raise PointerError(msg) from exc
        yield state


def export(pointers: PromptPointers) -> list[Path]:
    """Write each pointed version to ``<exports>/<name>.prompt.yaml``.

    The person moving a pointer runs it and commits the result with the pointer
    change, so the embedded default is reviewed text. An earlier export for a prompt
    the file no longer names is removed; other files in the directory are left.

    :param pointers: the pointer file.
    :returns: the files written.
    :rtype: list[Path]
    :raises PointerError: if a pointed version does not exist, ``exports`` is a
        file, or a version's text does not read back exactly.
    """
    prompts = resolve(pointers)
    texts = {
        name: render_export(name, pointers.app, prompt.template)
        for name, prompt in prompts.items()
    }
    try:
        pointers.exports.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:
        msg = f"exports {pointers.exports} is a file, not a directory"
        raise PointerError(msg) from exc
    for stale in _unpointed_exports(pointers):
        stale.unlink()
    written = []
    for name, text in texts.items():
        path = pointers.exports / f"{name}{EXPORT_SUFFIX}"
        path.write_text(text, encoding="utf-8", newline="")
        written.append(path)
    return written


def render_export(name: str, app: str, template: TextTemplate | ChatTemplate) -> str:
    """Return the YAML ``export`` writes for one prompt version.

    A multi-line string is written as a literal block, which reads as the prompt
    does, and any other string in YAML's usual style, when that form reads back
    exactly and leaves no line ending in whitespace (Unicode whitespace included,
    e.g. a no-break space). Otherwise (trailing spaces, U+0085, U+2028) the string
    is written double-quoted and escaped, so no tool that strips trailing whitespace
    can change an export's text.

    :param name: the prompt.
    :param app: the app, whose namespace in the registry holds the prompt.
    :param template: the version's template.
    :returns: the export's text.
    :rtype: str
    :raises PointerError: if the text would not read back exactly, or a line would
        end in whitespace.
    """
    content: dict[str, Any] = (
        {"messages": [dict(message) for message in template.messages]}
        if isinstance(template, ChatTemplate)
        else {"template": template.template}
    )
    record = {
        "name": name,
        "app": app,
        "version": template.version,
        "type": template.type,
        "template_hash": template.template_hash,
        **content,
    }
    text = yaml.dump(
        record,
        Dumper=_ExportDumper,
        sort_keys=False,
        allow_unicode=True,
        width=_NO_WRAP,
    )
    if not _clean(text, record):
        msg = f"{name}: the export would not read back as the version's text"
        raise PointerError(msg)
    return text


def _clean(text: str, value: object) -> bool:
    """Whether ``text`` reads back as ``value`` in any YAML reader, unchanged by hooks.

    No line may end in whitespace, and no raw U+0085, U+2028 or U+2029 may appear:
    YAML 1.1 (PyYAML) reads them as line breaks and YAML 1.2 does not, so another
    reader would see different text. Double quotes escape all three.
    """
    return (
        not _AMBIGUOUS_BREAKS.intersection(text)
        and yaml.safe_load(text) == value
        and all(line == line.rstrip() for line in text.split("\n"))
    )


_AMBIGUOUS_BREAKS = frozenset("\x85\u2028\u2029")


def check_exports(
    pointers: PromptPointers, prompts: Mapping[str, ManagedPrompt[Any]]
) -> list[str]:
    """Compare the committed exports with the pointed versions.

    :param pointers: the pointer file.
    :param prompts: each prompt at its pointed version, from ``resolve``.
    :returns: what does not match, one line per problem; empty when all do.
    :rtype: list[str]
    """
    problems = [
        f"{path.name} is for a prompt the pointer file does not name"
        for path in _unpointed_exports(pointers)
    ]
    for name, prompt in prompts.items():
        path = pointers.exports / f"{name}{EXPORT_SUFFIX}"
        try:
            exported = _read_export(path)
        except FileNotFoundError:
            problems.append(f"{name}: no export at {path}")
            continue
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
            problems.append(f"{name}: {path} is not an export ol-llm-prompts wrote")
            continue
        if exported.version != prompt.version:
            problems.append(
                f"{name}: the export is {exported.version}, the pointer says "
                f"{prompt.version}"
            )
        elif exported.template_hash != prompt.template_hash:
            problems.append(f"{name}: the export's text is not {prompt.version}'s")
    return problems


def load_default(name: str, *, directory: Path, template: type[TemplateT]) -> TemplateT:
    """Read a committed export as the embedded default to pass to ``get_prompt``.

    :param name: the prompt.
    :param directory: the pointer file's ``exports`` directory.
    :param template: ``TextTemplate`` or ``ChatTemplate``, the prompt's structure.
    :returns: the template, with the exported version.
    :rtype: TextTemplate | ChatTemplate
    :raises FileNotFoundError: if nothing was exported for the prompt.
    :raises TypeError: if the export is of the other structure.
    """
    loaded = _read_export(directory / f"{name}{EXPORT_SUFFIX}")
    if not isinstance(loaded, template):
        structure = "chat" if isinstance(loaded, ChatTemplate) else "text"
        msg = f"{name!r} was exported as a {structure} prompt, not {template.__name__}"
        raise TypeError(msg)
    return loaded


def _read_export(path: Path) -> TextTemplate | ChatTemplate:
    exported: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "messages" in exported:
        return ChatTemplate(
            messages=exported["messages"],
            type=exported["type"],
            version=exported["version"],
        )
    return TextTemplate(
        template=exported["template"],
        type=exported["type"],
        version=exported["version"],
    )


def _unpointed_exports(pointers: PromptPointers) -> list[Path]:
    """List this app's exports for prompts the file no longer names.

    Only a file ``export`` wrote for this app counts, so another app's export, or
    any other YAML beside it, is never reported or removed.
    """
    if not pointers.exports.is_dir():
        return []
    return sorted(
        path
        for path in pointers.exports.glob(f"*{EXPORT_SUFFIX}")
        if _export_name(path) not in pointers.prompts and _is_export(path, pointers.app)
    )


def _export_name(path: Path) -> str:
    return path.name.removesuffix(EXPORT_SUFFIX)


def _is_export(path: Path, app: str) -> bool:
    try:
        exported = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return False
    return (
        isinstance(exported, dict)
        and exported.get("app") == app
        and exported.get("name") == _export_name(path)
        and ("template" in exported or "messages" in exported)
    )


class _ExportDumper(yaml.SafeDumper):
    """Writes each string in its most readable form that reads back cleanly."""


_STR_TAG = "tag:yaml.org,2002:str"


_YAML12_NON_STRING = re.compile(
    r"null|Null|NULL|~|true|True|TRUE|false|False|FALSE"
    r"|[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+"
    r"|[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?"
    r"|[-+]?\.(inf|Inf|INF)|\.(nan|NaN|NAN)"
)
"""What a YAML 1.2 core-schema reader resolves to something other than a string.

PyYAML resolves YAML 1.1, which writes most of these quoted already but leaves
e.g. ``0o7`` and ``1e3`` plain, so another reader would load them as numbers."""


def _export_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    if _YAML12_NON_STRING.fullmatch(value):
        return dumper.represent_scalar(_STR_TAG, value, style='"')
    style = "|" if "\n" in value else None
    probe = yaml.serialize(
        yaml.ScalarNode(_STR_TAG, value, style=style),
        Dumper=yaml.SafeDumper,
        allow_unicode=True,
        width=_NO_WRAP,
    )
    if not _clean(probe, value):
        style = '"'
    return dumper.represent_scalar(_STR_TAG, value, style=style)


_ExportDumper.add_representer(str, _export_str)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``ol-llm-prompts`` command.

    ``check`` exits 2 if a pointed version does not exist or a committed export
    is not its pointed version and, with ``--environment``, 1 if a label is not on
    its pointed version. ``promote`` moves the label and prints what moved.
    ``export`` writes the embedded defaults into ``exports``. Each exits 2 when the
    pointer file cannot be read or acted on.

    :param argv: the arguments; ``sys.argv[1:]`` when omitted.
    :returns: the exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        prog="ol-llm-prompts", description="Act on an app's prompt pointer file."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="every pointed version exists and matches its export"
    )
    check.add_argument("pointers", type=Path, help="the pointer file")
    check.add_argument(
        "--environment", help="also report where this label is against the pointers"
    )
    move = commands.add_parser("promote", help="move a label to the pointed versions")
    move.add_argument("pointers", type=Path, help="the pointer file")
    move.add_argument("--environment", default=PRODUCTION, help="the label to move")
    dump = commands.add_parser("export", help="write the pointed templates")
    dump.add_argument("pointers", type=Path, help="the pointer file")
    args = parser.parse_args(argv)
    try:
        pointers = PromptPointers.load(args.pointers)
        if args.command == "export":
            for path in export(pointers):
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
                sys.stdout.flush()
            return 0
        problems = check_exports(pointers, resolve(pointers))
    except PointerError as exc:
        sys.stderr.write(f"{exc}\n")
        return POINTER_ERROR_STATUS
    if problems:
        listed = "".join(f"\n- {problem}" for problem in problems)
        sys.stderr.write(
            f"The committed exports do not match the pointers:{listed}\n"
            f"Run `ol-llm-prompts export {args.pointers}` and commit the result.\n"
        )
        return POINTER_ERROR_STATUS
    sys.stdout.write(
        f"{args.pointers}: every pointed version exists and matches its export\n"
    )
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
