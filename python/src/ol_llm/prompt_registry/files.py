"""A read-only prompt registry over committed exports, for tests and local work.

An app's exports directory (``ol_llm.pointers``) holds one
``<name>.prompt.yaml`` per managed prompt: the version its pointer file names,
which is the version the ``production`` label is moved to. So this registry
serves each prompt at that version and under that label, and has nothing else.
Moving a label or creating a version raises.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ol_llm.prompt_registry.interface import PromptRegistry, RegistryVersion
from ol_llm.templates import ChatTemplate, TextTemplate

EXPORT_SUFFIX = ".prompt.yaml"
"""What an export is named: ``<name>.prompt.yaml``."""

PRODUCTION = "production"
"""The label an export stands for."""


@dataclass(frozen=True, kw_only=True)
class Export:
    """A committed export.

    :param name: the prompt.
    :param app: the app, whose namespace in the registry holds the prompt.
    :param template: the exported version's template.
    """

    name: str
    app: str
    template: TextTemplate | ChatTemplate


def read_export(path: Path) -> Export:
    """Read an export that ``ol-llm-prompts export`` wrote.

    :param path: the file.
    :returns: the export.
    :rtype: Export
    :raises FileNotFoundError: if there is no such file.
    :raises KeyError: if a field is missing.
    :raises yaml.YAMLError: if the file is not YAML.
    """
    exported: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    template: TextTemplate | ChatTemplate = (
        ChatTemplate(
            messages=exported["messages"],
            type=exported["type"],
            version=exported["version"],
        )
        if "messages" in exported
        else TextTemplate(
            template=exported["template"],
            type=exported["type"],
            version=exported["version"],
        )
    )
    return Export(name=exported["name"], app=exported["app"], template=template)


class FilePromptRegistry(PromptRegistry):
    """Serves the exports in one directory.

    :param directory: the exports directory.
    :param labels: the labels each export is served under.
    """

    def __init__(
        self, directory: Path, *, labels: Collection[str] = (PRODUCTION,)
    ) -> None:
        """Serve the exports in ``directory`` under ``labels``."""
        self.directory = directory
        self.labels = frozenset(labels)

    @property
    def read_only(self) -> bool:
        """Always: exports change only by committing new ones."""
        return True

    def fetch(
        self,
        name: str,
        *,
        app: str,
        version: str | None = None,
        label: str | None = None,
    ) -> RegistryVersion | None:
        """Return the export for ``name``, if it is that version or label.

        :param name: the prompt's name.
        :param app: the app; an export for another app is not served.
        :param version: the version, e.g. ``v7``.
        :param label: the label; only the configured ones are served.
        :returns: the exported version, or ``None``.
        :rtype: RegistryVersion | None
        :raises ValueError: unless exactly one of ``version`` and ``label`` is
            given.
        """
        if (version is None) == (label is None):
            msg = "Pass exactly one of version and label"
            raise ValueError(msg)
        if label is not None and label not in self.labels:
            return None
        try:
            export = read_export(self.directory / f"{name}{EXPORT_SUFFIX}")
        except FileNotFoundError:
            return None
        if export.app != app or export.name != name:
            return None
        if version is not None and export.template.version != version:
            return None
        return RegistryVersion(template=export.template)

    def move_label(self, name: str, *, app: str, version: str, label: str) -> None:
        """Refuse: exports change only by committing new ones.

        :raises NotImplementedError: always.
        """
        msg = "The file prompt registry is read-only; move the label in the registry"
        raise NotImplementedError(msg)

    def create_version(
        self,
        name: str,
        template: TextTemplate | ChatTemplate,
        *,
        app: str,
        change_description: str | None = None,
    ) -> RegistryVersion:
        """Refuse: exports change only by committing new ones.

        :raises NotImplementedError: always.
        """
        msg = "The file prompt registry is read-only; create versions in the registry"
        raise NotImplementedError(msg)
