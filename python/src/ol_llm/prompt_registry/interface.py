"""What a prompt registry does for ol-llm (library spec §8.3 seam 2).

A registry holds an app's prompts as numbered versions and moves labels between
them. A label is a mutable pointer to a version: Opik calls it an environment,
Langfuse a label, MLflow an alias. Versions are opaque strings, as the registry
names them. ol-llm's runtime policy (the cache, the cold-start deadline, the
embedded fallback) sits above this interface in ``ol_llm.prompts``, so every
registry gets it.
"""

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field

from ol_llm.templates import ChatTemplate, TextTemplate


class RegistryUnavailableError(Exception):
    """The registry could not be reached, as opposed to refusing the request.

    ``get_prompt`` falls back to the embedded default on this and only this, so a
    backend raises it for transport failures and server errors, never for a
    refusal or a configuration error, which falling back would hide.
    """


@dataclass(frozen=True, kw_only=True)
class RegistryVersion:
    """One version of a prompt, as a registry returned it.

    :param template: the template, carrying the registry's version name.
    :param ref: the registry's own object for the version, which it uses to link
        the version to a trace; ``None`` when it has no such link.
    """

    template: TextTemplate | ChatTemplate
    ref: object | None = field(default=None, repr=False, compare=False)


class PromptRegistry(ABC):
    """A store of prompt versions and the labels on them, per app."""

    @property
    def enabled(self) -> bool:
        """Whether ``get_prompt`` fetches from this registry at all.

        When ``False`` it returns the embedded default unmarked, as local
        development without the registry does.
        """
        return True

    @property
    def read_only(self) -> bool:
        """Whether ``move_label`` and ``create_version`` refuse.

        ``ol-llm-prompts promote`` checks it first, since a read-only registry can
        report a label as in place without any registry having moved it.
        """
        return False

    def context_key(self) -> Hashable:
        """Return what, in the calling context, changes what ``fetch`` returns.

        ``get_prompt`` caches per value of it, and fetches in a copy of the
        caller's context, so a backend can read the same state there.

        :returns: a hashable value; ``None`` when nothing does.
        :rtype: Hashable
        """
        return None

    @abstractmethod
    def fetch(
        self,
        name: str,
        *,
        app: str,
        version: str | None = None,
        label: str | None = None,
    ) -> RegistryVersion | None:
        """Fetch one version of a prompt, by version or by the label on it.

        :param name: the prompt's name.
        :param app: the app, whose namespace holds the prompt.
        :param version: the version, e.g. ``v7``.
        :param label: the label, e.g. ``production``.
        :returns: the version, or ``None`` when there is no such prompt, version,
            or label.
        :rtype: RegistryVersion | None
        :raises RegistryUnavailableError: if the registry could not be reached.
        """

    def label_version(self, name: str, *, app: str, label: str) -> str | None:
        """Return the version a label is on.

        :param name: the prompt's name.
        :param app: the app, whose namespace holds the prompt.
        :param label: the label.
        :returns: the version, or ``None`` when no version carries the label.
        :rtype: str | None
        :raises RegistryUnavailableError: if the registry could not be reached.
        """
        found = self.fetch(name, app=app, label=label)
        return None if found is None else found.template.version

    @abstractmethod
    def move_label(self, name: str, *, app: str, version: str, label: str) -> None:
        """Put a label on a version, taking it off whichever version had it.

        The version keeps the labels it already has.

        :param name: the prompt's name.
        :param app: the app, whose namespace holds the prompt.
        :param version: the version to label.
        :param label: the label to move.
        :raises LookupError: if there is no such prompt or version.
        :raises NotImplementedError: if the registry is read-only.
        """

    @abstractmethod
    def create_version(
        self,
        name: str,
        template: TextTemplate | ChatTemplate,
        *,
        app: str,
        change_description: str | None = None,
    ) -> RegistryVersion:
        """Make ``template`` the latest version of a prompt.

        Idempotent: a version is created only when the template's text or type
        differs from the latest version's.

        :param name: the prompt's name.
        :param template: the template; its class says whether the prompt is text
            or chat, which cannot change once the prompt exists.
        :param app: the app, whose namespace holds the prompt.
        :param change_description: recorded on a new version.
        :returns: the latest version, new or existing.
        :rtype: RegistryVersion
        :raises TypeError: if the prompt exists with the other structure.
        :raises NotImplementedError: if the registry is read-only.
        """

    def link(self, ref: object) -> None:  # noqa: B027 - optional, a no-op by default
        """Link a fetched version to the current trace, if the registry can.

        :param ref: the version's ``RegistryVersion.ref``.
        """
