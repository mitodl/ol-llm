"""Prompt templates and how ol-llm renders them (library spec §8.2 item 4).

ol-llm renders templates itself, so an embedded default and a version fetched from
any registry render through the same code. ``mustache`` is placeholder substitution
only, with the matching rules of Opik's implementation (opik
``api_objects/prompt/text/prompt_template.py``) so versions already in Opik render
as they did: a placeholder is ``{{name}}`` (``MUSTACHE_PLACEHOLDER``), the
name stripped of whitespace, so ``{{ name }}`` is ``{{name}}`` and a third brace
stays literal text. No sections, partials, or escaping. ``jinja2`` renders in
Jinja's sandbox, as Opik does; it is supported but not portable to a registry
that does not render Jinja.
"""

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from jinja2.sandbox import SandboxedEnvironment

TemplateType = Literal["mustache", "jinja2"]
"""Mustache unless the prompt needs conditionals or loops (spec §3 item 9)."""

MUSTACHE_PLACEHOLDER = re.compile(r"\{\{([^{}]*?)\}\}")
"""A mustache placeholder. Braces are excluded from the name, so a ``{`` right
before ``{{name}}`` (``{{{name}}}``, a JSON example) stays literal."""

_MEDIA_PARTS = ("image_url", "video_url", "audio_url")
"""Chat content parts whose ``url`` is a template, as in Opik's chat renderer."""

_JINJA2 = SandboxedEnvironment()


class PromptVariablesError(ValueError):
    """The variables passed to a mustache template are not exactly its placeholders.

    :param placeholders: the template's placeholder names.
    :param variables: the names passed.
    """

    def __init__(self, placeholders: set[str], variables: set[str]) -> None:  # noqa: D107
        self.placeholders = placeholders
        self.variables = variables
        missing = sorted(placeholders - variables)
        extra = sorted(variables - placeholders)
        super().__init__(
            f"The variables do not match the template's placeholders: "
            f"missing {missing}, unexpected {extra}"
        )


@dataclass(frozen=True, kw_only=True)
class _Template(ABC):
    type: TemplateType = "mustache"
    version: str | None = None

    @abstractmethod
    def _content(self) -> str | list[dict[str, Any]]: ...

    @property
    def template_hash(self) -> str:
        """The SHA-256 of the template and its type, as hex.

        What batch idempotency keys are built on, not the version (spec §3 item
        5): a run that fell back to the embedded default gets the same hash as
        one that fetched the same text, so a registry outage does not cause
        reprocessing.

        :returns: the digest, the same for a fetched version and the embedded
            default exported from it.
        :rtype: str
        """
        canonical = json.dumps(
            {"type": self.type, "template": self._content()},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class TextTemplate(_Template):
    """A text prompt: a system or instructions string.

    :param template: the template text.
    :param type: how placeholders are written.
    :param version: the registry version this text is, e.g. ``v7``, when known.
        For an embedded default it is the pointed version that was exported.
    """

    template: str

    def _content(self) -> str:
        return self.template

    def format(self, **variables: Any) -> str:  # noqa: ANN401
        """Render the template.

        :param variables: a value for every placeholder. ``None`` renders empty.
        :returns: the rendered text.
        :rtype: str
        :raises PromptVariablesError: for a mustache template, if the variables
            are not exactly its placeholders.
        """
        if self.type == "mustache":
            _check_variables(placeholders(self.template), variables)
        return _render(self.template, variables, self.type)


@dataclass(frozen=True, kw_only=True)
class ChatTemplate(_Template):
    """A chat prompt: a list of messages with ``role`` and ``content``.

    :param messages: the template messages, in OpenAI's chat format.
    :param type: how placeholders are written.
    :param version: the registry version these messages are, when known.
    """

    messages: Sequence[Mapping[str, Any]]

    def _content(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self.messages]

    def format(self, **variables: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        """Render every message.

        A string ``content`` is a template. In a list of parts, a ``text`` part's
        ``text`` and the ``url`` of an ``image_url``, ``video_url`` or
        ``audio_url`` part are templates, and a media part with no URL or one that
        renders empty is left out; every other key and part is kept as it is, so
        media reaches the model as media.

        :param variables: a value for every placeholder.
        :returns: the rendered messages.
        :rtype: list[dict[str, Any]]
        :raises PromptVariablesError: for a mustache template, if the variables
            are not exactly the placeholders across all messages.
        """
        if self.type == "mustache":
            found: set[str] = set()

            def collect(text: str) -> str:
                found.update(placeholders(text))
                return text

            for message in self.messages:
                _map_content(message.get("content"), collect)
            _check_variables(found, variables)
        return [
            {
                **message,
                "content": _map_content(
                    message.get("content"),
                    lambda text: _render(text, variables, self.type),
                ),
            }
            for message in self.messages
        ]


TemplateT = TypeVar("TemplateT", TextTemplate, ChatTemplate)


def placeholders(text: str) -> set[str]:
    """Return the placeholder names in a mustache template, whitespace stripped.

    :param text: the template.
    :returns: the names.
    :rtype: set[str]
    """
    return {name.strip() for name in MUSTACHE_PLACEHOLDER.findall(text)}


def _check_variables(found: set[str], variables: Mapping[str, Any]) -> None:
    passed = {name.strip() for name in variables}
    if passed != found:
        raise PromptVariablesError(found, passed)


def _render(
    text: str, variables: Mapping[str, Any], template_type: TemplateType
) -> str:
    if template_type == "jinja2":
        return _JINJA2.from_string(text).render(**variables)
    values = {name.strip(): value for name, value in variables.items()}

    # One pass, so a value that itself holds "{{...}}" is inserted as it is.
    def substitute(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if name not in values:
            return match.group(0)
        value = values[name]
        return "" if value is None else str(value)

    return MUSTACHE_PLACEHOLDER.sub(substitute, text)


def _map_content(content: object, apply: Callable[[str], str]) -> object:
    """Apply ``apply`` to each template string in a message's content.

    :returns: the content with each template string replaced by what ``apply``
        returned; everything else is kept as it is.
    """
    if isinstance(content, str):
        return apply(content)
    if not isinstance(content, list):
        return content
    mapped: list[object] = []
    for part in content:
        if not isinstance(part, Mapping):
            mapped.append(part)
            continue
        new = dict(part)
        if isinstance(part.get("text"), str):
            new["text"] = apply(part["text"])
        for media in _MEDIA_PARTS:
            if isinstance(part.get(media), Mapping) and isinstance(
                part[media].get("url"), str
            ):
                new[media] = {**part[media], "url": apply(part[media]["url"])}
        # A media part with no URL, or one that renders empty (an optional image
        # left out), is dropped, as Opik's renderer does.
        if not any(
            isinstance(new.get(m), dict) and not new[m].get("url") for m in _MEDIA_PARTS
        ):
            mapped.append(new)
    return mapped
