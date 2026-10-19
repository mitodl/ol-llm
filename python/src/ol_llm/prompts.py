"""Managed prompts (lifecycle spec §2 and §3): Opik is the system of record.

Deployed code fetches a prompt by its environment label with ``get_prompt``. The
Opik SDK keeps each fetched prompt in a process-wide cache, serves it from there,
and refreshes it in a background thread every ``OPIK_PROMPT_CACHE_TTL_SECONDS``
(300 by default), keeping the cached value when a refresh fails. So a label move
reaches every process within one TTL, and only a fetch with nothing cached can
fail.

A cold fetch with Opik unreachable raises, after as long as the SDK's own retries
take. ``get_prompt`` waits ``COLD_FETCH_DEADLINE_SECONDS`` for it and then returns
the embedded default the caller passed, the text of the pointed production
version exported into the image at build time, marked degraded with
``embedded_prompt``. The fetch carries on in the background and fills the cache,
so the next call gets the managed version.

With tracing off (``OPIK_TRACK_DISABLE=true``) there is no Opik to fetch from, so
``get_prompt`` returns the embedded default unmarked: that is local development.
"""

import contextvars
import hashlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, cast, overload

import httpx
import opik
from opik import context_storage, opik_context
from opik.api_objects.prompt import (
    BasePrompt,
    ChatPromptTemplate,
    PromptTemplate,
    PromptType,
    mask_context,
)
from opik.rest_api.core import ApiError

from ol_llm import tracing
from ol_llm.attribution import DegradedReason
from ol_llm.settings import get_settings

logger = logging.getLogger(__name__)

COLD_FETCH_DEADLINE_SECONDS = 2.0
"""How long ``get_prompt`` waits for Opik before using the embedded default. The
SDK took 11 s to give up on an unreachable Opik (spec §3 item 4)."""

TemplateType = Literal["mustache", "jinja2"]
"""Mustache unless the prompt needs conditionals or loops (spec §3 item 9)."""

PromptSource = Literal["opik", "embedded"]


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
        one that fetched the same text, so an Opik outage does not cause
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
    :param version: the Opik version this text is, e.g. ``v7``, when known. For an
        embedded default it is the pointed version the build exported.
    """

    template: str

    def _content(self) -> str:
        return self.template

    def format(self, **variables: Any) -> str:  # noqa: ANN401
        """Render the template.

        :param variables: a value for every placeholder.
        :returns: the rendered text.
        :rtype: str
        :raises opik.exceptions.PromptPlaceholdersDontMatchFormatArguments: for a
            mustache template, if the variables are not exactly its placeholders.
        """
        return PromptTemplate(self.template, type=PromptType(self.type)).format(
            **variables
        )


@dataclass(frozen=True, kw_only=True)
class ChatTemplate(_Template):
    """A chat prompt: a list of messages with ``role`` and ``content``.

    :param messages: the template messages, in OpenAI's chat format.
    :param type: how placeholders are written.
    :param version: the Opik version these messages are, when known.
    """

    messages: Sequence[Mapping[str, Any]]

    def _content(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self.messages]

    def format(self, **variables: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        """Render every message.

        :param variables: a value for every placeholder.
        :returns: the rendered messages.
        :rtype: list[dict[str, Any]]
        :raises opik.exceptions.PromptPlaceholdersDontMatchFormatArguments: for a
            mustache template, if the variables are not exactly its placeholders.
        """
        rendered = ChatPromptTemplate(
            self._content(),
            template_type=PromptType(self.type),
            validate_placeholders=True,
        ).format(variables)
        return cast("list[dict[str, Any]]", rendered)


TemplateT = TypeVar("TemplateT", TextTemplate, ChatTemplate)


@dataclass(frozen=True, kw_only=True)
class ManagedPrompt(Generic[TemplateT]):
    """A prompt from Opik, or the embedded default standing in for it.

    Pass it to ``run`` as ``prompt=`` so the run's trace links the version and the
    result carries its name and version.

    :param name: the prompt's name, ``<use_case>`` or ``<use_case>.<part>``.
    :param template: the template that was used.
    :param source: ``opik`` when fetched, ``embedded`` for the default.
    :param degraded_reasons: ``embedded_prompt`` when Opik could not be reached.
    :param opik_prompt: the SDK's prompt, for the trace link; ``None`` for the
        embedded default.
    """

    name: str
    template: TemplateT
    source: PromptSource
    degraded_reasons: tuple[DegradedReason, ...] = ()
    opik_prompt: BasePrompt | None = field(default=None, repr=False, compare=False)

    @property
    def version(self) -> str | None:
        """The Opik version, e.g. ``v7``, when known."""
        return self.template.version

    @property
    def template_hash(self) -> str:
        """The template's SHA-256; see ``TextTemplate.template_hash``."""
        return self.template.template_hash

    @overload
    def format(
        self: "ManagedPrompt[TextTemplate]",
        **variables: Any,  # noqa: ANN401
    ) -> str: ...

    @overload
    def format(
        self: "ManagedPrompt[ChatTemplate]",
        **variables: Any,  # noqa: ANN401
    ) -> list[dict[str, Any]]: ...

    def format(self, **variables: Any) -> str | list[dict[str, Any]]:
        """Render the template; see ``TextTemplate.format`` and ``ChatTemplate.format``.

        :param variables: a value for every placeholder.
        :returns: the text, or the messages for a chat prompt.
        :rtype: str | list[dict[str, Any]]
        """
        return self.template.format(**variables)


def get_prompt(
    name: str,
    *,
    project_name: str,
    default: TemplateT,
    environment: str | None = None,
    version: str | None = None,
) -> ManagedPrompt[TemplateT]:
    """Fetch a prompt by environment label, or a pinned version, from Opik.

    Fetch per run rather than once at startup: the SDK cache makes a repeat fetch
    free, and a label move then reaches the next run. A batch run fetches once at
    the start and uses that version throughout (spec §3 item 2). Called inside an
    Opik trace, it links the version to the trace as the SDK does.

    :param name: the prompt's name.
    :param project_name: the app's Opik project, which holds its prompts.
    :param default: the embedded default: what to use when Opik is unreachable or
        tracing is off. Its class says whether the prompt is text or chat.
    :param environment: the label to resolve; the deployment environment
        (``Settings.environment``) when neither this nor ``version`` is given.
    :param version: a pinned version, e.g. ``v7``, for batch runs and
        evaluations. The SDK refreshes a pinned version on the same TTL as a
        label.
    :returns: the prompt.
    :rtype: ManagedPrompt
    :raises ValueError: if both ``environment`` and ``version`` are given.
    :raises LookupError: if Opik has no such prompt, label, or version.
    :raises opik.exceptions.PromptTemplateStructureMismatch: if Opik holds the
        prompt as chat and the default is text, or the reverse.
    """
    if environment is not None and version is not None:
        msg = "Pass environment or version, not both"
        raise ValueError(msg)
    if not tracing.enabled():
        return ManagedPrompt(name=name, template=default, source="embedded")
    if version is None and environment is None:
        environment = get_settings().environment
    lookup = _Lookup(
        name=name,
        project_name=project_name,
        chat=isinstance(default, ChatTemplate),
        environment=environment,
        version=version,
        masks=frozenset((mask_context.get_active_prompt_masks() or {}).items()),
    )
    try:
        fetched = _fetch_within_deadline(lookup)
    except Exception as exc:
        if not _unreachable(exc):
            raise
        logger.warning(
            "Opik did not return prompt %r (%s); using the embedded default",
            name,
            type(exc).__name__,
            exc_info=exc,
        )
        return ManagedPrompt(
            name=name,
            template=default,
            source="embedded",
            degraded_reasons=("embedded_prompt",),
        )
    if fetched is None:
        selector = f"version {version}" if version else f"environment {environment}"
        msg = f"Opik has no prompt {name!r} in {project_name!r} for {selector}"
        raise LookupError(msg)
    prompt = ManagedPrompt(
        name=name,
        template=cast("TemplateT", _template(fetched)),
        source="opik",
        opik_prompt=fetched,
    )
    attach_prompt(prompt)
    return prompt


def attach_prompt(prompt: ManagedPrompt[Any]) -> None:
    """Link a fetched prompt's version to the current trace and span.

    Does nothing for the embedded default, outside a trace, or with tracing off.

    :param prompt: the prompt.
    """
    if prompt.opik_prompt is not None:
        opik_context.attach_prompt_to_current_trace(prompt.opik_prompt)
        opik_context.attach_prompt_to_current_span(prompt.opik_prompt)


@dataclass(frozen=True, kw_only=True)
class _Lookup:
    name: str
    project_name: str
    chat: bool
    environment: str | None
    version: str | None
    masks: frozenset[tuple[str, str]]


_inflight_lock = threading.Lock()
_inflight: dict[_Lookup, Future[BasePrompt | None]] = {}


def _client() -> opik.Opik:
    # Settles the tracing configuration first: that registers the Keycloak hook,
    # which the SDK reads only when it builds its client.
    tracing.get_tracing_config()
    return opik.get_global_client()


def _fetch(lookup: _Lookup) -> BasePrompt | None:
    client = _client()
    getter = client.get_chat_prompt if lookup.chat else client.get_prompt
    return getter(
        name=lookup.name,
        project_name=lookup.project_name,
        environment=lookup.environment,
        version=lookup.version,
    )


def _fetch_within_deadline(lookup: _Lookup) -> BasePrompt | None:
    """Fetch through the SDK cache in a worker thread, waiting at most the deadline.

    A cached prompt comes back at once. Concurrent cold fetches of one prompt
    under the same masks share a worker, which keeps going after the deadline so
    its result reaches the cache. The worker has no trace, so the SDK links
    nothing there; ``get_prompt`` links the result in the caller's context.
    """
    with _inflight_lock:
        if (future := _inflight.get(lookup)) is None:
            future = _inflight[lookup] = Future()
            try:
                threading.Thread(
                    target=contextvars.copy_context().run,
                    args=(_resolve, lookup, future),
                    name="ol-llm-prompt-fetch",
                    daemon=True,
                ).start()
            except BaseException:
                del _inflight[lookup]
                raise
    return future.result(timeout=COLD_FETCH_DEADLINE_SECONDS)


def _resolve(lookup: _Lookup, future: Future[BasePrompt | None]) -> None:
    # Runs in a copy of the caller's context for the SDK's prompt masks (Opik's
    # local runner), minus the trace: a fetch that outlives the deadline must
    # not link a version the caller did not use.
    context_storage.clear_all()
    # Retired before it resolves, so a caller arriving later starts a fresh fetch
    # instead of getting this one's error.
    try:
        result = _fetch(lookup)
    except Exception as exc:  # noqa: BLE001 - handed to the waiting caller
        _retire(lookup)
        future.set_exception(exc)
    else:
        _retire(lookup)
        future.set_result(result)


def _retire(lookup: _Lookup) -> None:
    with _inflight_lock:
        del _inflight[lookup]


_TRANSPORT_FAILURES = (
    TimeoutError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
)


def _unreachable(exc: Exception) -> bool:
    """Whether ``exc`` means Opik could not be reached, not that it said no.

    Only transport failures, timeouts, and 5xx responses count. A 4xx from Opik,
    or from Keycloak on the token request (``invalid_client`` for a wrong secret),
    and a bad URL are configuration errors: falling back on them would hide the
    error in every process, and the degraded mark could not reach Opik either,
    since span export uses the same token.
    """
    if isinstance(exc, _TRANSPORT_FAILURES):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.is_server_error
    if isinstance(exc, ApiError):
        return (
            exc.status_code is not None
            and exc.status_code >= httpx.codes.INTERNAL_SERVER_ERROR
        )
    return False


def _template(prompt: BasePrompt) -> TextTemplate | ChatTemplate:
    template_type: TemplateType = PromptType(prompt.type).value
    if isinstance(prompt, opik.ChatPrompt):
        return ChatTemplate(
            messages=prompt.template, type=template_type, version=prompt.version
        )
    return TextTemplate(
        template=cast("opik.Prompt", prompt).prompt,
        type=template_type,
        version=prompt.version,
    )
