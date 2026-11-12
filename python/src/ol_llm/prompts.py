"""Managed prompts (lifecycle spec §2 and §3), from the prompt registry.

Deployed code fetches a prompt by its label with ``get_prompt``. The registry is
whichever ``ol_llm.prompt_registry`` is configured (Opik by default); the runtime
policy here applies to every one. Each fetched version is kept in a process-wide
cache for ``PROMPT_CACHE_TTL_SECONDS``. The first call after that fetches it again,
waiting up to ``COLD_FETCH_DEADLINE_SECONDS``, and gets the cached version if the
fetch fails, finds nothing, or is slow; a failed refresh is tried again after
another TTL. So a label move reaches every process within one TTL, and only a fetch
with nothing cached can fail.

A cold fetch waits ``COLD_FETCH_DEADLINE_SECONDS`` and then, if the registry has
not answered or could not be reached, returns the embedded default the caller
passed, the text of the pointed production version committed beside the pointer
file (``ol_llm.pointers``), marked degraded with ``embedded_prompt``. A fetch that
missed the deadline carries on in the background and fills the cache, so the next
call gets the managed version.

With the registry disabled (for Opik, tracing off: ``OPIK_TRACK_DISABLE=true``)
``get_prompt`` does not fetch and returns the embedded default unmarked: that is
local development. ``fetch_prompt``, ``create_prompt`` and ``promote_prompt`` still
call the registry.

``create_prompt`` and ``promote_prompt`` are for the CI jobs that sync prompts
and move the ``production`` label from the pointer file (spec §2.3).
"""

import contextvars
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Hashable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, cast, overload

from ol_llm.attribution import DegradedReason
from ol_llm.loops import register_at_fork
from ol_llm.prompt_registry import (
    PromptRegistry,
    RegistryUnavailableError,
    RegistryVersion,
    prompt_registry,
)
from ol_llm.settings import get_settings
from ol_llm.templates import (
    ChatTemplate,
    PromptVariablesError,
    TemplateT,
    TemplateType,
    TextTemplate,
)

__all__ = [
    "COLD_FETCH_DEADLINE_SECONDS",
    "PROMPT_CACHE_MAX_ENTRIES",
    "PROMPT_CACHE_TTL_SECONDS",
    "ChatTemplate",
    "ManagedPrompt",
    "PromptSource",
    "PromptVariablesError",
    "TemplateT",
    "TemplateType",
    "TextTemplate",
    "attach_prompt",
    "create_prompt",
    "fetch_prompt",
    "get_prompt",
    "promote_prompt",
]

logger = logging.getLogger(__name__)

COLD_FETCH_DEADLINE_SECONDS = 2.0
"""How long ``get_prompt`` waits for the registry before using the embedded
default. The Opik SDK took 11 s to give up on an unreachable Opik (spec §3 item 4)."""

PROMPT_CACHE_TTL_SECONDS = 300.0
"""How long a fetched version is served before it is fetched again (spec §3)."""

PROMPT_CACHE_MAX_ENTRIES = 128
"""How many fetched versions the cache keeps, least recently used out first, as the
Opik SDK's cache did."""

PromptSource = Literal["registry", "embedded"]
"""Where a prompt came from: the prompt registry, or the embedded default."""


@dataclass(frozen=True, kw_only=True)
class ManagedPrompt(Generic[TemplateT]):
    """A prompt from the prompt registry, or the embedded default standing in for it.

    Pass it to ``run`` as ``prompt=`` so the run's trace links the version and the
    result carries its name and version.

    :param name: the prompt's name, ``<use_case>`` or ``<use_case>.<part>``.
    :param template: the template that was used.
    :param source: ``registry`` when fetched, ``embedded`` for the default.
    :param degraded_reasons: ``embedded_prompt`` when the registry could not be
        reached.
    :param registry_ref: the registry's own object for this version, for the trace
        link; ``None`` for the embedded default or a registry with no such link.
        Opaque to callers.
    """

    name: str
    template: TemplateT
    source: PromptSource
    degraded_reasons: tuple[DegradedReason, ...] = ()
    registry_ref: object | None = field(default=None, repr=False, compare=False)

    @property
    def version(self) -> str | None:
        """The registry's version, e.g. ``v7``, when known."""
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
    app: str,
    default: TemplateT,
    environment: str | None = None,
    version: str | None = None,
) -> ManagedPrompt[TemplateT]:
    """Fetch a prompt by label, or a pinned version, from the prompt registry.

    Fetch per run rather than once at startup: the cache makes a repeat fetch
    free, and a label move then reaches the next run. A batch run fetches once at
    the start and uses that version throughout (spec §3 item 2). Called inside a
    trace, it links the version to the trace where the registry can.

    :param name: the prompt's name.
    :param app: the app, whose namespace in the registry holds its prompts.
    :param default: the embedded default: what to use when the registry is
        unreachable or disabled. Its class says whether the prompt is text or chat.
    :param environment: the label to resolve; the deployment environment
        (``Settings.environment``) when neither this nor ``version`` is given.
    :param version: a pinned version, e.g. ``v7``, for batch runs and
        evaluations. Cached on the same TTL as a label.
    :returns: the prompt.
    :rtype: ManagedPrompt
    :raises ValueError: if both ``environment`` and ``version`` are given.
    :raises LookupError: if the registry has no such prompt, label, or version.
    :raises TypeError: if the registry holds the prompt as chat and the default
        is text, or the reverse.
    """
    if environment is not None and version is not None:
        msg = "Pass environment or version, not both"
        raise ValueError(msg)
    registry = prompt_registry()
    if not registry.enabled:
        return ManagedPrompt(name=name, template=default, source="embedded")
    if version is None and environment is None:
        environment = get_settings().environment
    lookup = _Lookup(
        registry=registry,
        name=name,
        app=app,
        label=environment,
        version=version,
        context=registry.context_key(),
    )
    try:
        found = _cached_fetch(lookup)
    except (RegistryUnavailableError, TimeoutError) as exc:
        logger.warning(
            "The prompt registry did not return prompt %r (%s); using the "
            "embedded default",
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
    if found is None:
        selector = f"version {version}" if version else f"environment {environment}"
        msg = f"The prompt registry has no prompt {name!r} in {app!r} for {selector}"
        raise LookupError(msg)
    if isinstance(found.template, ChatTemplate) != isinstance(default, ChatTemplate):
        msg = (
            f"Prompt {name!r} is a {_structure(found.template)} prompt in the "
            f"registry, but the default is {type(default).__name__}"
        )
        raise TypeError(msg)
    prompt = ManagedPrompt(
        name=name,
        template=cast("TemplateT", found.template),
        source="registry",
        registry_ref=found.ref,
    )
    attach_prompt(prompt)
    return prompt


def attach_prompt(prompt: ManagedPrompt[Any]) -> None:
    """Link a fetched prompt's version to the current trace and span.

    Does nothing for the embedded default, for a registry with no trace link, or
    outside a trace.

    :param prompt: the prompt.
    """
    if prompt.registry_ref is not None:
        prompt_registry().link(prompt.registry_ref)


def create_prompt(
    name: str,
    template: TemplateT,
    *,
    app: str,
    change_description: str | None = None,
) -> ManagedPrompt[TemplateT]:
    """Make ``template`` the latest version of a prompt, for a CI sync job.

    Idempotent: a new version is created only when the template text or type
    differs from the latest version's.

    :param name: the prompt's name.
    :param template: the template; its class says whether the prompt is text or
        chat, which cannot change once the prompt exists.
    :param app: the app, whose namespace in the registry holds the prompt.
    :param change_description: recorded on a new version.
    :returns: the latest version, new or existing.
    :rtype: ManagedPrompt
    :raises TypeError: if the prompt exists with the other structure.
    """
    created = prompt_registry().create_version(
        name, template, app=app, change_description=change_description
    )
    return _managed(name, created)


def promote_prompt(name: str, *, version: str, environment: str, app: str) -> None:
    """Move a label to a version, for the promotion CI job.

    The version keeps the labels it already has. Clears this process's cached
    entries for the prompt.

    :param name: the prompt's name.
    :param version: the version to label, e.g. ``v7``.
    :param environment: the label to move, e.g. ``production``.
    :param app: the app, whose namespace in the registry holds the prompt.
    :raises LookupError: if there is no such prompt or version.
    """
    prompt_registry().move_label(name, app=app, version=version, label=environment)
    with _inflight_lock:
        for lookup in [k for k in _cache if (k.name, k.app) == (name, app)]:
            del _cache[lookup]
        # A fetch in flight may have read the label before the move.
        for lookup in [k for k in _inflight if (k.name, k.app) == (name, app)]:
            del _inflight[lookup]


def fetch_prompt(
    name: str,
    *,
    app: str,
    version: str | None = None,
    environment: str | None = None,
) -> ManagedPrompt[Any] | None:
    """Fetch one version of a prompt from the registry, text or chat, for a CI job.

    Unlike ``get_prompt`` it needs no embedded default, since it takes the
    structure from the registry, and it neither caches nor falls back: a CI job
    that cannot reach the registry should fail.

    :param name: the prompt's name.
    :param app: the app, whose namespace in the registry holds the prompt.
    :param version: the version, e.g. ``v7``.
    :param environment: the label to resolve, e.g. ``production``.
    :returns: the prompt, or ``None`` when the registry has no such prompt,
        version, or label.
    :rtype: ManagedPrompt | None
    :raises ValueError: unless exactly one of ``version`` and ``environment`` is
        given.
    :raises RegistryUnavailableError: if the registry could not be reached.
    """
    if (version is None) == (environment is None):
        msg = "Pass exactly one of version and environment"
        raise ValueError(msg)
    found = prompt_registry().fetch(name, app=app, version=version, label=environment)
    return None if found is None else _managed(name, found)


def _managed(name: str, found: RegistryVersion) -> ManagedPrompt[Any]:
    return ManagedPrompt(
        name=name, template=found.template, source="registry", registry_ref=found.ref
    )


def _structure(template: TextTemplate | ChatTemplate) -> str:
    return "chat" if isinstance(template, ChatTemplate) else "text"


@dataclass(frozen=True, kw_only=True)
class _Lookup:
    registry: PromptRegistry
    name: str
    app: str
    label: str | None
    version: str | None
    context: Hashable


@dataclass
class _Entry:
    found: RegistryVersion
    checked_at: float


_inflight_lock = threading.Lock()
_inflight: dict[_Lookup, Future[RegistryVersion | None]] = {}
_cache: OrderedDict[_Lookup, _Entry] = OrderedDict()


def _after_fork_in_child() -> None:
    # The parent's fetch threads didn't survive the fork, so their futures would
    # never resolve, and a parent thread may have held the lock. The cache is
    # plain data and stays.
    global _inflight_lock, _inflight  # noqa: PLW0603
    _inflight_lock = threading.Lock()
    _inflight = {}


register_at_fork(after_in_child=_after_fork_in_child)


def _cached_fetch(lookup: _Lookup) -> RegistryVersion | None:
    """Return the version for ``lookup``, from the cache while it is fresh.

    Otherwise fetch in a worker thread and wait at most the deadline. Concurrent
    fetches of one lookup share a worker, which keeps going after the deadline so
    its result reaches the cache. A stale entry is served when its refresh fails,
    finds nothing, or misses the deadline, as the Opik SDK's cache did, and at once
    to every caller but the one that started the refresh.

    :raises TimeoutError: if a cold fetch missed the deadline.
    """
    with _inflight_lock:
        entry = _cache.get(lookup)
        if entry is not None:
            _cache.move_to_end(lookup)
            fresh = time.monotonic() - entry.checked_at < PROMPT_CACHE_TTL_SECONDS
            # Only the caller that starts a refresh waits for it.
            if fresh or lookup in _inflight:
                return entry.found
        future = _start(lookup)
    try:
        found = future.result(timeout=COLD_FETCH_DEADLINE_SECONDS)
    except Exception:
        if entry is None:
            raise
        return entry.found
    return found if found is not None or entry is None else entry.found


def _start(lookup: _Lookup) -> Future[RegistryVersion | None]:
    """Start a worker for ``lookup`` unless one is running; hold the lock."""
    if (future := _inflight.get(lookup)) is not None:
        return future
    future = _inflight[lookup] = Future()
    try:
        # A copy of the caller's context, for registry state such as Opik's
        # prompt masks that context_key reads.
        threading.Thread(
            target=contextvars.copy_context().run,
            args=(_resolve, lookup, future),
            name="ol-llm-prompt-fetch",
            daemon=True,
        ).start()
    except BaseException:
        del _inflight[lookup]
        raise
    return future


def _resolve(lookup: _Lookup, future: Future[RegistryVersion | None]) -> None:
    # Retired before it resolves, so a caller arriving later starts a fresh fetch
    # instead of getting this one's error. A worker that promote_prompt evicted
    # writes nothing, so a version fetched before the move is not cached after it.
    try:
        found = lookup.registry.fetch(
            lookup.name, app=lookup.app, version=lookup.version, label=lookup.label
        )
    except Exception as exc:  # noqa: BLE001 - handed to the waiting caller
        _settle(lookup, future, None, exc)
        future.set_exception(exc)
        return
    _settle(lookup, future, found, None)
    future.set_result(found)


def _settle(
    lookup: _Lookup,
    future: Future[RegistryVersion | None],
    found: RegistryVersion | None,
    exc: Exception | None,
) -> None:
    """Record a fetch's outcome in the cache, unless promote_prompt evicted it.

    A failed or empty refresh keeps the cached version and is tried again after
    another TTL, not on every call.
    """
    with _inflight_lock:
        if _inflight.get(lookup) is not future:
            return
        del _inflight[lookup]
        now = time.monotonic()
        if found is not None:
            _cache[lookup] = _Entry(found, now)
            _cache.move_to_end(lookup)
            while len(_cache) > PROMPT_CACHE_MAX_ENTRIES:
                _cache.popitem(last=False)
            return
        if (entry := _cache.get(lookup)) is None:
            return
        entry.checked_at = now
    logger.warning(
        "Could not refresh prompt %r (%s); serving the cached version",
        lookup.name,
        "not found" if exc is None else type(exc).__name__,
        exc_info=exc,
    )
