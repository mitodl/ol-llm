"""What call sites hold: the token handling, tracing, and checks around a backend.

Every operation gets a span with the §4 attribution fields (app, use case,
environment, and the per-call ``Attribution``), so recall is visible in the calling
trace. Keys, apps and counts are recorded always. Note text, queries and metadata
are recorded only when ``include_content`` is on.
"""

import builtins
import json
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from opentelemetry.trace import Span

from ol_llm.attribution import NO_ATTRIBUTION, Attribution
from ol_llm.loops import run_coroutine
from ol_llm.memory.errors import MemoryUnsupported
from ol_llm.memory.interface import (
    DEFAULT_SEARCH_LIMIT,
    Capability,
    ForgetStore,
    LearnerCredential,
    MemoryBackend,
    Note,
    NoteStore,
    ServiceCredential,
    TokenSource,
    check_general_name,
    check_key,
)
from ol_llm.memory.tokens import TokenExchange
from ol_llm.policy import _NAME
from ol_llm.settings import get_settings
from ol_llm.traces import _record_error
from ol_llm.tracing import include_content, tracer_provider

Scope = Literal["learner", "general", "admin"]

_AttributeValue = str | int | float | bool | Sequence[str]


@dataclass(frozen=True, kw_only=True)
class _Call:
    """Who is calling, for the spans and the capability checks."""

    backend: MemoryBackend
    use_case: str
    attribution: Attribution
    scope: Scope
    general_name: str | None = None

    def require(self, capability: Capability, operation: str) -> None:
        if capability not in self.backend.capabilities:
            msg = f"the {self.backend.name} memory backend does not support {operation}"
            raise MemoryUnsupported(msg, operation=operation)

    @contextmanager
    def span(
        self,
        operation: str,
        attributes: Mapping[str, _AttributeValue] | None = None,
        content: Mapping[str, _AttributeValue] | None = None,
    ) -> Generator[Span | None]:
        """Open the operation's span; ``None`` when tracing is off."""
        provider = tracer_provider(self.backend.app)
        if provider is None:
            yield None
            return
        with_content = include_content()
        recorded: dict[str, _AttributeValue] = {
            "app": self.backend.app,
            "use_case": self.use_case,
            "environment": get_settings().environment,
            "memory.operation": operation,
            "memory.scope": self.scope,
            "memory.backend": self.backend.name,
            **(attributes or {}),
        }
        if self.general_name is not None:
            recorded["memory.general_name"] = self.general_name
        for name in ("user_id", "thread_id", "run_id"):
            if (value := getattr(self.attribution, name)) is not None:
                recorded[name] = value
        if with_content:
            recorded.update(content or {})
        with provider.get_tracer("ol_llm").start_as_current_span(
            f"memory {operation}",
            attributes=recorded,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except Exception as exc:
                _record_error(span, exc, content=with_content)
                raise


def _record_notes(span: Span | None, notes: Sequence[Note]) -> None:
    if span is None or not span.is_recording():
        return
    span.set_attributes(
        {
            "memory.results_count": len(notes),
            "memory.result_keys": [f"{note.app}.{note.key}" for note in notes],
        }
    )
    if include_content():
        span.set_attribute(
            "memory.results",
            json.dumps(
                [
                    {
                        "app": note.app,
                        "key": note.key,
                        "text": note.text,
                        "score": note.score,
                    }
                    for note in notes
                ]
            ),
        )


class _Notes:
    """The five note operations, traced, over one backend store."""

    def __init__(self, store: NoteStore, call: _Call) -> None:
        self._store = store
        self._call = call

    @property
    def app(self) -> str:
        """The app this handle writes as."""
        return self._call.backend.app

    async def put(
        self, key: str, text: str, *, metadata: Mapping[str, str] | None = None
    ) -> None:
        """Store or replace this app's note ``key``. There is no update in place.

        :param key: the note's key: lowercase letters, digits, ``_`` and ``-``.
            Never learner content; the service does not redact keys.
        :param text: the note.
        :param metadata: stored with the note; redacted like the text.
        :raises ValueError: if the key is not valid.
        :raises MemoryFenced: while the learner's memory is being forgotten.
        :raises MemoryForbidden: if this credential may not write here.
        """
        check_key(key)
        content = {"memory.text": text}
        if metadata:
            content["memory.metadata"] = json.dumps(dict(metadata))
        with self._call.span("put", {"memory.key": key}, content):
            await self._store.put(key, text, metadata=metadata)

    async def get(self, key: str, *, app: str | None = None) -> Note:
        """Return the note ``key`` written by ``app``, this app when omitted.

        :param key: the note's key.
        :param app: the app that wrote it.
        :returns: the note.
        :rtype: Note
        :raises MemoryNotFound: if there is no such note.
        """
        check_key(key)
        writer = app or self.app
        with self._call.span(
            "get", {"memory.key": key, "memory.writer": writer}
        ) as span:
            note = await self._store.get(key, app=writer)
            _record_notes(span, [note])
        return note

    async def search(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> builtins.list[Note]:
        """Return up to ``limit`` notes most relevant to ``query``, best first.

        :param query: what to look for.
        :param limit: the most notes to return.
        :returns: the notes, each with its score.
        :rtype: list[Note]
        :raises ValueError: if ``limit`` is not positive.
        :raises MemoryUnsupported: if the backend cannot search.
        """
        if limit < 1:
            msg = f"limit must be positive, not {limit}"
            raise ValueError(msg)
        self._call.require("search", "search")
        with self._call.span(
            "search", {"memory.limit": limit}, {"memory.query": query}
        ) as span:
            notes = await self._store.search(query, limit=limit)
            _record_notes(span, notes)
        return notes

    async def delete(self, key: str) -> None:
        """Delete this app's note ``key``. Deleting a missing note is not an error.

        :param key: the note's key.
        :raises MemoryForbidden: if this credential may not write here.
        """
        check_key(key)
        with self._call.span("delete", {"memory.key": key}):
            await self._store.delete(key)

    async def list(self) -> builtins.list[Note]:
        """Return every note, every app's.

        :returns: the notes.
        :rtype: list[Note]
        """
        with self._call.span("list") as span:
            notes = await self._store.list()
            _record_notes(span, notes)
        return notes


class GeneralMemory(_Notes):
    """A general memory: facts that belong to no learner, e.g. about courses.

    Every app reads it. Writing needs the general grant, so only a handle from
    ``Memory.general`` writes; one from ``LearnerMemory.general`` raises
    ``MemoryForbidden`` on ``put`` and ``delete``.
    """


class LearnerMemory(_Notes):
    """One learner's memory, as one app sees it, for one request.

    The learner is whoever the session token was for; nothing names the learner
    otherwise. Every app reads every app's notes for the learner, and writes and
    deletes only its own. The handle holds a token that expires (300 s on the
    deployed service), so get a new one for each request rather than keeping it.
    """

    def __init__(
        self, store: NoteStore, call: _Call, credential: LearnerCredential
    ) -> None:
        """Wrap the backend's store for the learner."""
        super().__init__(store, call)
        self._credential = credential

    def general(self, name: str) -> GeneralMemory:
        """Return the general memory ``name`` to read with this learner's token.

        :param name: e.g. ``courses``: lowercase letters, digits and ``-``.
        :returns: the general memory, read-only.
        :rtype: GeneralMemory
        :raises MemoryUnsupported: if the backend has no general memory.
        """
        check_general_name(name)
        self._call.require("general", "general memory")
        call = _Call(
            backend=self._call.backend,
            use_case=self._call.use_case,
            attribution=self._call.attribution,
            scope="general",
            general_name=name,
        )
        return GeneralMemory(self._call.backend.general(name, self._credential), call)


class AdminMemory:
    """The platform's forget path. Only the platform admin client uses it."""

    def __init__(self, store: ForgetStore, call: _Call) -> None:
        """Wrap the backend's forget path."""
        self._store = store
        self._call = call

    async def forget(self, learner_id: str) -> int:
        """Delete everything stored for one learner, for every app, in one call.

        Writes for the learner are refused for as long as the backend's fence
        lasts afterwards (an hour on the deployed service). Repeating it is safe,
        and a caller that gets an error must repeat it: until it completes, the
        learner's memory is fenced but not gone.

        :param learner_id: the learner's platform user id (the Keycloak ``sub``).
        :returns: how many stored items were removed.
        :rtype: int
        """
        with self._call.span("forget", {"user_id": learner_id}) as span:
            removed = await self._store.forget(learner_id)
            if span is not None and span.is_recording():
                span.set_attribute("memory.removed_count", removed)
        return removed


class Memory:
    """An app's memory service client: a backend, plus the tokens it needs.

    Build it from configuration with ``ol_llm.memory.build_memory`` rather than
    naming a backend, so the backend can change without touching call sites.

    :param backend: the backend.
    :param exchange: turns a learner's session token into one for the service;
        ``None`` hands the session token to the backend as it is, for a backend
        without authentication.
    :param general_tokens: the general writer's client-credentials tokens.
    :param admin_tokens: the platform admin client's tokens.
    """

    def __init__(
        self,
        backend: MemoryBackend,
        *,
        exchange: TokenExchange | None = None,
        general_tokens: TokenSource | None = None,
        admin_tokens: TokenSource | None = None,
    ) -> None:
        """Hold the backend and the token sources."""
        self.backend = backend
        self._exchange = exchange
        self._general_tokens = general_tokens
        self._admin_tokens = admin_tokens

    @property
    def app(self) -> str:
        """The app this client writes as."""
        return self.backend.app

    def _call(
        self,
        use_case: str,
        attribution: Attribution,
        scope: Scope,
        general_name: str | None = None,
    ) -> _Call:
        if not _NAME.fullmatch(use_case):
            msg = (
                "use_case must be lowercase letters, digits, and hyphens, e.g. "
                f"'learner-memory', not {use_case!r}"
            )
            raise ValueError(msg)
        return _Call(
            backend=self.backend,
            use_case=use_case,
            attribution=attribution,
            scope=scope,
            general_name=general_name,
        )

    async def learner(
        self,
        session_token: str,
        *,
        use_case: str,
        attribution: Attribution = NO_ATTRIBUTION,
    ) -> LearnerMemory:
        """Return the session's learner's memory, exchanging the token now.

        Call it at the start of each request: the exchanged token is short-lived.

        :param session_token: the learner's session access token, as the gateway
            forwards it. For a backend without authentication, whatever it takes
            (the learner id, for the in-memory backend).
        :param use_case: the call site, e.g. ``tutor-recall``, for the spans.
        :param attribution: the per-call fields for the spans.
        :returns: the learner's memory.
        :rtype: LearnerMemory
        :raises MemoryUnauthenticated: if the session token is refused.
        """
        call = self._call(use_case, attribution, "learner")
        token = session_token
        if self._exchange is not None:
            with call.span("exchange"):
                token = await self._exchange.exchange(session_token)
        credential = LearnerCredential(token)
        return LearnerMemory(self.backend.learner(credential), call, credential)

    def general(
        self,
        name: str,
        *,
        use_case: str,
        attribution: Attribution = NO_ATTRIBUTION,
    ) -> GeneralMemory:
        """Return the general memory ``name`` under the app's general grant.

        :param name: e.g. ``courses``: lowercase letters, digits and ``-``.
        :param use_case: the call site, for the spans.
        :param attribution: the per-call fields for the spans.
        :returns: the general memory.
        :rtype: GeneralMemory
        :raises MemoryUnsupported: if the backend has no general memory.
        :raises ValueError: if the backend needs a general client and none is
            configured.
        """
        check_general_name(name)
        call = self._call(use_case, attribution, "general", name)
        call.require("general", "general memory")
        store = self.backend.general(name, ServiceCredential(self._general_tokens))
        return GeneralMemory(store, call)

    def admin(
        self, *, use_case: str, attribution: Attribution = NO_ATTRIBUTION
    ) -> AdminMemory:
        """Return the forget path, as the platform admin client.

        :param use_case: the call site, e.g. ``account-deletion``, for the spans.
        :param attribution: the per-call fields for the spans.
        :returns: the forget path.
        :rtype: AdminMemory
        :raises MemoryUnsupported: if the backend cannot forget.
        :raises ValueError: if the backend needs an admin client and none is
            configured.
        """
        call = self._call(use_case, attribution, "admin")
        call.require("forget", "forget")
        return AdminMemory(
            self.backend.admin(ServiceCredential(self._admin_tokens)), call
        )


class _SyncNotes:
    """The note operations of an async handle, run on the thread's ol-llm loop."""

    def __init__(self, notes: _Notes) -> None:
        self._notes = notes

    @property
    def app(self) -> str:
        """The app this handle writes as."""
        return self._notes.app

    def put(
        self, key: str, text: str, *, metadata: Mapping[str, str] | None = None
    ) -> None:
        """Store or replace this app's note ``key``; see ``LearnerMemory.put``."""
        run_coroutine(self._notes.put(key, text, metadata=metadata))

    def get(self, key: str, *, app: str | None = None) -> Note:
        """Return the note ``key`` written by ``app``; see ``LearnerMemory.get``."""
        return run_coroutine(self._notes.get(key, app=app))

    def search(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> builtins.list[Note]:
        """Return the notes most relevant to ``query``; see ``LearnerMemory.search``."""
        return run_coroutine(self._notes.search(query, limit=limit))

    def delete(self, key: str) -> None:
        """Delete this app's note ``key``; see ``LearnerMemory.delete``."""
        run_coroutine(self._notes.delete(key))

    def list(self) -> builtins.list[Note]:
        """Return every note, every app's; see ``LearnerMemory.list``."""
        return run_coroutine(self._notes.list())


class SyncGeneralMemory(_SyncNotes):
    """``GeneralMemory`` for sync code."""


class SyncLearnerMemory(_SyncNotes):
    """``LearnerMemory`` for sync code. Get a new one for each request."""

    def __init__(self, notes: LearnerMemory) -> None:
        """Wrap the async handle."""
        super().__init__(notes)
        self._learner = notes

    def general(self, name: str) -> SyncGeneralMemory:
        """Return the general memory ``name``, read-only; see ``LearnerMemory``."""
        return SyncGeneralMemory(self._learner.general(name))


class SyncAdminMemory:
    """``AdminMemory`` for sync code."""

    def __init__(self, admin: AdminMemory) -> None:
        """Wrap the async forget path."""
        self._admin = admin

    def forget(self, learner_id: str) -> int:
        """Delete everything stored for one learner; see ``AdminMemory.forget``."""
        return run_coroutine(self._admin.forget(learner_id))


class SyncMemory:
    """``Memory`` for sync code (a Django view, a Celery task).

    Every call runs on the calling thread's ol-llm event loop
    (``ol_llm.run_coroutine``), so the backend's connections are built once per
    thread and reused, rather than per call as ``asyncio.run`` would. It shares
    the wrapped ``Memory``'s backend and token sources. Do not call it from inside
    a running event loop; ``await`` the ``Memory`` there instead.

    :param memory: the async client, e.g. from ``build_memory``.
    """

    def __init__(self, memory: Memory) -> None:
        """Wrap ``memory``."""
        self.memory = memory

    @property
    def app(self) -> str:
        """The app this client writes as."""
        return self.memory.app

    def learner(
        self,
        session_token: str,
        *,
        use_case: str,
        attribution: Attribution = NO_ATTRIBUTION,
    ) -> SyncLearnerMemory:
        """Return the session's learner's memory; see ``Memory.learner``."""
        return SyncLearnerMemory(
            run_coroutine(
                self.memory.learner(
                    session_token, use_case=use_case, attribution=attribution
                )
            )
        )

    def general(
        self,
        name: str,
        *,
        use_case: str,
        attribution: Attribution = NO_ATTRIBUTION,
    ) -> SyncGeneralMemory:
        """Return the general memory ``name``; see ``Memory.general``."""
        return SyncGeneralMemory(
            self.memory.general(name, use_case=use_case, attribution=attribution)
        )

    def admin(
        self, *, use_case: str, attribution: Attribution = NO_ATTRIBUTION
    ) -> SyncAdminMemory:
        """Return the forget path; see ``Memory.admin``."""
        return SyncAdminMemory(
            self.memory.admin(use_case=use_case, attribution=attribution)
        )
