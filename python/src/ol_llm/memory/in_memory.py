"""A memory backend held in process, for tests and local development.

It enforces what the deployed service enforces, so code tested against it behaves
the same against Hindsight: one memory per learner shared by every app, writes
stamped with the writing app, general memory written only under a service
credential, and a forget that fences the learner's memory against later writes.

Search ranks by the words a note shares with the query. It is not semantic.

There is no authentication. The learner is ``identify(token)``, the token itself
by default, so a test hands the learner's id where a session token goes.
"""

import builtins
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ol_llm.memory.errors import MemoryFenced, MemoryForbidden, MemoryNotFound
from ol_llm.memory.interface import (
    ALL_CAPABILITIES,
    DEFAULT_SEARCH_LIMIT,
    Capability,
    ForgetStore,
    LearnerCredential,
    Note,
    NoteStore,
    ServiceCredential,
)

DEFAULT_FENCE_SECONDS = 3600
"""How long a forgotten learner's memory refuses writes, as on the deployed service."""

_WORD = re.compile(r"\w+")


def _same(token: str) -> str:
    return token


@dataclass
class _Stored:
    text: str
    metadata: Mapping[str, str]


@dataclass
class InMemoryStore:
    """The state every app's ``InMemoryBackend`` in a process shares.

    :param fence_seconds: how long a forget refuses writes for the learner.
    """

    fence_seconds: float = DEFAULT_FENCE_SECONDS
    _banks: dict[str, dict[tuple[str, str], _Stored]] = field(
        default_factory=dict, init=False, repr=False
    )
    _fences: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def clear(self) -> None:
        """Forget everything, fences included."""
        with self._lock:
            self._banks.clear()
            self._fences.clear()

    def write(self, bank: str, app: str, key: str, stored: _Stored) -> None:
        """Store or replace a note, unless the bank is fenced.

        :raises MemoryFenced: if the bank is fenced.
        """
        with self._lock:
            if time.monotonic() < self._fences.get(bank, 0.0):
                msg = "this learner's memory is being forgotten"
                raise MemoryFenced(msg, operation="put", status_code=409)
            self._banks.setdefault(bank, {})[app, key] = stored

    def remove(self, bank: str, app: str, key: str) -> None:
        """Delete a note if it is there."""
        with self._lock:
            self._banks.get(bank, {}).pop((app, key), None)

    def notes(self, bank: str) -> dict[tuple[str, str], _Stored]:
        """Return a bank's notes by app and key, oldest write first."""
        with self._lock:
            return dict(self._banks.get(bank, {}))

    def forget(self, bank: str) -> int:
        """Delete a bank and fence it; return how many notes it held."""
        with self._lock:
            removed = len(self._banks.pop(bank, {}))
            self._fences[bank] = time.monotonic() + self.fence_seconds
        return removed


_default_store = InMemoryStore()


def default_store() -> InMemoryStore:
    """Return the store a configured in-memory backend uses, one per process.

    :returns: the store.
    :rtype: InMemoryStore
    """
    return _default_store


def _learner_bank(learner_id: str) -> str:
    return f"learner:{learner_id}"


class _Notes:
    def __init__(
        self, store: InMemoryStore, *, app: str, bank: str, writable: bool
    ) -> None:
        self._store = store
        self._app = app
        self._bank = bank
        self._writable = writable

    def _check_write(self, operation: str) -> None:
        if not self._writable:
            msg = "writing general memory needs the general grant"
            raise MemoryForbidden(msg, operation=operation, status_code=403)

    async def put(
        self, key: str, text: str, *, metadata: Mapping[str, str] | None = None
    ) -> None:
        self._check_write("put")
        # Removed first, so a replaced note lists as the newest.
        self._store.remove(self._bank, self._app, key)
        self._store.write(
            self._bank,
            self._app,
            key,
            _Stored(text=text, metadata=dict(metadata or {})),
        )

    async def get(self, key: str, *, app: str | None = None) -> Note:
        writer = app or self._app
        if (stored := self._store.notes(self._bank).get((writer, key))) is None:
            msg = f"no note {key!r} from {writer}"
            raise MemoryNotFound(msg, operation="get", status_code=404)
        return _note(writer, key, stored)

    async def search(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> builtins.list[Note]:
        words = set(_WORD.findall(query.lower()))
        scored = [
            _note(app, key, stored, score=len(shared) / len(words))
            for (app, key), stored in self._store.notes(self._bank).items()
            if (shared := words & set(_WORD.findall(stored.text.lower())))
        ]
        scored.sort(key=lambda note: -(note.score or 0.0))
        return scored[:limit]

    async def delete(self, key: str) -> None:
        self._check_write("delete")
        self._store.remove(self._bank, self._app, key)

    async def list(self) -> builtins.list[Note]:
        return [
            _note(app, key, stored)
            for (app, key), stored in self._store.notes(self._bank).items()
        ]


def _note(app: str, key: str, stored: _Stored, score: float | None = None) -> Note:
    return Note(
        key=key, text=stored.text, app=app, metadata=dict(stored.metadata), score=score
    )


class _Admin:
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def forget(self, learner_id: str) -> int:
        return self._store.forget(_learner_bank(learner_id))


class InMemoryBackend:
    """One app's view of an ``InMemoryStore``.

    :param app: the app it writes as.
    :param store: the shared state; the process's default store when omitted.
    :param identify: maps a learner credential's token to the learner id.
    :param capabilities: what it implements, to test a call site against a
        backend that lacks something.
    """

    name = "in-memory"

    def __init__(
        self,
        app: str,
        *,
        store: InMemoryStore | None = None,
        identify: Callable[[str], str] = _same,
        capabilities: frozenset[Capability] = ALL_CAPABILITIES,
    ) -> None:
        """Hold the store; nothing is connected."""
        self.app = app
        self.capabilities = capabilities
        self._store = store if store is not None else _default_store
        self._identify = identify

    def learner(self, credential: LearnerCredential) -> NoteStore:
        """Return the credential's learner's notes."""
        return _Notes(
            self._store,
            app=self.app,
            bank=_learner_bank(self._identify(credential.token)),
            writable=True,
        )

    def general(
        self, name: str, credential: LearnerCredential | ServiceCredential
    ) -> NoteStore:
        """Return a general memory; only a service credential writes it."""
        return _Notes(
            self._store,
            app=self.app,
            bank=f"general:{name}",
            writable=isinstance(credential, ServiceCredential),
        )

    def admin(self, credential: ServiceCredential) -> ForgetStore:  # noqa: ARG002
        """Return the forget path."""
        return _Admin(self._store)
