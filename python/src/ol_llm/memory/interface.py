"""What a memory backend implements, and the values that cross the interface.

A backend is one app's view of the memory service. It authorizes each call with
the credential it is handed and never takes the learner as an argument: the
learner is whoever the credential is for (spec §5.3 item 5). The library wraps
every backend in ``ol_llm.memory.Memory``, which does the token handling, the
tracing, and the capability checks, so a backend does none of them.

A backend that holds connections keeps them in an ``ol_llm.loops.LoopLocal`` with
a close function, so the runner closes them with their loop and a forked child
drops them. There is no close method on the interface.
"""

import builtins
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

Capability = Literal["search", "general", "forget"]
"""What a backend may lack. A backend without one raises ``MemoryUnsupported``
for it, through ``Memory``, rather than doing nothing."""

ALL_CAPABILITIES: frozenset[Capability] = frozenset({"search", "general", "forget"})

DEFAULT_SEARCH_LIMIT = 3
DEFAULT_TIMEOUT_SECONDS = 10.0
"""Bound on each request to the memory service."""

_KEY = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
_GENERAL_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,61}")


def check_key(key: str) -> str:
    """Return ``key`` if it is a valid note key.

    Lowercase letters, digits, ``_`` and ``-``, at most 63 characters. Keys name
    notes in the service's document ids, which it does not redact, so a key must
    never carry learner content.

    :param key: the key.
    :returns: the key.
    :rtype: str
    :raises ValueError: if it is not valid.
    """
    if not _KEY.fullmatch(key):
        msg = f"note keys are lowercase letters, digits, _ and -, not {key!r}"
        raise ValueError(msg)
    return key


def check_general_name(name: str) -> str:
    """Return ``name`` if it is a valid general memory name, e.g. ``courses``.

    :param name: the name.
    :returns: the name.
    :rtype: str
    :raises ValueError: if it is not lowercase letters, digits and ``-``, at most
        62 characters.
    """
    if not _GENERAL_NAME.fullmatch(name):
        msg = f"general memory names are lowercase letters, digits and -, not {name!r}"
        raise ValueError(msg)
    return name


@dataclass(frozen=True, kw_only=True)
class Note:
    """One stored note.

    :param key: the key the writing app stored it under.
    :param text: the note, as stored. The deployed service redacts emails, phone
        numbers and secrets before storing it.
    :param app: the app that wrote it.
    :param metadata: what the writer stored with it.
    :param score: the relevance score, on search results only.
    """

    key: str
    text: str
    app: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    score: float | None = None


class TokenSource(Protocol):
    """Where a service credential's bearer tokens come from."""

    async def token(self, *, refresh: bool = False) -> str:
        """Return a current token.

        :param refresh: fetch a new one even if the cached one has not expired;
            asked once after the service answers 401.
        :returns: the token.
        :rtype: str
        """
        ...


@dataclass(frozen=True)
class LearnerCredential:
    """A learner's credential: a token issued for the memory service.

    :param token: on the deployed service, a token exchanged from the learner's
        session (``ol_llm.memory.KeycloakTokenExchange``). A backend without
        authentication defines what it is, e.g. the in-memory backend's
        ``identify``.
    """

    token: str = field(repr=False)


@dataclass(frozen=True)
class ServiceCredential:
    """A client-credentials grant: the general writer's, or the platform admin's.

    :param tokens: where its tokens come from; ``None`` for a backend without
        authentication.
    """

    tokens: TokenSource | None


class NoteStore(Protocol):
    """One bank's notes, as one app sees them through one credential.

    Every app reads every app's notes; an app writes and deletes only its own.
    There is no update in place: ``put`` again with the same key replaces a note.
    """

    async def put(
        self, key: str, text: str, *, metadata: Mapping[str, str] | None = None
    ) -> None:
        """Store or replace this app's note ``key``.

        :raises MemoryFenced: while the learner's memory is being forgotten.
        :raises MemoryForbidden: if the credential may not write here.
        """
        ...

    async def get(self, key: str, *, app: str | None = None) -> Note:
        """Return the note ``key`` written by ``app``, this app when omitted.

        :raises MemoryNotFound: if there is no such note.
        """
        ...

    async def search(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> builtins.list[Note]:
        """Return up to ``limit`` notes most relevant to ``query``, best first."""
        ...

    async def delete(self, key: str) -> None:
        """Delete this app's note ``key``. Deleting a missing note is not an error.

        Allowed while the learner's memory is being forgotten.
        """
        ...

    async def list(self) -> builtins.list[Note]:
        """Return every note in the bank, every app's."""
        ...


class ForgetStore(Protocol):
    """The platform's forget path. Only the platform admin client holds one."""

    async def forget(self, learner_id: str) -> int:
        """Delete everything stored for one learner, for every app, in one call.

        A write for the learner after this returns is refused for as long as the
        backend's fence lasts, so in-flight work cannot bring the memory back.
        Repeating a forget is safe.

        :param learner_id: the learner's platform user id (the Keycloak ``sub``).
        :returns: how many stored items were removed.
        :rtype: int
        """
        ...


class MemoryBackend(Protocol):
    """One app's client of a memory service.

    :param name: the name the backend is registered under, for traces.
    :param app: the app it writes as.
    :param capabilities: what it implements; see ``Capability``.
    """

    name: str
    app: str
    capabilities: frozenset[Capability]

    def learner(self, credential: LearnerCredential) -> NoteStore:
        """Return the credential's learner's notes."""
        ...

    def general(
        self, name: str, credential: LearnerCredential | ServiceCredential
    ) -> NoteStore:
        """Return the general memory ``name``.

        Any app credential reads it. Writes need the general grant, which only a
        ``ServiceCredential`` can carry; with a learner credential they raise
        ``MemoryForbidden``.
        """
        ...

    def admin(self, credential: ServiceCredential) -> ForgetStore:
        """Return the forget path, authorized by the platform admin's credential."""
        ...
