"""The memory backend for the deployed Hindsight service, over its HTTP API.

The service's own extensions enforce everything here (ol-infrastructure
``applications/hindsight``); this client only builds requests they accept. A
learner's bank is ``u-<sub>``, where ``sub`` is the learner token's subject. A
general memory ``<name>`` is the bank ``g-<name>``. Every write carries the tag
``app:<app>`` and the document id ``<app>.<key>``, so a second ``put`` of a key
replaces the first, and the server refuses one app changing another's note.

It sends with ``hindsight-client``'s generated API (the ``memory`` extra), which
takes the bearer token per call, so one ``ApiClient`` per event loop serves every
learner. The generated client's ``aiohttp`` transport and its retry layer are not
used (see ``_Transport``), nor is the high-level ``Hindsight`` class, whose recall
retries on 429 and 503: a request is sent once, and only ``forget`` starts over.
"""

import asyncio
import base64
import builtins
import dataclasses
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from typing import Any, TypeVar

import httpx2
from hindsight_client_api.api.banks_api import BanksApi
from hindsight_client_api.api.documents_api import DocumentsApi
from hindsight_client_api.api.memory_api import MemoryApi
from hindsight_client_api.api.operations_api import OperationsApi
from hindsight_client_api.api_client import ApiClient
from hindsight_client_api.configuration import Configuration
from hindsight_client_api.exceptions import ApiException
from hindsight_client_api.models.cancel_operation_response import (
    CancelOperationResponse,
)
from hindsight_client_api.models.content import Content
from hindsight_client_api.models.list_memory_units_response import (
    ListMemoryUnitsResponse,
)
from hindsight_client_api.models.memory_item import MemoryItem
from hindsight_client_api.models.recall_request import RecallRequest
from hindsight_client_api.models.retain_request import RetainRequest

from ol_llm.loops import LoopLocal
from ol_llm.memory.errors import (
    MemoryNotFound,
    MemoryStoreError,
    MemoryTimeout,
    MemoryUnauthenticated,
    from_status,
    from_transport,
)
from ol_llm.memory.interface import (
    ALL_CAPABILITIES,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,
    ForgetStore,
    LearnerCredential,
    Note,
    NoteStore,
    ServiceCredential,
    TokenSource,
)

_R = TypeVar("_R")

log = logging.getLogger(__name__)

LEARNER_BANK_PREFIX = "u-"
GENERAL_BANK_PREFIX = "g-"
LEARNER_HEADER = "x-ol-learner"
"""Where the admin client names the learner it forgets."""

DEFAULT_RECALL_MAX_TOKENS = 4096
"""Hindsight's own default token budget for one recall."""
MEMORIES_PAGE_SIZE = 100
OPERATIONS_PAGE_SIZE = 100
FORGET_ATTEMPTS = 5
FORGET_BACKOFF_SECONDS = 2.0
DRAIN_POLL_SECONDS = 1.0
DRAIN_TIMEOUT_SECONDS = 600.0
_NOT_FOUND = 404
_UNAUTHORIZED = 401
_CONFLICT = 409

_LEARNER_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)

# Patched in tests so the forget's backoff and drain polling do not wait.
_sleep = asyncio.sleep

Credential = str | TokenSource
Send = Callable[[ApiClient, dict[str, Any]], Awaitable[_R]]


def learner_bank(learner_id: str) -> str:
    """Return a learner's bank id.

    :param learner_id: the learner's Keycloak user id.
    :returns: the bank id.
    :rtype: str
    """
    return f"{LEARNER_BANK_PREFIX}{learner_id}"


def token_subject(token: str) -> str:
    """Return a learner token's ``sub`` without verifying the token.

    Only to name the bank: the service verifies the token and refuses any bank
    that is not its subject's.

    :param token: the exchanged learner token.
    :returns: the subject.
    :rtype: str
    :raises MemoryUnauthenticated: if it is not a JWT or carries no ``sub``.
    """
    try:
        payload = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
    except (IndexError, ValueError) as exc:
        msg = "the learner token is not a JWT"
        raise MemoryUnauthenticated(msg, operation="learner") from exc
    if not isinstance(subject := claims.get("sub"), str) or not subject:
        msg = (
            "the learner token carries no sub; exchange the session token with "
            "the memory audience scope"
        )
        raise MemoryUnauthenticated(msg, operation="learner")
    return subject


class _Answer:
    """An ``httpx2`` response in the shape the generated client reads."""

    def __init__(self, response: httpx2.Response) -> None:
        self.response = response
        self.status = response.status_code
        self.reason = response.reason_phrase
        self.data = response.content

    async def read(self) -> bytes:
        return self.data

    def getheaders(self) -> httpx2.Headers:
        return self.response.headers

    def getheader(self, name: str, default: str | None = None) -> str | None:
        value: str | None = self.response.headers.get(name, default)
        return value


class _Transport:
    """The generated client's transport, on ``httpx2`` in place of its ``aiohttp``.

    ``ApiClient`` sends every request through ``rest_client.request`` and closes it
    with ``rest_client.close``. Its own ``aiohttp`` session logs "Unclosed client
    session" when it is dropped unclosed, which is what happens to a value built on
    a loop ol-llm does not close (``asyncio.run`` per call, a thread pool's threads
    exiting). An ``httpx2`` client is dropped quietly, as every other client in
    ol-llm is.
    """

    def __init__(self, http: httpx2.AsyncClient) -> None:
        self.http = http

    async def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str] | None = None,
        body: object = None,
        post_params: object = None,
        _request_timeout: float | None = None,
    ) -> _Answer:
        """Send one JSON request, as the generated ``RESTClientObject.request``."""
        if post_params:
            msg = "the Hindsight backend sends JSON bodies only, not form posts"
            raise ValueError(msg)
        # The generated methods name the content type themselves for a body
        # ("Content-Type"), and headers compare case-insensitively only once
        # they are httpx2.Headers, so set the default there, not in a dict.
        sent = httpx2.Headers(dict(headers or {}))
        if body is not None:
            sent.setdefault("content-type", "application/json")
        response = await self.http.request(
            method,
            url,
            headers=sent,
            content=None if body is None else json.dumps(body).encode(),
            timeout=_request_timeout,
        )
        return _Answer(response)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http.aclose()


def _api_client(base_url: str, http: httpx2.AsyncClient) -> ApiClient:
    client = ApiClient(Configuration(host=base_url.rstrip("/")))
    # ApiClient only calls request() and close() on it, which _Transport provides.
    client.rest_client = _Transport(http)  # type: ignore[assignment]
    return client


async def _close(client: ApiClient) -> None:
    await client.close()  # type: ignore[no-untyped-call]


class _Api:
    """Sends with the running loop's ``ApiClient`` and turns failures into errors."""

    def __init__(
        self, base_url: str, *, timeout_seconds: float, http: httpx2.AsyncClient | None
    ) -> None:
        self._timeout = timeout_seconds
        self._given = None if http is None else _api_client(base_url, http)
        self._clients = LoopLocal(
            lambda: _api_client(base_url, httpx2.AsyncClient()), _close
        )

    async def send(
        self,
        send: "Send[_R]",
        *,
        operation: str,
        credential: Credential,
        headers: Mapping[str, str] | None = None,
        missing_ok: bool = False,
    ) -> _R | None:
        """Send once, and once more with a new token after a 401 from a token source.

        :param send: calls one generated API method with the client and the
            per-call arguments (the bearer token, the timeout, extra headers).
        :returns: what the method returns, or ``None`` for a 404 when
            ``missing_ok``.
        :raises MemoryStoreError: for any other failure.
        """
        refreshes = (False,) if isinstance(credential, str) else (False, True)
        for refresh in refreshes:
            token = (
                credential
                if isinstance(credential, str)
                else await credential.token(refresh=refresh)
            )
            # The generated methods add to the headers dict they are given.
            call: dict[str, Any] = {
                "authorization": f"Bearer {token}",
                "_request_timeout": self._timeout,
                "_headers": dict(headers or {}),
            }
            try:
                return await send(self._given or self._clients.get(), call)
            except ApiException as exc:
                status = int(exc.status or 0)
                if status == _UNAUTHORIZED and not refresh and len(refreshes) > 1:
                    continue
                if missing_ok and status == _NOT_FOUND:
                    return None
                # Not the body: a 422 echoes the request, note text included,
                # and this message reaches logs and Sentry.
                message = f"{operation}: {status} {exc.reason}"
                raise from_status(status, message, operation=operation) from exc
            except httpx2.TransportError as exc:
                raise from_transport(exc, operation=operation) from exc
        raise AssertionError  # pragma: no cover - the loop returns or raises


class _DrainTimedOut(MemoryTimeout):
    """The forget's drain outlasted ``DRAIN_TIMEOUT_SECONDS``; forget does not retry.

    Still a ``MemoryTimeout``, so the caller repeats the forget later, as it must
    after any failure.
    """


class _NotANote(Exception):  # noqa: N818 - private, never raised to callers
    """A memory unit no app wrote through this library, e.g. a consolidated one."""


def _writer(tags: Sequence[str] | None) -> str:
    apps = [tag.removeprefix("app:") for tag in tags or [] if tag.startswith("app:")]
    if len(apps) != 1:
        msg = f"{len(apps)} app tags, not exactly one"
        raise _NotANote(msg)
    return apps[0]


def _key(document_id: str | None, app: str) -> str:
    if not (document_id or "").startswith(f"{app}."):
        msg = f"a document id outside {app}.*"
        raise _NotANote(msg)
    return (document_id or "").removeprefix(f"{app}.")


def _chunk_index(chunk_id: str | None) -> int:
    """Return a unit's place in its note, from the server's chunk id.

    In the ``chunks`` extraction mode the deployed service runs, a note over the
    chunk size (3000 characters by default) is stored as one unit per chunk, and
    each unit's chunk id ends in ``_<chunk index>`` whatever the bank and document
    ids hold (hindsight-api ``engine/chunk_ids.py``). A unit without one sorts
    first.
    """
    index = (chunk_id or "").rpartition("_")[2]
    return int(index) if index.isdigit() else 0


def _metadata(metadata: Mapping[str, object] | None) -> dict[str, str]:
    return {str(k): str(v) for k, v in (metadata or {}).items()}


class _Unit:
    """What a note is built from, from a listed memory unit or a recall result."""

    def __init__(self, item: Any) -> None:  # noqa: ANN401 - two generated models
        self.app = _writer(item.tags)
        self.key = _key(item.document_id, self.app)
        self.index = _chunk_index(item.chunk_id)
        self.text = str(item.text or "")
        self.metadata = _metadata(item.metadata)


def _readable(items: Sequence[Any]) -> builtins.list[tuple[Any, _Unit]]:
    """Pair each item with its unit, skipping (and logging) any that is not a note.

    One such unit (consolidation, were it turned on, or a write that bypassed the
    library) must not fail every read of the learner's memory. Only its id is
    logged, never its text.
    """
    readable = []
    for item in items:
        try:
            readable.append((item, _Unit(item)))
        except _NotANote as exc:
            log.warning("Skipping memory unit %s: %s", item.id, exc)
    return readable


def _notes(units: builtins.list[_Unit]) -> builtins.list[Note]:
    """Group memory units into notes, one per document, in the order listed.

    A chunked note's units are joined with newlines in chunk order. The service
    strips whitespace at chunk boundaries, so a reassembled note has its text in
    order but not always the original whitespace where it was split.
    """
    grouped: dict[tuple[str, str], builtins.list[_Unit]] = {}
    for unit in units:
        grouped.setdefault((unit.app, unit.key), []).append(unit)
    return [
        Note(
            key=key,
            text="\n".join(
                unit.text for unit in sorted(units, key=lambda unit: unit.index)
            ),
            app=app,
            metadata=units[0].metadata,
        )
        for (app, key), units in grouped.items()
    ]


class _Notes:
    def __init__(
        self,
        api: _Api,
        *,
        app: str,
        bank: str,
        credential: Credential,
        recall_max_tokens: int,
    ) -> None:
        self._api = api
        self._app = app
        self._bank = bank
        self._credential = credential
        self._recall_max_tokens = recall_max_tokens

    async def put(
        self, key: str, text: str, *, metadata: Mapping[str, str] | None = None
    ) -> None:
        item = MemoryItem(
            content=Content(text),
            document_id=f"{self._app}.{key}",
            tags=[f"app:{self._app}"],
            metadata=dict(metadata) if metadata else None,
        )
        request = RetainRequest(items=[item])
        await self._api.send(
            lambda client, call: MemoryApi(client).retain_memories(
                self._bank, request, **call
            ),
            operation="put",
            credential=self._credential,
        )

    async def _page(
        self,
        client: ApiClient,
        call: dict[str, Any],
        *,
        document_id: str | None,
        offset: int,
    ) -> ListMemoryUnitsResponse:
        return await MemoryApi(client).list_memories(
            self._bank,
            document_id=document_id,
            limit=MEMORIES_PAGE_SIZE,
            offset=offset,
            **call,
        )

    async def _units(
        self, operation: str, document_id: str | None = None
    ) -> builtins.list[_Unit] | None:
        """Return every memory unit the listing matches, or ``None`` with no bank."""
        units: builtins.list[_Unit] = []
        listed = 0
        while True:
            page = await self._api.send(
                partial(self._page, document_id=document_id, offset=listed),
                operation=operation,
                credential=self._credential,
                missing_ok=True,
            )
            if page is None:
                return None
            listed += len(page.items)
            units.extend(unit for _, unit in _readable(page.items))
            if not page.items or listed >= page.total:
                return units

    async def get(self, key: str, *, app: str | None = None) -> Note:
        writer = app or self._app
        units = await self._units("get", f"{writer}.{key}")
        if not units:
            msg = f"no note {key!r} from {writer}"
            raise MemoryNotFound(msg, operation="get", status_code=404)
        [note] = _notes(units)
        return note

    async def search(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> builtins.list[Note]:
        request = RecallRequest(query=query, max_tokens=self._recall_max_tokens)
        response = await self._api.send(
            lambda client, call: MemoryApi(client).recall_memories(
                self._bank, request, **call
            ),
            operation="search",
            credential=self._credential,
            missing_ok=True,
        )
        if response is None:
            return []
        # Recall answers with units, best first, and a chunked note's chunks are
        # separate units, so keep each note's best score, then read the matched
        # notes whole.
        best: dict[tuple[str, str], float | None] = {}
        for result, unit in _readable(response.results):
            if (unit.app, unit.key) not in best:
                best[unit.app, unit.key] = (
                    None if result.scores is None else float(result.scores.final)
                )
            if len(best) == limit:
                break
        notes = await asyncio.gather(
            *(self._units("search", f"{app}.{key}") for app, key in best)
        )
        # A note deleted between the recall and the read is left out.
        return [
            dataclasses.replace(note, score=score)
            for score, units in zip(best.values(), notes, strict=True)
            for note in _notes(units or [])
        ]

    async def delete(self, key: str) -> None:
        await self._api.send(
            lambda client, call: DocumentsApi(client).delete_document(
                self._bank, f"{self._app}.{key}", **call
            ),
            operation="delete",
            credential=self._credential,
            missing_ok=True,
        )

    async def list(self) -> builtins.list[Note]:
        return _notes(await self._units("list") or [])


async def _cancel(
    client: ApiClient, call: dict[str, Any], *, bank: str, operation_id: str
) -> CancelOperationResponse:
    return await OperationsApi(client).cancel_operation(bank, operation_id, **call)


class _Admin:
    def __init__(self, api: _Api, tokens: TokenSource) -> None:
        self._api = api
        self._tokens = tokens

    async def forget(self, learner_id: str) -> int:
        """Delete everything for one learner, in the service's two-phase sequence.

        A bank delete cascades away the bank's operation rows, the only record of
        work in flight, so first clear the bank's memories, which raises the
        service's fence and keeps the bank row; cancel pending operations and wait
        for processing ones; then delete the bank. The sequence is idempotent, so a
        failure that can clear up is retried from the start rather than left half
        done. Until the bank is deleted, the fence is all that keeps the learner's
        memory from coming back, so a caller that still gets an error must retry.
        """
        if not _LEARNER_ID.fullmatch(learner_id):
            msg = f"learner ids are Keycloak user ids (UUIDs), not {learner_id!r}"
            raise ValueError(msg)
        for attempt in range(FORGET_ATTEMPTS):
            try:
                return await self._forget(learner_id)
            except MemoryStoreError as exc:
                # A drain that timed out already waited DRAIN_TIMEOUT_SECONDS;
                # starting over would wait that long again, up to five times.
                last = attempt == FORGET_ATTEMPTS - 1
                if not exc.retryable or isinstance(exc, _DrainTimedOut) or last:
                    raise
                await _sleep(FORGET_BACKOFF_SECONDS * 2**attempt)
        raise AssertionError  # pragma: no cover - the loop returns or raises

    async def _send(self, send: "Send[_R]", learner_id: str) -> _R | None:
        return await self._api.send(
            send,
            operation="forget",
            credential=self._tokens,
            headers={LEARNER_HEADER: learner_id},
            missing_ok=True,
        )

    async def _forget(self, learner_id: str) -> int:
        bank = learner_bank(learner_id)
        removed = 0
        cleared = await self._send(
            lambda client, call: MemoryApi(client).clear_bank_memories(bank, **call),
            learner_id,
        )
        if cleared is not None:
            removed += cleared.deleted_count or 0
            await self._drain(bank, learner_id)
        # An earlier attempt may have deleted the bank before its answer was lost.
        deleted = await self._send(
            lambda client, call: BanksApi(client).delete_bank(bank, **call),
            learner_id,
        )
        if deleted is None:
            return removed
        return removed + (deleted.deleted_count or 0)

    async def _drain(self, bank: str, learner_id: str) -> None:
        deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
        while True:
            pending = await self._operations(bank, learner_id, "pending")
            for operation_id in pending:
                try:
                    await self._send(
                        partial(_cancel, bank=bank, operation_id=operation_id),
                        learner_id,
                    )
                except MemoryStoreError as exc:
                    # A worker claimed it between the listing and the cancel. It is
                    # processing now, and the loop waits for it like any other.
                    if exc.status_code != _CONFLICT:
                        raise
            if not pending and not await self._operations(
                bank, learner_id, "processing"
            ):
                return
            if time.monotonic() > deadline:
                msg = f"{bank} still had work in flight after {DRAIN_TIMEOUT_SECONDS}s"
                raise _DrainTimedOut(msg, operation="forget")
            await _sleep(DRAIN_POLL_SECONDS)

    async def _operations(
        self, bank: str, learner_id: str, status: str
    ) -> builtins.list[str]:
        listing = await self._send(
            lambda client, call: OperationsApi(client).list_operations(
                bank, status=status, limit=OPERATIONS_PAGE_SIZE, **call
            ),
            learner_id,
        )
        return [] if listing is None else [op.id for op in listing.operations]


class HindsightBackend:
    """One app's client of the deployed Hindsight service.

    :param base_url: the service's URL, e.g.
        ``http://hindsight.hindsight.svc.cluster.local:8888``.
    :param app: the app's name as the service maps its exchange client, e.g.
        ``learn-ai``. It prefixes the app's document ids and tags its writes.
    :param timeout_seconds: bound on each request.
    :param recall_max_tokens: the token budget of one search on the service.
    :param http: the HTTP client to send with, for tests; by default one per event
        loop, closed with the loop when ol-llm owns it.
    """

    name = "hindsight"
    capabilities = ALL_CAPABILITIES

    def __init__(
        self,
        base_url: str,
        *,
        app: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        recall_max_tokens: int = DEFAULT_RECALL_MAX_TOKENS,
        http: httpx2.AsyncClient | None = None,
    ) -> None:
        """Hold the configuration; nothing is connected until the first request."""
        self.app = app
        self._recall_max_tokens = recall_max_tokens
        self._api = _Api(base_url, timeout_seconds=timeout_seconds, http=http)

    def _store(self, bank: str, credential: Credential) -> NoteStore:
        return _Notes(
            self._api,
            app=self.app,
            bank=bank,
            credential=credential,
            recall_max_tokens=self._recall_max_tokens,
        )

    def learner(self, credential: LearnerCredential) -> NoteStore:
        """Return the bank of the learner the token was exchanged for."""
        return self._store(
            learner_bank(token_subject(credential.token)), credential.token
        )

    def general(
        self, name: str, credential: LearnerCredential | ServiceCredential
    ) -> NoteStore:
        """Return the general bank ``g-<name>``."""
        if isinstance(credential, LearnerCredential):
            return self._store(f"{GENERAL_BANK_PREFIX}{name}", credential.token)
        return self._store(f"{GENERAL_BANK_PREFIX}{name}", _require(credential))

    def admin(self, credential: ServiceCredential) -> ForgetStore:
        """Return the forget path, as the platform admin client."""
        return _Admin(self._api, _require(credential))


def _require(credential: ServiceCredential) -> TokenSource:
    if credential.tokens is None:
        msg = "the Hindsight backend needs a Keycloak client for service calls"
        raise ValueError(msg)
    return credential.tokens
