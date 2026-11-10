"""The Hindsight backend: request shapes and failures against a scripted service,
and the conformance checks against a live Hindsight when one is configured.

The live run is opt-in: set ``OL_LLM_MEMORY_IT_CONFIG`` to a JSON file shaped like
``LiveConfig`` below and run ``pytest -m hindsight``. It creates Keycloak users,
writes notes, and forgets learners, which fences their banks for an hour, so
point it only at a disposable stack such as ol-infrastructure's
``scripts/hindsight/compose.yaml`` harness, never at QA or Production.
"""

import asyncio
import base64
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import httpx
import pytest
from conftest import record_sleeps
from ol_llm import run_coroutine
from ol_llm.memory import (
    ClientCredentialsTokens,
    HindsightBackend,
    KeycloakExchangeClient,
    KeycloakTokenExchange,
    LearnerCredential,
    Memory,
    MemoryFenced,
    MemoryForbidden,
    MemoryNotFound,
    MemoryRejected,
    MemoryTimeout,
    MemoryUnauthenticated,
    MemoryUnavailable,
    ServiceCredential,
)
from ol_llm.memory import hindsight as hindsight_module
from ol_llm.memory.conformance import CHECKS, Check, ConformanceLearner
from ol_llm.tracing import KeycloakClient

_T = TypeVar("_T")

BASE = "http://hindsight.test:8888"
LEARNER = "0b7e1c52-9a52-4c43-8f0e-6a1f3f2f9d11"
BANK = f"/v1/default/banks/u-{LEARNER}"


def run(coroutine: Awaitable[_T]) -> _T:
    async def wait() -> _T:
        return await coroutine

    return asyncio.run(wait())


def jwt(claims: dict[str, object]) -> str:
    def part(value: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return f"{part({'alg': 'RS256'})}.{part(claims)}.signature"


LEARNER_TOKEN = jwt({"sub": LEARNER, "azp": "ol-learn-ai-client"})

Reply = httpx.Response | Callable[[httpx.Request], httpx.Response]


@dataclass
class Service:
    """Answers each (method, path) from a script, in turn, and records requests.

    It sits under the backend's HTTP client, so everything above it (the generated
    client's serialization, deserialization and exceptions) is the real thing.
    """

    script: dict[tuple[str, str], list[Reply]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def on(self, method: str, path: str, *replies: Reply) -> "Service":
        self.script.setdefault((method, path), []).extend(replies)
        return self

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        replies = self.script.get((request.method, request.url.path))
        assert replies, f"unscripted {request.method} {request.url.path}"
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        return reply(request) if callable(reply) else reply

    def backend(self, app: str = "learn-ai") -> HindsightBackend:
        return HindsightBackend(
            BASE,
            app=app,
            recall_max_tokens=512,
            http=httpx.AsyncClient(transport=httpx.MockTransport(self)),
        )

    def seen(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]


def unit(
    document_id: str, text: str, tags: list[str] | None = None, **extra: object
) -> dict[str, object]:
    writer = document_id.split(".", maxsplit=1)[0]
    return {
        "id": str(uuid.uuid4()),
        "document_id": document_id,
        "text": text,
        "tags": [f"app:{writer}"] if tags is None else tags,
        **extra,
    }


def page(*items: dict[str, object], total: int | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "items": list(items),
            "total": len(items) if total is None else total,
            "limit": hindsight_module.MEMORIES_PAGE_SIZE,
            "offset": 0,
        },
    )


RETAINED = httpx.Response(
    200,
    json={"success": True, "bank_id": f"u-{LEARNER}", "items_count": 1, "async": False},
)


class Tokens:
    def __init__(self, *tokens: str) -> None:
        self.tokens = list(tokens)
        self.refreshes: list[bool] = []

    async def token(self, *, refresh: bool = False) -> str:
        self.refreshes.append(refresh)
        if refresh:
            self.tokens.pop(0)
        return self.tokens[0]


@pytest.fixture(autouse=True)
def no_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    return record_sleeps(monkeypatch, hindsight_module)


def learner_notes(service: Service, app: str = "learn-ai") -> Any:  # noqa: ANN401
    return service.backend(app).learner(LearnerCredential(LEARNER_TOKEN))


def test_put_retains_under_the_apps_document_id_and_tag() -> None:
    service = Service().on("POST", f"{BANK}/memories", RETAINED)
    run(learner_notes(service).put("pace", "Weekends.", metadata={"source": "chat"}))

    [request] = service.requests
    assert request.headers["authorization"] == f"Bearer {LEARNER_TOKEN}"
    assert request.headers.get_list("content-type") == ["application/json"]
    assert json.loads(request.content) == {
        "items": [
            {
                "content": "Weekends.",
                "document_id": "learn-ai.pace",
                "tags": ["app:learn-ai"],
                "metadata": {"source": "chat"},
                "resolve_entities": True,
            }
        ],
        "async": False,
    }
    assert request.extensions["timeout"]["read"] == (
        hindsight_module.DEFAULT_TIMEOUT_SECONDS
    )


def test_list_reads_every_page_and_groups_units_by_note() -> None:
    service = Service().on(
        "GET",
        f"{BANK}/memories/list",
        page(
            unit("learn-ai.pace", "Part one.", metadata={"source": "chat"}),
            unit("mit-learn.goal", "A certificate."),
            total=3,
        ),
        page(unit("learn-ai.pace", "Part two."), total=3),
    )
    notes = run(learner_notes(service).list())

    assert [(n.app, n.key, n.text, dict(n.metadata)) for n in notes] == [
        ("learn-ai", "pace", "Part one.\nPart two.", {"source": "chat"}),
        ("mit-learn", "goal", "A certificate.", {}),
    ]
    offsets = [r.url.params["offset"] for r in service.requests]
    assert offsets == ["0", "2"]


def test_an_empty_memory_lists_and_searches_as_empty() -> None:
    missing = httpx.Response(404, json={"detail": "bank not found"})
    service = (
        Service()
        .on("GET", f"{BANK}/memories/list", missing)
        .on("POST", f"{BANK}/memories/recall", missing)
    )
    notes = learner_notes(service)
    assert run(notes.list()) == []
    assert run(notes.search("anything")) == []
    with pytest.raises(MemoryNotFound):
        run(notes.get("pace"))


def test_get_reads_one_document() -> None:
    service = Service().on(
        "GET", f"{BANK}/memories/list", page(unit("mit-learn.goal", "A certificate."))
    )
    note = run(learner_notes(service).get("goal", app="mit-learn"))

    assert (note.app, note.key, note.text) == ("mit-learn", "goal", "A certificate.")
    assert service.requests[0].url.params["document_id"] == "mit-learn.goal"


def test_get_of_a_missing_note_is_not_found() -> None:
    service = Service().on("GET", f"{BANK}/memories/list", page())
    with pytest.raises(MemoryNotFound) as raised:
        run(learner_notes(service).get("pace"))
    assert raised.value.operation == "get"


def by_document(**units: list[dict[str, object]]) -> Reply:
    """Answer a units listing with the units of the document it asks for, or all."""

    def answer(request: httpx.Request) -> httpx.Response:
        if (document_id := request.url.params.get("document_id")) is None:
            return page(*[u for listed in units.values() for u in listed])
        return page(*units.get(document_id, []))

    return answer


def chunk(
    document_id: str, text: str, index: int, **extra: object
) -> dict[str, object]:
    """One chunk of a long note, with the chunk id the service gives it."""
    chunk_id = f"u-{LEARNER}_{document_id}_{index}"
    return unit(document_id, text, chunk_id=chunk_id, **extra)


def test_search_recalls_within_the_budget_and_keeps_the_limit() -> None:
    results = [
        {**unit(f"learn-ai.n{i}", f"Note {i}."), "scores": {"final": 1 - i / 10}}
        for i in range(4)
    ]
    service = (
        Service()
        .on(
            "POST",
            f"{BANK}/memories/recall",
            httpx.Response(200, json={"results": results}),
        )
        .on(
            "GET",
            f"{BANK}/memories/list",
            by_document(**{f"learn-ai.n{i}": [results[i]] for i in range(4)}),
        )
    )
    found = run(learner_notes(service).search("notes", limit=2))

    assert [(n.key, n.text, n.score) for n in found] == [
        ("n0", "Note 0.", 1.0),
        ("n1", "Note 1.", 0.9),
    ]
    assert json.loads(service.requests[0].content) == {
        "query": "notes",
        "max_tokens": 512,
        "trace": False,
        "tags_match": "any",
    }


def test_a_chunked_note_is_read_whole_in_chunk_order() -> None:
    # The service lists a note's chunks in no particular order; they share the
    # note's timestamps.
    chunks = [
        chunk("learn-ai.essay", "Part three.", 2),
        chunk("learn-ai.essay", "Part one.", 0, metadata={"source": "chat"}),
        chunk("learn-ai.essay", "Part two.", 1),
    ]
    service = Service().on(
        "GET", f"{BANK}/memories/list", by_document(**{"learn-ai.essay": chunks})
    )
    notes = learner_notes(service)
    whole = "Part one.\nPart two.\nPart three."

    assert run(notes.get("essay")).text == whole
    assert [n.text for n in run(notes.list())] == [whole]


def test_search_returns_each_note_once_whole_with_its_best_score() -> None:
    results = [
        {**chunk("learn-ai.essay", "Part two.", 1), "scores": {"final": 0.9}},
        {**unit("mit-learn.goal", "A certificate."), "scores": {"final": 0.8}},
        {**chunk("learn-ai.essay", "Part one.", 0), "scores": {"final": 0.7}},
        {**unit("learn-ai.pace", "Weekends."), "scores": {"final": 0.6}},
    ]
    service = (
        Service()
        .on(
            "POST",
            f"{BANK}/memories/recall",
            httpx.Response(200, json={"results": results}),
        )
        .on(
            "GET",
            f"{BANK}/memories/list",
            by_document(
                **{
                    "learn-ai.essay": [results[0], results[2]],
                    "mit-learn.goal": [results[1]],
                }
            ),
        )
    )
    found = run(learner_notes(service).search("essay", limit=2))

    assert [(n.app, n.key, n.text, n.score) for n in found] == [
        ("learn-ai", "essay", "Part one.\nPart two.", 0.9),
        ("mit-learn", "goal", "A certificate.", 0.8),
    ]


def test_delete_removes_the_apps_document_and_tolerates_a_missing_one() -> None:
    service = Service().on(
        "DELETE",
        f"{BANK}/documents/learn-ai.pace",
        httpx.Response(
            200,
            json={
                "success": True,
                "message": "deleted",
                "document_id": "learn-ai.pace",
                "memory_units_deleted": 1,
            },
        ),
        httpx.Response(404, json={"detail": "not found"}),
    )
    notes = learner_notes(service)
    run(notes.delete("pace"))
    run(notes.delete("pace"))
    assert len(service.requests) == 2


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, MemoryUnauthenticated),
        (403, MemoryForbidden),
        (404, MemoryNotFound),
        (409, MemoryFenced),
        (422, MemoryRejected),
        (429, MemoryUnavailable),
        (503, MemoryUnavailable),
    ],
)
def test_a_refused_write_raises_by_status(status: int, error: type[Exception]) -> None:
    service = Service().on(
        "POST", f"{BANK}/memories", httpx.Response(status, json={"detail": "no"})
    )
    with pytest.raises(error) as raised:
        run(learner_notes(service).put("pace", "Weekends."))
    assert type(raised.value) is error
    assert raised.value.status_code == status  # type: ignore[attr-defined]
    assert len(service.requests) == 1


def test_an_error_message_leaves_out_the_response_body() -> None:
    echoed = {"detail": [{"loc": ["body"], "input": {"content": "Weekends."}}]}
    service = Service().on("POST", f"{BANK}/memories", httpx.Response(422, json=echoed))
    with pytest.raises(MemoryRejected) as raised:
        run(learner_notes(service).put("pace", "Weekends."))
    assert "Weekends" not in str(raised.value)
    assert "422" in str(raised.value)


def test_a_timeout_is_retryable() -> None:
    def hang(request: httpx.Request) -> httpx.Response:
        msg = "slow"
        raise httpx.ReadTimeout(msg, request=request)

    service = Service().on("POST", f"{BANK}/memories/recall", hang)
    with pytest.raises(MemoryTimeout) as raised:
        run(learner_notes(service).search("anything"))
    assert raised.value.retryable


@pytest.mark.parametrize(
    ("raised", "error"),
    [
        (httpx.ConnectError("refused"), MemoryUnavailable),
        (httpx.RemoteProtocolError("dropped"), MemoryUnavailable),
        (httpx.UnsupportedProtocol("nope://"), MemoryRejected),
    ],
)
def test_a_transport_failure_is_classified_by_type(
    raised: Exception, error: type[Exception]
) -> None:
    def fail(_request: httpx.Request) -> httpx.Response:
        raise raised

    service = Service().on("POST", f"{BANK}/memories", fail)
    with pytest.raises(error) as caught:
        run(learner_notes(service).put("pace", "Weekends."))
    assert type(caught.value) is error
    assert caught.value.__cause__ is raised


@pytest.mark.parametrize(
    "token", ["not-a-jwt", jwt({"azp": "ol-learn-ai-client"}), jwt({"sub": ""})]
)
def test_a_learner_token_without_a_subject_is_refused(token: str) -> None:
    with pytest.raises(MemoryUnauthenticated):
        Service().backend().learner(LearnerCredential(token))


def test_a_service_token_is_refreshed_once_after_a_401() -> None:
    general = "/v1/default/banks/g-courses"
    service = Service().on(
        "POST",
        f"{general}/memories",
        httpx.Response(401, json={"detail": "expired"}),
        RETAINED,
    )
    tokens = Tokens("old", "new")
    notes = service.backend().general("courses", ServiceCredential(tokens))
    run(notes.put("schedule", "Every spring."))

    assert [r.headers["authorization"] for r in service.requests] == [
        "Bearer old",
        "Bearer new",
    ]
    assert tokens.refreshes == [False, True]


def test_a_learner_reads_general_memory_with_its_own_token() -> None:
    general = "/v1/default/banks/g-courses"
    service = Service().on(
        "GET", f"{general}/memories/list", page(unit("catalog.schedule", "Spring."))
    )
    notes = service.backend().general("courses", LearnerCredential(LEARNER_TOKEN))
    [note] = run(notes.list())

    assert note.app == "catalog"
    assert service.requests[0].headers["authorization"] == f"Bearer {LEARNER_TOKEN}"


def test_service_calls_need_tokens() -> None:
    backend = Service().backend()
    with pytest.raises(ValueError, match="Keycloak client"):
        backend.general("courses", ServiceCredential(None))
    with pytest.raises(ValueError, match="Keycloak client"):
        backend.admin(ServiceCredential(None))


@pytest.mark.parametrize(
    "stray",
    [
        unit("learn-ai.pace", "secret text", tags=[]),
        unit("learn-ai.pace", "secret text", tags=["app:learn-ai", "app:mit-learn"]),
        unit("other.pace", "secret text", tags=["app:learn-ai"]),
        {"id": "observation-1", "text": "secret text", "tags": None},
    ],
)
def test_a_unit_no_app_wrote_is_skipped_and_logged_without_its_text(
    stray: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    kept = unit("mit-learn.goal", "A certificate.")
    service = (
        Service()
        .on("GET", f"{BANK}/memories/list", page(stray, kept))
        .on(
            "POST",
            f"{BANK}/memories/recall",
            httpx.Response(
                200,
                json={
                    "results": [
                        {**stray, "scores": {"final": 1.0}},
                        {**kept, "scores": {"final": 0.5}},
                    ]
                },
            ),
        )
    )
    notes = learner_notes(service)

    assert [n.key for n in run(notes.list())] == ["goal"]
    assert [n.key for n in run(notes.search("anything"))] == ["goal"]
    assert str(stray["id"]) in caplog.text
    assert "secret text" not in caplog.text


# -- forget -----------------------------------------------------------------------


def operations(*ids: str) -> httpx.Response:
    listed = [
        {
            "id": i,
            "task_type": "retain",
            "items_count": 1,
            "created_at": "2026-11-10T12:00:00Z",
            "status": "pending",
            "error_message": None,
        }
        for i in ids
    ]
    return httpx.Response(
        200,
        json={
            "bank_id": f"u-{LEARNER}",
            "total": len(ids),
            "limit": hindsight_module.OPERATIONS_PAGE_SIZE,
            "offset": 0,
            "operations": listed,
        },
    )


def cancelled(operation_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"success": True, "message": "cancelled", "operation_id": operation_id},
    )


def forget_service() -> Service:
    def by_status(*replies: tuple[str, httpx.Response]) -> Reply:
        queued = list(replies)

        def answer(request: httpx.Request) -> httpx.Response:
            status = request.url.params["status"]
            index = next(i for i, (s, _) in enumerate(queued) if s == status)
            return queued.pop(index)[1]

        return answer

    return (
        Service()
        .on(
            "DELETE",
            f"{BANK}/memories",
            httpx.Response(200, json={"success": True, "deleted_count": 3}),
        )
        .on(
            "GET",
            f"{BANK}/operations",
            by_status(
                ("pending", operations("op-1", "op-2")),
                ("pending", operations()),
                ("processing", operations("op-2")),
                ("pending", operations()),
                ("processing", operations()),
            ),
        )
        .on("DELETE", f"{BANK}/operations/op-1", cancelled("op-1"))
        .on(
            "DELETE",
            f"{BANK}/operations/op-2",
            httpx.Response(409, json={"detail": "already processing"}),
        )
        .on(
            "DELETE",
            BANK,
            httpx.Response(200, json={"success": True, "deleted_count": 1}),
        )
    )


def test_forget_clears_drains_then_deletes_the_bank(no_waits: list[float]) -> None:
    service = forget_service()
    admin = service.backend().admin(ServiceCredential(Tokens("admin-token")))

    assert run(admin.forget(LEARNER)) == 4
    assert service.seen() == [
        ("DELETE", f"{BANK}/memories"),
        ("GET", f"{BANK}/operations"),
        ("DELETE", f"{BANK}/operations/op-1"),
        ("DELETE", f"{BANK}/operations/op-2"),
        ("GET", f"{BANK}/operations"),
        ("GET", f"{BANK}/operations"),
        ("GET", f"{BANK}/operations"),
        ("GET", f"{BANK}/operations"),
        ("DELETE", BANK),
    ]
    assert {r.headers["x-ol-learner"] for r in service.requests} == {LEARNER}
    assert {r.headers["authorization"] for r in service.requests} == {
        "Bearer admin-token"
    }
    assert no_waits == [hindsight_module.DRAIN_POLL_SECONDS] * 2


def test_forget_of_an_absent_bank_skips_the_drain() -> None:
    missing = httpx.Response(404, json={"detail": "bank not found"})
    service = (
        Service().on("DELETE", f"{BANK}/memories", missing).on("DELETE", BANK, missing)
    )
    admin = service.backend().admin(ServiceCredential(Tokens("admin-token")))
    assert run(admin.forget(LEARNER)) == 0
    assert service.seen() == [("DELETE", f"{BANK}/memories"), ("DELETE", BANK)]


def test_forget_starts_over_after_a_failure_that_can_clear(
    no_waits: list[float],
) -> None:
    service = (
        Service()
        .on(
            "DELETE",
            f"{BANK}/memories",
            httpx.Response(503, json={"detail": "deadlock"}),
            httpx.Response(200, json={"success": True, "deleted_count": 2}),
        )
        .on("GET", f"{BANK}/operations", operations())
        .on("DELETE", BANK, httpx.Response(200, json={"success": True}))
    )
    admin = service.backend().admin(ServiceCredential(Tokens("admin-token")))
    assert run(admin.forget(LEARNER)) == 2
    assert no_waits == [hindsight_module.FORGET_BACKOFF_SECONDS]


def test_forget_does_not_start_over_after_a_drain_times_out(
    no_waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hindsight_module, "DRAIN_TIMEOUT_SECONDS", -1.0)
    service = (
        Service()
        .on(
            "DELETE",
            f"{BANK}/memories",
            httpx.Response(200, json={"success": True, "deleted_count": 1}),
        )
        .on("GET", f"{BANK}/operations", operations("op-1"))
        .on(
            "DELETE",
            f"{BANK}/operations/op-1",
            httpx.Response(409, json={"detail": "already processing"}),
        )
    )
    admin = service.backend().admin(ServiceCredential(Tokens("admin-token")))
    with pytest.raises(MemoryTimeout) as raised:
        run(admin.forget(LEARNER))

    assert raised.value.retryable
    assert service.seen().count(("DELETE", f"{BANK}/memories")) == 1
    assert no_waits == []


def test_forget_gives_up_after_its_attempts(no_waits: list[float]) -> None:
    service = Service().on(
        "DELETE", f"{BANK}/memories", httpx.Response(503, json={"detail": "down"})
    )
    admin = service.backend().admin(ServiceCredential(Tokens("admin-token")))
    with pytest.raises(MemoryUnavailable):
        run(admin.forget(LEARNER))
    assert len(service.requests) == hindsight_module.FORGET_ATTEMPTS
    assert len(no_waits) == hindsight_module.FORGET_ATTEMPTS - 1


def test_forget_does_not_retry_a_refusal() -> None:
    service = Service().on(
        "DELETE", f"{BANK}/memories", httpx.Response(403, json={"detail": "no"})
    )
    admin = service.backend().admin(ServiceCredential(Tokens("admin-token")))
    with pytest.raises(MemoryForbidden):
        run(admin.forget(LEARNER))
    assert len(service.requests) == 1


def test_forget_takes_only_keycloak_user_ids() -> None:
    admin = Service().backend().admin(ServiceCredential(Tokens("admin-token")))
    with pytest.raises(ValueError, match="UUID"):
        run(admin.forget("learner-1"))


# -- the live service (opt-in) ------------------------------------------------------

_LIVE_CONFIG = os.environ.get("OL_LLM_MEMORY_IT_CONFIG")


@dataclass(frozen=True, kw_only=True)
class LiveClient:
    client_id: str
    client_secret: str


@dataclass(frozen=True, kw_only=True)
class LiveConfig:
    """What the live run needs, read from ``OL_LLM_MEMORY_IT_CONFIG``.

    For the compose harness: ``url`` ``http://127.0.0.1:8888``, ``keycloak_url``
    ``http://127.0.0.1:8098``, ``realm`` ``ol-memory-harness``, the realm's
    clients and secrets from its ``keycloak.py``, and its master admin login.
    Each app's client must allow the password grant, which stands in for the
    learner's browser login.

    :param url: the Hindsight API.
    :param keycloak_url: Keycloak's base URL.
    :param realm: the realm the service trusts.
    :param keycloak_admin: master-realm admin ``username`` and ``password``, to
        create learners.
    :param learner_password: the password given to each created learner.
    :param apps: two apps by name, each with the client it logs learners in
        through and exchanges as.
    :param general_writer: the app whose client holds the general grant.
    :param admin: the platform admin client.
    """

    url: str
    keycloak_url: str
    realm: str
    keycloak_admin: dict[str, str]
    learner_password: str
    apps: dict[str, LiveClient]
    general_writer: str
    admin: LiveClient

    @classmethod
    def load(cls, path: str) -> "LiveConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            **{
                **raw,
                "apps": {app: LiveClient(**c) for app, c in raw["apps"].items()},
                "admin": LiveClient(**raw["admin"]),
            }
        )

    @property
    def token_url(self) -> str:
        return f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token"


class LiveHarness:
    search_attempts = 10
    search_wait_seconds = 3.0

    def __init__(self, config: LiveConfig) -> None:
        self.config = config
        first, second = config.apps
        self.apps = (first, second)
        self.general_writer = config.general_writer
        self._memories: dict[str, Memory] = {}

    def memory(self, app: str) -> Memory:
        if (memory := self._memories.get(app)) is None:
            config = self.config
            client = config.apps[app]

            def keycloak(c: LiveClient, scope: str | None = None) -> KeycloakClient:
                return KeycloakClient(
                    token_url=config.token_url,
                    client_id=c.client_id,
                    client_secret=c.client_secret,
                    scope=scope,
                )

            memory = self._memories[app] = Memory(
                HindsightBackend(config.url, app=app, timeout_seconds=60),
                exchange=KeycloakTokenExchange(
                    KeycloakExchangeClient(
                        token_url=config.token_url,
                        client_id=client.client_id,
                        client_secret=client.client_secret,
                    )
                ),
                general_tokens=ClientCredentialsTokens(
                    keycloak(client, "ol-memory-audience ol-memory-general")
                )
                if app == config.general_writer
                else None,
                admin_tokens=ClientCredentialsTokens(keycloak(config.admin)),
            )
        return memory

    async def new_learner(self) -> ConformanceLearner:
        config = self.config
        username = f"conformance-{uuid.uuid4().hex[:12]}"
        async with httpx.AsyncClient(timeout=60) as http:
            admin = await http.post(
                f"{config.keycloak_url}/realms/master/protocol/openid-connect/token",
                data={
                    "grant_type": "password",
                    "client_id": "admin-cli",
                    **config.keycloak_admin,
                },
            )
            admin_token = admin.raise_for_status().json()["access_token"]
            headers = {"authorization": f"Bearer {admin_token}"}
            users = f"{config.keycloak_url}/admin/realms/{config.realm}/users"
            (
                await http.post(
                    users,
                    headers=headers,
                    json={
                        "username": username,
                        "enabled": True,
                        "email": f"{username}@example.org",
                        "emailVerified": True,
                        "firstName": username,
                        "lastName": "Conformance",
                        "credentials": [
                            {
                                "type": "password",
                                "value": config.learner_password,
                                "temporary": False,
                            }
                        ],
                    },
                )
            ).raise_for_status()
            [user] = (
                await http.get(
                    users,
                    headers=headers,
                    params={"username": username, "exact": "true"},
                )
            ).json()
            tokens = {}
            for app, client in config.apps.items():
                login = await http.post(
                    config.token_url,
                    data={
                        "grant_type": "password",
                        "client_id": client.client_id,
                        "client_secret": client.client_secret,
                        "username": username,
                        "password": config.learner_password,
                        "scope": "openid",
                    },
                )
                tokens[app] = login.raise_for_status().json()["access_token"]
        return ConformanceLearner(id=user["id"], session_tokens=tokens)

    def general_name(self) -> str:
        return f"conformance-{uuid.uuid4().hex[:8]}"


@pytest.mark.hindsight
@pytest.mark.skipif(
    _LIVE_CONFIG is None, reason="set OL_LLM_MEMORY_IT_CONFIG for a live Hindsight"
)
@pytest.mark.parametrize("check", CHECKS, ids=lambda check: check.__name__)
def test_live_hindsight_conforms(check: Check) -> None:
    assert _LIVE_CONFIG is not None
    harness = LiveHarness(LiveConfig.load(_LIVE_CONFIG))

    run_coroutine(check(harness))
