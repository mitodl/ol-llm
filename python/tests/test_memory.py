import asyncio
import base64
import json
import pickle
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, EntryPoints
from typing import Any, TypeVar

import ol_llm.memory
import opik
import pytest
from conftest import runner_loop
from ol_llm import Attribution, close_thread_loop, plugins, run_coroutine, tracing
from ol_llm.attribution import NO_ATTRIBUTION
from ol_llm.memory import (
    ALL_CAPABILITIES,
    Capability,
    HindsightBackend,
    InMemoryBackend,
    InMemoryStore,
    LearnerCredential,
    LearnerMemory,
    Memory,
    MemoryConfig,
    MemoryFenced,
    MemoryForbidden,
    MemoryNotFound,
    MemoryUnavailable,
    MemoryUnsupported,
    NoteStore,
    SyncMemory,
    backend_factory,
    build_memory,
    register_backend,
)
from ol_llm.memory import config as memory_config
from ol_llm.memory.conformance import (
    CHECKS,
    Check,
    ConformanceLearner,
)
from ol_llm.memory.in_memory import default_store
from ol_llm.memory.tokens import ClientCredentialsTokens, KeycloakTokenExchange
from ol_llm.tracing import TracingConfig, configure_tracing
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

_T = TypeVar("_T")

APPS = ("learn-ai", "mit-learn")
OPIK_METADATA = "opik.metadata."
USE_CASE = "tutor-recall"


def run(coroutine: Awaitable[_T]) -> _T:
    async def wait() -> _T:
        return await coroutine

    return asyncio.run(wait())


@dataclass
class InMemoryHarness:
    store: InMemoryStore = field(default_factory=InMemoryStore)
    apps: tuple[str, str] = APPS
    general_writer: str = APPS[0]
    search_attempts: int = 1
    search_wait_seconds: float = 0.0

    def memory(self, app: str) -> Memory:
        return Memory(InMemoryBackend(app, store=self.store))

    async def new_learner(self) -> ConformanceLearner:
        learner_id = str(uuid.uuid4())
        return ConformanceLearner(
            id=learner_id, session_tokens=dict.fromkeys(self.apps, learner_id)
        )

    def general_name(self) -> str:
        return f"conformance-{uuid.uuid4().hex[:8]}"


@pytest.mark.parametrize("check", CHECKS, ids=lambda check: check.__name__)
def test_the_in_memory_backend_conforms(check: Check) -> None:
    run(check(InMemoryHarness()))


def learner(
    memory: Memory,
    session: str = "learner-1",
    attribution: Attribution = NO_ATTRIBUTION,
) -> LearnerMemory:
    return run(memory.learner(session, use_case=USE_CASE, attribution=attribution))


def in_memory(
    app: str = "learn-ai", capabilities: frozenset[Capability] = ALL_CAPABILITIES
) -> Memory:
    return Memory(
        InMemoryBackend(app, store=InMemoryStore(), capabilities=capabilities)
    )


def spans(exported: InMemorySpanExporter) -> list[dict[str, Any]]:
    tracing.flush()
    return [
        {"name": span.name, **(span.attributes or {})}
        for span in exported.get_finished_spans()
    ]


def test_each_operation_has_a_span_with_attribution(
    exported: InMemorySpanExporter,
) -> None:
    attribution = Attribution(user_id="user-1", thread_id="thread-1", run_id="run-1")
    notes = learner(in_memory(), attribution=attribution)
    run(notes.put("pace", "Studies on weekends.", metadata={"source": "chat"}))
    run(notes.get("pace"))
    run(notes.search("weekends"))
    run(notes.list())
    run(notes.delete("pace"))

    recorded = spans(exported)
    assert [span["name"] for span in recorded] == [
        "memory put",
        "memory get",
        "memory search",
        "memory list",
        "memory delete",
    ]
    # As Opik gets them: the attribution and keys as metadata, the content
    # outside it, where Opik shows it as the span's input.
    for span in recorded:
        assert {k: span[OPIK_METADATA + k] for k in ("app", "use_case")} == {
            "app": "learn-ai",
            "use_case": USE_CASE,
        }
        assert span[OPIK_METADATA + "environment"] == "ci"
        assert {k: span[OPIK_METADATA + k] for k in ("user_id", "run_id")} == {
            "user_id": "user-1",
            "run_id": "run-1",
        }
        assert span["gen_ai.conversation.id"] == "thread-1"
        assert span[OPIK_METADATA + "memory.scope"] == "learner"
        assert span[OPIK_METADATA + "memory.backend"] == "in-memory"
        assert not {"app", "user_id", "memory.scope"} & span.keys()
    put, get, search, *_ = recorded
    assert put[OPIK_METADATA + "memory.key"] == "pace"
    assert put["memory.text"] == "Studies on weekends."
    assert search["memory.query"] == "weekends"
    assert search[OPIK_METADATA + "memory.result_keys"] == ("learn-ai.pace",)
    assert "Studies on weekends." in search["memory.results"]
    assert get[OPIK_METADATA + "memory.results_count"] == 1


def test_spans_carry_no_note_content_without_include_content(
    exported: InMemorySpanExporter,
) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(include_content=False))
    notes = learner(in_memory())
    run(notes.put("pace", "secret text", metadata={"source": "secret meta"}))
    run(notes.search("secret text"))
    run(notes.get("pace"))
    with pytest.raises(MemoryNotFound):
        run(notes.get("missing"))

    recorded = spans(exported)
    assert len(recorded) == 4
    assert all("secret" not in repr(span) for span in recorded), recorded
    assert recorded[1][OPIK_METADATA + "memory.result_keys"] == ("learn-ai.pace",)
    [failed] = [
        span
        for span in exported.get_finished_spans()
        if span.status.status_code == StatusCode.ERROR
    ]
    assert failed.status.description is None


def test_a_failed_operation_marks_its_span(exported: InMemorySpanExporter) -> None:
    store = InMemoryStore()
    notes = run(
        Memory(InMemoryBackend("learn-ai", store=store)).learner(
            "learner-1", use_case=USE_CASE
        )
    )
    run(
        Memory(InMemoryBackend("learn-ai", store=store))
        .admin(use_case="account-deletion")
        .forget("learner-1")
    )
    with pytest.raises(MemoryFenced):
        run(notes.put("pace", "late"))
    tracing.flush()

    forget, put = exported.get_finished_spans()
    assert (forget.attributes or {})[OPIK_METADATA + "user_id"] == "learner-1"
    assert (forget.attributes or {})[OPIK_METADATA + "memory.removed_count"] == 0
    assert put.status.status_code == StatusCode.ERROR
    assert [event.name for event in put.events] == ["exception"]


def test_a_span_lands_in_the_calling_trace(exported: InMemorySpanExporter) -> None:
    notes = learner(in_memory())

    @opik.track(project_name="learn-ai")
    def answer() -> None:
        run(notes.search("anything"))

    with opik.record_traces_locally() as storage:
        answer()
        [trace] = storage.trace_trees
    tracing.flush()
    [span] = exported.get_finished_spans()
    assert (span.attributes or {})["opik.trace_id"] == trace.id


def test_no_span_while_tracing_is_off(exported: InMemorySpanExporter) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False))
    run(learner(in_memory()).list())
    assert spans(exported) == []


def test_a_missing_capability_raises_unsupported() -> None:
    memory = in_memory(capabilities=frozenset())
    notes = learner(memory)
    with pytest.raises(MemoryUnsupported) as search:
        run(notes.search("anything"))
    with pytest.raises(MemoryUnsupported):
        notes.general("courses")
    with pytest.raises(MemoryUnsupported):
        memory.general("courses", use_case=USE_CASE)
    with pytest.raises(MemoryUnsupported):
        memory.admin(use_case=USE_CASE)
    assert search.value.operation == "search"
    run(notes.put("pace", "Puts and gets need no capability."))
    assert run(notes.get("pace")).text == "Puts and gets need no capability."


@pytest.mark.parametrize(
    "call",
    [
        lambda notes: notes.put("Has Spaces", "x"),
        lambda notes: notes.put("learn-ai.pace", "x"),
        lambda notes: notes.get(""),
        lambda notes: notes.delete("a" * 64),
        lambda notes: notes.search("x", limit=0),
    ],
)
def test_bad_keys_and_limits_fail_before_the_backend(
    call: Callable[[LearnerMemory], Awaitable[object]],
) -> None:
    with pytest.raises(ValueError, match=r"keys|limit"):
        run(call(learner(in_memory())))


def test_names_are_checked() -> None:
    memory = in_memory()
    with pytest.raises(ValueError, match="use_case"):
        run(memory.learner("learner-1", use_case="Tutor Recall"))
    with pytest.raises(ValueError, match="general memory names"):
        memory.general("Courses", use_case=USE_CASE)


def test_the_learner_comes_from_the_credential() -> None:
    store = InMemoryStore()
    backend = InMemoryBackend(
        "learn-ai", store=store, identify=lambda token: token.split(":")[1]
    )
    memory = Memory(backend)
    run(learner(memory, "session:learner-1").put("pace", "Weekends."))
    assert [n.key for n in run(learner(memory, "other-session:learner-1").list())] == [
        "pace"
    ]
    assert run(learner(memory, "session:learner-2").list()) == []


def test_the_session_token_is_exchanged_before_the_backend_sees_it() -> None:
    exchanged: list[str] = []

    class Exchange:
        async def exchange(self, subject_token: str) -> str:
            exchanged.append(subject_token)
            return f"learner-of-{subject_token}"

    store = InMemoryStore()
    memory = Memory(InMemoryBackend("learn-ai", store=store), exchange=Exchange())
    run(learner(memory, "session-1").put("pace", "Weekends."))
    direct = Memory(InMemoryBackend("learn-ai", store=store))
    assert [n.key for n in run(learner(direct, "learner-of-session-1").list())] == [
        "pace"
    ]
    assert exchanged == ["session-1"]


def test_the_admin_handle_is_its_own_type() -> None:
    memory = in_memory()
    notes = learner(memory)
    assert not hasattr(notes, "forget")
    assert not hasattr(memory.general("courses", use_case=USE_CASE), "forget")


def test_a_forget_fence_expires() -> None:
    store = InMemoryStore(fence_seconds=0)
    memory = Memory(InMemoryBackend("learn-ai", store=store))
    notes = learner(memory)
    run(notes.put("pace", "Weekends."))
    assert run(memory.admin(use_case=USE_CASE).forget("learner-1")) == 1
    run(notes.put("pace", "After the fence."))
    assert [n.text for n in run(notes.list())] == ["After the fence."]


def test_delete_is_allowed_while_fenced() -> None:
    store = InMemoryStore()
    memory = Memory(InMemoryBackend("learn-ai", store=store))
    notes = learner(memory)
    run(memory.admin(use_case=USE_CASE).forget("learner-1"))
    run(notes.delete("pace"))


def test_general_writes_need_a_service_credential() -> None:
    memory = in_memory()
    general = learner(memory).general("courses")
    with pytest.raises(MemoryForbidden) as refused:
        run(general.put("x", "y"))
    with pytest.raises(MemoryForbidden):
        run(general.delete("x"))
    assert refused.value.status_code == 403


@pytest.mark.parametrize(
    "error",
    [
        MemoryNotFound("gone", operation="get", status_code=404),
        MemoryUnavailable("down", operation="search", status_code=503),
    ],
)
def test_errors_survive_pickling(error: Exception) -> None:
    copy = pickle.loads(pickle.dumps(error))  # noqa: S301
    assert type(copy) is type(error)
    assert copy.__dict__ == error.__dict__
    assert copy.args == error.args


def test_retryable_classes() -> None:
    assert MemoryUnavailable.retryable
    assert not MemoryFenced.retryable
    assert not MemoryNotFound.retryable


# -- configuration -----------------------------------------------------------------

ENV = {
    "OL_LLM_MEMORY_BACKEND": "hindsight",
    "OL_LLM_MEMORY_APP": "learn-ai",
    "OL_LLM_MEMORY_URL": "http://hindsight.example:8888",
    "OL_LLM_MEMORY_TOKEN_URL": "https://sso.example/realms/olapps/protocol/openid-connect/token",
    "OL_LLM_MEMORY_EXCHANGE_CLIENT_ID": "ol-learn-ai-client",
    "OL_LLM_MEMORY_EXCHANGE_CLIENT_SECRET": "exchange-secret",
    "OL_LLM_MEMORY_ADMIN_CLIENT_ID": "ol-memory-admin-client",
    "OL_LLM_MEMORY_ADMIN_CLIENT_SECRET": "admin-secret",
}


def test_config_from_env() -> None:
    config = MemoryConfig.from_env(ENV)

    assert (config.backend, config.app, config.url) == (
        "hindsight",
        "learn-ai",
        "http://hindsight.example:8888",
    )
    assert config.timeout_seconds == 10
    assert config.exchange is not None
    assert config.exchange.client_id == "ol-learn-ai-client"
    assert config.exchange.audience == "ol-memory-client"
    assert config.exchange.scope == "ol-memory-audience"
    assert config.general_client is None
    assert config.admin_client is not None
    assert config.admin_client.scope is None
    assert "secret" not in repr(config)


def test_the_general_client_requests_the_grant_by_default() -> None:
    config = MemoryConfig.from_env(
        {
            **ENV,
            "OL_LLM_MEMORY_GENERAL_CLIENT_ID": "catalog-indexer",
            "OL_LLM_MEMORY_GENERAL_CLIENT_SECRET": "general-secret",
        }
    )
    assert config.general_client is not None
    assert config.general_client.scope == "ol-memory-audience ol-memory-general"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"OL_LLM_MEMORY_BACKEND": ""}, "OL_LLM_MEMORY_BACKEND"),
        ({"OL_LLM_MEMORY_APP": ""}, "OL_LLM_MEMORY_APP"),
        ({"OL_LLM_MEMORY_EXCHANGE_CLIENT_SECRET": ""}, "set together"),
        ({"OL_LLM_MEMORY_TOKEN_URL": ""}, "TOKEN_URL"),
        ({"OL_LLM_MEMORY_TIMEOUT_SECONDS": "soon"}, "not a number"),
        ({"OL_LLM_MEMORY_TIMEOUT_SECONDS": "0"}, "positive"),
    ],
)
def test_incomplete_config_fails(change: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MemoryConfig.from_env({**ENV, **change})


def test_build_memory_wires_the_backend_and_tokens() -> None:
    memory = build_memory(MemoryConfig.from_env(ENV))

    assert isinstance(memory.backend, HindsightBackend)
    assert memory.app == "learn-ai"
    assert isinstance(memory._exchange, KeycloakTokenExchange)  # noqa: SLF001
    assert isinstance(memory._admin_tokens, ClientCredentialsTokens)  # noqa: SLF001
    assert memory._general_tokens is None  # noqa: SLF001


def test_build_memory_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OL_LLM_MEMORY_BACKEND", "in-memory")
    monkeypatch.setenv("OL_LLM_MEMORY_APP", "mit-learn")
    memory = build_memory()
    assert isinstance(memory.backend, InMemoryBackend)
    assert memory.backend.capabilities == ALL_CAPABILITIES
    default_store().clear()


def test_hindsight_needs_a_url() -> None:
    with pytest.raises(ValueError, match="url"):
        build_memory(MemoryConfig(backend="hindsight", app="learn-ai"))


def test_an_unknown_backend_names_the_known_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugins, "entry_points", lambda **_: EntryPoints(()))
    with pytest.raises(ValueError, match=r"'hindsight', 'in-memory'"):
        build_memory(MemoryConfig(backend="cognee", app="learn-ai"))


def test_backends_register_by_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    def entry_points(**kwargs: str) -> EntryPoints:
        calls.append(kwargs)
        return EntryPoints(
            (
                EntryPoint(
                    name="other",
                    value="ol_llm.memory.config:_in_memory",
                    group=memory_config.ENTRY_POINT_GROUP,
                ),
            )
        )

    monkeypatch.setattr(plugins, "entry_points", entry_points)
    monkeypatch.setattr(
        memory_config._backends,  # noqa: SLF001
        "_factories",
        dict(memory_config._backends._factories),  # noqa: SLF001
    )
    memory = build_memory(MemoryConfig(backend="other", app="learn-ai"))
    assert isinstance(memory.backend, InMemoryBackend)
    assert backend_factory("other") is memory_config._in_memory  # noqa: SLF001
    assert calls == [{"group": "ol_llm.memory_backends", "name": "other"}]


def test_backends_register_in_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_config._backends,  # noqa: SLF001
        "_factories",
        dict(memory_config._backends._factories),  # noqa: SLF001
    )
    store = InMemoryStore()
    register_backend("shared", lambda config: InMemoryBackend(config.app, store=store))
    memory = build_memory(MemoryConfig(backend="shared", app="learn-ai"))
    run(learner(memory).put("pace", "Weekends."))
    assert store.notes("learner:learner-1")


def test_the_hindsight_backend_without_its_extra_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "ol_llm.memory.hindsight", raising=False)
    for name in [m for m in sys.modules if m.startswith("hindsight_client_api")]:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"ol-llm\[memory\]"):
        build_memory(
            MemoryConfig(backend="hindsight", app="learn-ai", url="http://hindsight")
        )
    with pytest.raises(ImportError, match=r"ol-llm\[memory\]"):
        _ = ol_llm.memory.HindsightBackend


# -- the sync facade ------------------------------------------------------------


def test_sync_memory_runs_every_operation_on_the_thread_loop() -> None:
    store = InMemoryStore()
    memory = SyncMemory(Memory(InMemoryBackend("learn-ai", store=store)))
    loops_seen: list[asyncio.AbstractEventLoop] = []

    class Spy(InMemoryBackend):
        def learner(self, credential: LearnerCredential) -> NoteStore:
            loops_seen.append(asyncio.get_running_loop())
            return super().learner(credential)

    spied = SyncMemory(Memory(Spy("learn-ai", store=store)))
    notes = spied.learner("learner-1", use_case=USE_CASE)
    notes.put("pace", "Weekends.", metadata={"source": "chat"})
    assert notes.get("pace").text == "Weekends."
    assert [n.key for n in notes.search("weekends")] == ["pace"]
    assert [n.key for n in notes.list()] == ["pace"]
    notes.delete("pace")
    assert notes.list() == []
    assert loops_seen == [runner_loop()]

    memory.general("courses", use_case=USE_CASE).put("schedule", "Spring.")
    general = memory.learner("learner-1", use_case=USE_CASE).general("courses")
    assert [n.text for n in general.list()] == ["Spring."]

    memory.learner("learner-2", use_case=USE_CASE).put("pace", "Evenings.")
    assert memory.admin(use_case="account-deletion").forget("learner-2") == 1
    with pytest.raises(MemoryFenced):
        memory.learner("learner-2", use_case=USE_CASE).put("pace", "Again.")


def test_sync_memory_refuses_a_running_loop() -> None:
    memory = SyncMemory(Memory(InMemoryBackend("learn-ai")))

    async def inside() -> None:
        with pytest.raises(RuntimeError, match="await the coroutine instead"):
            memory.learner("learner-1", use_case=USE_CASE)

    asyncio.run(inside())


def test_the_runner_closes_the_hindsight_client_with_its_loop(
    keepalive_url: str,
) -> None:
    claims = base64.urlsafe_b64encode(json.dumps({"sub": "learner"}).encode())
    token = f"e30.{claims.decode().rstrip('=')}.signature"
    backend = HindsightBackend(keepalive_url, app="learn-ai")
    notes = backend.learner(LearnerCredential(token))

    assert run_coroutine(notes.search("anything")) == []
    clients = backend._api._clients._values  # noqa: SLF001
    [client] = clients.values()
    close_thread_loop()

    assert client.rest_client.http.is_closed
    assert not clients
