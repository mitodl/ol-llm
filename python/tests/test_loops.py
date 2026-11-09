"""The runner, loop-end closing, fork safety, and the lifecycle matrix.

Every resource ol-llm keeps per event loop goes through ``SCENARIOS``: two
sequential ``asyncio.run`` calls, ``run_coroutine`` from a thread pool,
``async_to_sync`` from a sync caller, and a forked child of a parent that already
used it. A new loop-local client (a memory backend's, say) adds itself to
``RESOURCES``.
"""

import asyncio
import gc
import gzip
import inspect
import json
import os
import pickle
import select
import signal
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Coroutine, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from typing import Any

import pytest
from asgiref.sync import async_to_sync
from conftest import PRIMARY, runner_loop
from ol_llm import (
    KeycloakClient,
    TracingConfig,
    UseCase,
    close_thread_loop,
    complete,
    complete_sync,
    configure_tracing,
    embed,
    embeddings,
    loops,
    models,
    prompts,
    run_coroutine,
    tracing,
)
from ol_llm import settings as settings_module
from ol_llm.models import shared_model
from ol_llm.policy import resolve_policy
from ol_llm.settings import GatewayRoute, Settings, configure
from ol_llm.testing import override_models
from ol_llm.tracing import OTLP_TRACES_PATH, KeycloakTokens
from opik.api_objects.opik_client import reset_global_client
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

FORK_TIMEOUT_SECONDS = 30
# The autouse ``exported`` fixture swaps this for an in-memory exporter.
REAL_EXPORTER = tracing._exporter  # noqa: SLF001

CHAT = UseCase(app="mit-learn", name="summary", profile="request", model=PRIMARY)
SEARCH = UseCase(
    app="mit-learn",
    name="search-query",
    profile="request",
    model="openai:text-embedding-3-small",
)


def run_in_fork(target: Callable[[], object]) -> object:
    """Run ``target`` in a forked child and return what it returned.

    The child reports through a pipe and leaves with ``os._exit``, so none of the
    parent's test machinery runs in it.

    :raises pytest.fail.Exception: if the child raised or hung.
    """
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        # Python warns that forking a threaded process may deadlock; a prefork
        # worker pool does exactly that, which is the case under test.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        os.close(read_fd)
        try:
            report = ("ok", target())
        except BaseException:  # noqa: BLE001
            report = ("error", traceback.format_exc())
        with os.fdopen(write_fd, "wb") as out:
            pickle.dump(report, out)
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as reports:
        ready, _, _ = select.select([reports], [], [], FORK_TIMEOUT_SECONDS)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("the forked child hung")
        status, value = pickle.load(reports)  # noqa: S301 - our own child
    os.waitpid(pid, 0)
    if status == "error":
        pytest.fail(f"the forked child raised:\n{value}")
    return value


# The lifecycle matrix.

Call = Callable[[], Coroutine[Any, Any, object]]


def two_asyncio_runs(call: Call) -> list[object]:
    return [asyncio.run(call()) for _ in range(2)]


def run_coroutine_on_a_thread_pool(call: Call) -> list[object]:
    with ThreadPoolExecutor(2) as pool:
        return list(pool.map(lambda _: run_coroutine(call()), range(4)))


def async_to_sync_from_a_sync_caller(call: Call) -> list[object]:
    return [async_to_sync(call)() for _ in range(2)]


def a_forked_child(call: Call) -> list[object]:
    first = run_coroutine(call())
    return [first, run_in_fork(lambda: run_coroutine(call()))]


SCENARIOS = [
    two_asyncio_runs,
    run_coroutine_on_a_thread_pool,
    async_to_sync_from_a_sync_caller,
    a_forked_child,
]


@dataclass(frozen=True)
class Resource:
    call: Call
    expected: object


async def _chat() -> object:
    return (await complete(CHAT, "hi")).output


async def _embedding() -> object:
    return (await embed(SEARCH, ["hi"], input_type="query")).dimensions


def _token_resource(url: str) -> Resource:
    tokens = KeycloakTokens(
        KeycloakClient(token_url=f"{url}/token", client_id="app", client_secret="s")
    )

    async def call() -> object:
        return tokens.token(refresh=True)

    return Resource(call, "token")


RESOURCES: dict[str, Callable[[str], Resource]] = {
    "model": lambda _url: Resource(_chat, "ok"),
    "embedding": lambda _url: Resource(_embedding, 3),
    "token": _token_resource,
}


@pytest.fixture(params=sorted(RESOURCES))
def resource(keepalive_url: str, request: pytest.FixtureRequest) -> Resource:
    route = GatewayRoute(base_url=f"{keepalive_url}/v1", api_key="app-key")
    configure(Settings(environment="ci", gateway_routes={"openai": route}))
    return RESOURCES[request.param](keepalive_url)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_loop_local_resource_survives_each_lifecycle(
    resource: Resource,
    scenario: Callable[[Call], list[object]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    results = scenario(resource.call)

    assert results
    assert all(result == resource.expected for result in results)
    assert not [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]


# The runner.


def test_run_coroutine_keeps_one_loop_per_thread() -> None:
    async def running() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    first = run_coroutine(running())
    second = run_coroutine(running())
    on_another = ThreadPoolExecutor(1).submit(run_coroutine, running()).result()

    assert first is second
    assert first is runner_loop()
    assert on_another is not first


def test_run_coroutine_refuses_a_running_loop_and_closes_the_coroutine() -> None:
    async def inner() -> None:
        pass

    async def outer() -> None:
        coro = inner()
        with pytest.raises(RuntimeError, match="await the coroutine instead"):
            run_coroutine(coro)
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED

    asyncio.run(outer())


def test_an_interrupted_coroutine_is_cancelled_and_cleaned_up() -> None:
    cleaned: list[bool] = []

    def interrupt() -> None:
        raise KeyboardInterrupt

    async def waiting() -> None:
        asyncio.get_running_loop().call_soon(interrupt)
        try:
            await asyncio.sleep(60)
        finally:
            cleaned.append(True)

    with pytest.raises(KeyboardInterrupt):
        run_coroutine(waiting())

    assert cleaned == [True]


def test_a_coroutine_sees_the_callers_context() -> None:
    var: ContextVar[str] = ContextVar("var", default="unset")
    var.set("caller")

    async def read() -> str:
        return var.get()

    assert run_coroutine(read()) == "caller"


# LoopLocal and closing.


class Client:
    def __init__(self) -> None:
        self.closed = False


async def close_client(client: Client) -> None:
    client.closed = True


def test_a_value_is_built_once_per_runner_loop_and_closed_with_it() -> None:
    built: list[Client] = []

    def build() -> Client:
        built.append(Client())
        return built[-1]

    local = loops.LoopLocal(build, close_client)

    async def use() -> Client:
        return local.get()

    first = run_coroutine(use())
    assert run_coroutine(use()) is first
    close_thread_loop()

    assert first.closed
    assert run_coroutine(use()) is not first
    assert len(built) == 2


def test_a_value_built_on_a_loop_ol_llm_does_not_own_is_dropped_not_closed() -> None:
    local = loops.LoopLocal(Client, close_client)

    async def use() -> Client:
        return local.get()

    first, second = asyncio.run(use()), asyncio.run(use())

    assert first is not second
    assert not first.closed


def test_a_close_that_fails_does_not_stop_the_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail(_client: Client) -> None:
        raise RuntimeError

    failing = loops.LoopLocal(Client, fail)
    closing = loops.LoopLocal(Client, close_client)

    async def use() -> Client:
        failing.get()
        return closing.get()

    client = run_coroutine(use())
    close_thread_loop()

    assert client.closed
    assert "Failed to close a loop-local client" in caplog.text


def test_closing_the_thread_loop_closes_every_provider_client(
    real_client: UseCase,
) -> None:
    # Start from a loop no earlier test built values on.
    close_thread_loop()
    complete_sync(real_client, "hi")
    loop = runner_loop()
    built: list[Model] = [
        local._values[loop]  # noqa: SLF001
        for local in list(loops._instances)  # noqa: SLF001
        if loop in local._values  # noqa: SLF001
    ]
    assert len(built) == len(real_client.model_ids)

    close_thread_loop()

    assert all(client_is_closed(model) for model in built)


def client_is_closed(model: Model) -> bool:
    provider = model._provider  # noqa: SLF001
    if (own := provider._own_http_client) is not None:  # noqa: SLF001
        return bool(own.is_closed)
    return bool(provider.client.is_closed())


def test_a_forked_child_drops_the_parents_loop_and_values() -> None:
    local = loops.LoopLocal(Client, close_client)

    async def use() -> Client:
        return local.get()

    parent = run_coroutine(use())
    parent_loop = runner_loop()

    def in_child() -> tuple[bool, bool, bool]:
        child = run_coroutine(use())
        return child is parent, runner_loop() is parent_loop, parent.closed

    assert run_in_fork(in_child) == (False, False, False)


def after_the_loop_blocks() -> str:
    # Long enough that the loop is waiting on the self-pipe when this returns.
    time.sleep(0.05)
    return "woken"


async def woken_via_a_thread() -> str:
    """Finish only if another thread can wake the running loop."""
    return await asyncio.wait_for(asyncio.to_thread(after_the_loop_blocks), 5)


def test_the_parents_loop_still_wakes_after_a_child_collects_garbage() -> None:
    # The child shares the parent loop's epoll instance; closing the inherited
    # loop there would unregister the parent's self-pipe.
    assert run_coroutine(woken_via_a_thread()) == "woken"
    run_in_fork(gc.collect)

    assert run_coroutine(woken_via_a_thread()) == "woken"


def test_a_worker_threads_loop_still_wakes_after_a_child_collects_garbage() -> None:
    # The worker thread doesn't survive the fork, so its loop is not the forking
    # thread's, but it shares the same epoll instance with the parent.
    with ThreadPoolExecutor(1) as worker:
        assert worker.submit(run_coroutine, woken_via_a_thread()).result() == "woken"
        run_in_fork(gc.collect)

        assert worker.submit(run_coroutine, woken_via_a_thread()).result() == "woken"
        worker.submit(close_thread_loop).result()


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (models, "_shared_models_lock"),
        (embeddings, "_shared_embedders_lock"),
        (settings_module, "_lock"),
        (prompts, "_inflight_lock"),
        (tracing, "_lock"),
    ],
)
def test_a_lock_another_thread_holds_at_the_fork_is_free_in_the_child(
    module: ModuleType, name: str
) -> None:
    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with getattr(module, name):
            holding.set()
            release.wait()

    holder = threading.Thread(target=hold)
    holder.start()
    holding.wait()
    try:
        assert run_in_fork(lambda: getattr(module, name).acquire(timeout=1))
    finally:
        release.set()
        holder.join()


def test_a_forked_child_starts_no_prompt_fetch_from_the_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The thread that would resolve it did not survive the fork.
    monkeypatch.setitem(prompts._inflight, "lookup", Future())  # type: ignore[index]  # noqa: SLF001

    assert run_in_fork(lambda: dict(prompts._inflight)) == {}  # noqa: SLF001


def test_concurrent_first_bedrock_calls_build_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Clients come from boto3's default session, which isn't thread-safe.
    built: list[object] = []

    def slow_client(_policy: object) -> object:
        time.sleep(0.05)
        built.append(client := object())
        return client

    monkeypatch.setattr(models, "_bedrock_client", slow_client)
    monkeypatch.setattr(models, "_bedrock_provider", lambda client: client)
    use_case = UseCase(
        app="mit-learn", name="summary", profile="request", model=PRIMARY
    )
    factory = models.provider_factory(
        use_case, resolve_policy(use_case), Settings(environment="ci")
    )

    with ThreadPoolExecutor(2) as pool:
        clients = list(pool.map(lambda _: factory("bedrock"), range(2)))

    assert len(built) == 1
    assert clients[0] is clients[1]


@pytest.mark.usefixtures("provider_env")
def test_a_bedrock_client_never_uses_boto3s_default_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each model's factory builds its own client, possibly on several threads at once.
    import boto3  # noqa: PLC0415

    def shared(*_args: object, **_kwargs: object) -> None:
        pytest.fail("built from boto3's default session")

    monkeypatch.setattr(boto3, "_get_default_session", shared)
    use_case = UseCase(
        app="mit-learn", name="summary", profile="request", model=PRIMARY
    )

    assert models._bedrock_client(resolve_policy(use_case)) is not None  # noqa: SLF001


def test_no_loop_is_created_between_the_snapshot_and_the_fork() -> None:
    # Later before-fork hooks (logging's among them) can release the GIL.
    loops._before_fork()  # noqa: SLF001
    try:
        with ThreadPoolExecutor(1) as worker:
            first_call = worker.submit(run_coroutine, asyncio.sleep(0))
            time.sleep(0.2)
            assert not first_call.done()
            loops._after_fork_in_parent()  # noqa: SLF001
            first_call.result(timeout=5)
            worker.submit(close_thread_loop).result()
    finally:
        if loops._loops_lock.locked():  # noqa: SLF001
            loops._loops_lock.release()  # noqa: SLF001


def test_a_thread_creating_a_loop_can_still_fork_and_call_ol_llm() -> None:
    # As a signal handler would, from inside the thread's own loop creation.
    def reenter() -> None:
        with loops._loops_lock:  # noqa: SLF001
            loops._before_fork()  # noqa: SLF001
            loops._after_fork_in_parent()  # noqa: SLF001
            run_coroutine(asyncio.sleep(0))
            close_thread_loop()

    thread = threading.Thread(target=reenter, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()


def test_without_register_at_fork_no_hook_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Windows has no fork, and no os.register_at_fork.
    monkeypatch.delattr(os, "register_at_fork")

    loops.register_at_fork(after_in_child=lambda: None)


@pytest.mark.usefixtures("provider_env")
def test_a_forked_child_builds_its_own_bedrock_client() -> None:
    bedrock = UseCase(
        app="mit-learn",
        name="summary",
        profile="request",
        model="bedrock:us.anthropic.claude-sonnet-5",
    )
    model = shared_model(bedrock)

    async def client_id() -> int:
        return id(model._loop_model()._provider.client)  # type: ignore[attr-defined]  # noqa: SLF001

    parent = run_coroutine(client_id())

    assert run_in_fork(lambda: run_coroutine(client_id())) != parent


# Tracing in a forked child.


class RecordingOpik(BaseHTTPRequestHandler):
    """Records every request body by path, and answers Keycloak token requests."""

    protocol_version = "HTTP/1.1"
    received: list[tuple[str, bytes]]

    def _answer(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        self.received.append((self.path, body))
        reply = (
            json.dumps({"access_token": "token", "expires_in": 300}).encode()
            if self.path.endswith("/token")
            else b"{}"
        )
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    do_GET = do_POST = do_PATCH = do_PUT = _answer  # noqa: N815

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def recording_opik(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[str, list[tuple[str, bytes]]]]:
    received: list[tuple[str, bytes]] = []
    handler = type("Handler", (RecordingOpik,), {"received": received})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(tracing, "_exporter", REAL_EXPORTER)
    # Set and restored here rather than by monkeypatch, which restores after the
    # autouse ``exported`` fixture's flush has built the SDK's next client.
    previous = os.environ["OPIK_URL_OVERRIDE"]
    os.environ["OPIK_URL_OVERRIDE"] = f"{url}/api"
    # The SDK's client reads its URL when it is built.
    reset_global_client()
    yield url, received
    os.environ["OPIK_URL_OVERRIDE"] = previous
    tracing.reset()
    reset_global_client()
    server.shutdown()
    server.server_close()


def replying(text: str) -> FunctionModel:
    def reply(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(reply)


def test_a_forked_child_sends_its_traces_and_spans(
    recording_opik: tuple[str, list[tuple[str, bytes]]],
) -> None:
    url, received = recording_opik
    tracing.reset()
    configure_tracing(
        TracingConfig(
            keycloak=KeycloakClient(
                token_url=f"{url}/token", client_id="app", client_secret="s"
            )
        )
    )
    with override_models(replying("ok")):
        complete_sync(CHAT, "from the parent")
    assert tracing.flush(10_000)
    tokens = tracing._state().tokens  # noqa: SLF001
    assert tokens is not None
    parent_http = tokens._http  # noqa: SLF001

    def in_child() -> bool:
        with override_models(replying("ok")):
            complete_sync(CHAT, "from the child")
        flushed = tracing.flush(10_000)
        return flushed and tokens._http is not parent_http  # noqa: SLF001

    assert run_in_fork(in_child)
    from_child = [path for path, body in received if b"from the child" in body]
    assert any("/v1/private/traces" in path for path in from_child), from_child
    assert any(path.endswith(OTLP_TRACES_PATH) for path in from_child), from_child
