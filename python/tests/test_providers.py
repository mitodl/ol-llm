"""Provider construction and the single retry layer against a local mock server.

The attempt counts depend on SDK defaults that can change in any release (spec
§6.3), so these tests make real HTTP requests to a server on localhost and count
what arrives.
"""

import asyncio
import gc
import json
import threading
import time
import weakref
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import ol_llm.catalog as catalog_module
import pytest
from botocore.exceptions import EventStreamError
from ol_llm import LLMRateLimited, LLMRejected, LLMUnavailable, UseCase, run_sync
from ol_llm.attribution import USE_CASE_HEADER
from ol_llm.catalog import Catalog, CatalogEntry, load_catalog
from ol_llm.embeddings import PolicyEmbeddingModel, build_embedder
from ol_llm.models import PolicyModel, build_agent, build_model
from ol_llm.settings import GatewayRoute, Settings, configure
from pydantic_ai.providers import mistral

HANG_SECONDS = 1.5
TEST_TIMEOUT_SECONDS = "0.3"


# Models the catalog does not list, so construction is tested for every provider
# the library builds and refused for one it does not.
UNLISTED_MODELS = (
    "anthropic:claude-haiku-4-5",
    "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "google:gemini-3.1-pro-preview",
    "groq:llama-3.3-70b",
)
# An embedding model for a provider no consumer embeds with yet.
UNLISTED_EMBEDDING_MODEL = "google:gemini-embedding-001"


@pytest.fixture(autouse=True)
def provider_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    released = load_catalog()
    unlisted = {
        model_id: CatalogEntry(
            id=model_id,
            kind="chat",
            input_usd_per_mtok=Decimal(1),
            output_usd_per_mtok=Decimal(1),
        )
        for model_id in UNLISTED_MODELS
    }
    unlisted[UNLISTED_EMBEDDING_MODEL] = CatalogEntry(
        id=UNLISTED_EMBEDDING_MODEL,
        kind="embedding",
        dimensions=3072,
        input_usd_per_mtok=Decimal("0.15"),
        output_usd_per_mtok=Decimal(0),
    )
    catalog = Catalog({**released.entries, **unlisted})
    monkeypatch.setattr(catalog_module, "load_catalog", lambda: catalog)


def use_case(model: str) -> UseCase:
    return UseCase(app="mit-learn", name="summary", profile="request", model=model)


def provider_client(model_id: str) -> Any:  # noqa: ANN401
    model = build_model(use_case(model_id))
    assert isinstance(model, PolicyModel)
    return model.wrapped.provider.client  # type: ignore[union-attr]


def embedding_client(model_id: str) -> Any:  # noqa: ANN401
    model = build_embedder(use_case(model_id)).model
    assert isinstance(model, PolicyEmbeddingModel)
    return model.wrapped._provider.client  # type: ignore[attr-defined]  # noqa: SLF001


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize(
    "model_id",
    [
        "openai:gpt-4o-mini",
        "openai-chat:gpt-4o-mini",
        "azure:gpt-4o",
        "anthropic:claude-haiku-4-5",
        "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ],
)
def test_sdk_retries_are_off(model_id: str) -> None:
    assert provider_client(model_id).max_retries == 0


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize(
    "model_id", ["openai:text-embedding-3-large", "azure:text-embedding-3-large"]
)
def test_embedding_clients_have_no_sdk_retries_and_the_timeout(model_id: str) -> None:
    # Embedding requests take no per-request timeout, so it is on the client.
    client = embedding_client(model_id)

    assert client.max_retries == 0
    assert client.timeout == 60


@pytest.mark.usefixtures("provider_env")
def test_bedrock_embedding_client_carries_the_timeout_and_no_retries() -> None:
    config = embedding_client("bedrock:amazon.titan-embed-text-v2:0")._client_config  # noqa: SLF001

    assert config.retries["total_max_attempts"] == 1
    assert config.read_timeout == 60


@pytest.mark.usefixtures("provider_env")
def test_google_client_carries_the_timeout() -> None:
    client = embedding_client(UNLISTED_EMBEDDING_MODEL)
    http_options = client._api_client._http_options  # noqa: SLF001

    assert http_options.timeout == 60_000
    assert http_options.retry_options is None


@pytest.mark.usefixtures("provider_env")
def test_bedrock_client_carries_the_timeout_and_no_retries() -> None:
    client = provider_client("bedrock:us.anthropic.claude-sonnet-5")
    config = client._client_config  # noqa: SLF001

    assert config.retries["total_max_attempts"] == 1
    assert config.read_timeout == 60
    assert config.connect_timeout == 60


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize(
    "model_id", ["google:gemini-3.1-pro-preview", "mistral:mistral-large-latest"]
)
def test_single_attempt_providers_build(model_id: str) -> None:
    assert isinstance(build_model(use_case(model_id)), PolicyModel)


@pytest.mark.usefixtures("provider_env")
def test_direct_calls_send_no_gateway_header() -> None:
    client = provider_client("openai:gpt-4o-mini")

    assert str(client.base_url) == "https://api.openai.com/v1/"
    assert USE_CASE_HEADER not in client.default_headers


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize(
    ("family", "model_id"),
    [("openai", "openai:gpt-4o-mini"), ("anthropic", "anthropic:claude-haiku-4-5")],
)
def test_gateway_route_replaces_the_vendor_api(family: str, model_id: str) -> None:
    route = GatewayRoute(base_url="https://gateway.example/llm/", api_key="app-key")
    configure(Settings(environment="qa", gateway_routes={family: route}))
    client = provider_client(model_id)

    assert str(client.base_url) == "https://gateway.example/llm/"
    assert client.api_key == "app-key"
    assert client.default_headers[USE_CASE_HEADER] == "summary"


@pytest.mark.usefixtures("provider_env")
def test_the_google_provider_closes_its_client_and_opens_a_new_one() -> None:
    model = build_model(use_case("google:gemini-3.1-pro-preview"))
    assert isinstance(model, PolicyModel)
    provider = model.wrapped.provider  # type: ignore[attr-defined]

    async def enter_and_exit() -> Any:  # noqa: ANN401
        async with provider:
            return provider.client._api_client._async_httpx_client  # noqa: SLF001

    first = asyncio.run(enter_and_exit())
    second = asyncio.run(enter_and_exit())

    assert first.is_closed
    assert second is not first
    assert second.is_closed
    assert second.timeout.read == 60


def test_unsupported_provider_is_refused() -> None:
    with pytest.raises(ValueError, match="does not build 'groq'"):
        build_model(use_case("groq:llama-3.3-70b"))


# The mock server


@dataclass
class MockProvider:
    mode: str = "unavailable"
    requests: list[dict[str, str]] = field(default_factory=list)
    url: str = ""


@pytest.fixture
def mock_provider() -> Generator[MockProvider]:
    state = MockProvider()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("content-length", 0)))
            state.requests.append(dict(self.headers.items()))
            if state.mode == "hang":
                time.sleep(HANG_SECONDS)
                return
            status = 503 if state.mode == "unavailable" else 400
            body = json.dumps({"error": {"message": state.mode, "type": state.mode}})
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def point_at(
    mock: MockProvider, model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = model_id.partition(":")[0]
    if family == "bedrock":
        monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", mock.url)
        return
    if family == "google":
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", mock.url)
        return
    if family == "mistral":
        # MistralProvider takes a base URL only as an argument.
        class LocalMistral(mistral.MistralProvider):
            def __init__(self) -> None:
                super().__init__(base_url=mock.url)

        monkeypatch.setattr(mistral, "MistralProvider", LocalMistral)
        return
    base_url = f"{mock.url}/v1" if family == "openai" else mock.url
    route = GatewayRoute(base_url=base_url, api_key="app-key")
    configure(Settings(environment="ci", gateway_routes={family: route}))


MOCKED_MODELS = [
    "openai:gpt-4o-mini",
    "anthropic:claude-haiku-4-5",
    "bedrock:us.anthropic.claude-sonnet-5",
    "google:gemini-3.1-pro-preview",
    "mistral:mistral-large-latest",
]


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize("model_id", MOCKED_MODELS)
@pytest.mark.parametrize(
    ("mode", "attempts", "error"),
    [
        ("unavailable", 3, LLMUnavailable),
        ("hang", 3, LLMUnavailable),
        ("rejected", 1, LLMRejected),
    ],
)
def test_one_retry_layer_per_provider(  # noqa: PLR0913, PLR0917
    mock_provider: MockProvider,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    mode: str,
    attempts: int,
    error: type[Exception],
) -> None:
    monkeypatch.setenv("OL_LLM_SUMMARY_TIMEOUT_SECONDS", TEST_TIMEOUT_SECONDS)
    point_at(mock_provider, model_id, monkeypatch)
    mock_provider.mode = mode

    with pytest.raises(error):
        build_agent(use_case(model_id)).run_sync("hi")

    assert len(mock_provider.requests) == attempts


# Bedrock's timeout is the boto3 client's, so it gets no timeout setting; the hang
# case above shows the client still enforces it.
@pytest.mark.usefixtures("provider_env")
@pytest.mark.filterwarnings("error::pydantic_ai.UnsupportedSettingWarning")
@pytest.mark.parametrize("model_id", MOCKED_MODELS)
def test_requests_send_no_setting_the_model_ignores(
    mock_provider: MockProvider, monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    point_at(mock_provider, model_id, monkeypatch)
    mock_provider.mode = "rejected"

    with pytest.raises(LLMRejected):
        build_agent(use_case(model_id)).run_sync("hi", model_settings={"timeout": 9})

    assert len(mock_provider.requests) == 1


MOCKED_EMBEDDING_MODELS = [
    "openai:text-embedding-3-large",
    "bedrock:amazon.titan-embed-text-v2:0",
    UNLISTED_EMBEDDING_MODEL,
]


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize("model_id", MOCKED_EMBEDDING_MODELS)
@pytest.mark.parametrize(
    ("mode", "attempts", "error"),
    [
        ("unavailable", 3, LLMUnavailable),
        ("hang", 3, LLMUnavailable),
        ("rejected", 1, LLMRejected),
    ],
)
def test_one_retry_layer_per_embedding_provider(  # noqa: PLR0913, PLR0917
    mock_provider: MockProvider,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    mode: str,
    attempts: int,
    error: type[Exception],
) -> None:
    monkeypatch.setenv("OL_LLM_SUMMARY_TIMEOUT_SECONDS", TEST_TIMEOUT_SECONDS)
    point_at(mock_provider, model_id, monkeypatch)
    mock_provider.mode = mode

    with pytest.raises(error):
        build_embedder(use_case(model_id)).embed_sync(["hi"], input_type="query")

    assert len(mock_provider.requests) == attempts


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize("model_id", MOCKED_MODELS[:2])
def test_gateway_requests_carry_the_use_case(
    mock_provider: MockProvider, monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    point_at(mock_provider, model_id, monkeypatch)
    mock_provider.mode = "rejected"

    with pytest.raises(LLMRejected):
        build_agent(use_case(model_id)).run_sync("hi")

    headers = {k.lower(): v for k, v in mock_provider.requests[0].items()}
    assert headers[USE_CASE_HEADER.lower()] == "summary"


# Bedrock hands back its stream before reading an event, so these fake the event
# stream itself: a throttle on the first read must be retried like one on opening.
THROTTLED = EventStreamError(
    {"Error": {"Code": "throttlingException", "Message": "slow down"}},
    "ConverseStream",
)
ANSWER = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
    {"contentBlockStop": {"contentBlockIndex": 0}},
    {"messageStop": {"stopReason": "end_turn"}},
    {
        "metadata": {
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        }
    },
]


@dataclass
class FakeEventStream:
    events: list[object]
    closed: bool = False

    def __iter__(self) -> Iterator[object]:
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event

    def close(self) -> None:
        self.closed = True


def stream_bedrock(
    monkeypatch: pytest.MonkeyPatch, *responses: list[object]
) -> tuple[Any, list[FakeEventStream]]:
    agent = build_agent(use_case("bedrock:us.anthropic.claude-sonnet-5"))
    opened: list[FakeEventStream] = []

    def converse_stream(**_params: object) -> dict[str, object]:
        stream = FakeEventStream(responses[min(len(opened), len(responses) - 1)])
        opened.append(stream)
        return {"stream": stream, "ResponseMetadata": {"RequestId": "r"}}

    client = agent.model.wrapped.provider.client  # type: ignore[union-attr]
    monkeypatch.setattr(client, "converse_stream", converse_stream)
    return agent, opened


async def consume(agent: Any) -> str:  # noqa: ANN401
    async with agent.run_stream("hi") as stream:
        output: str = await stream.get_output()
        return output


@pytest.mark.usefixtures("provider_env")
def test_bedrock_stream_retries_a_failed_first_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, opened = stream_bedrock(monkeypatch, [THROTTLED], ANSWER)

    assert asyncio.run(consume(agent)) == "ok"
    assert len(opened) == 2
    assert opened[0].closed


@pytest.mark.usefixtures("provider_env")
def test_bedrock_stream_gives_up_after_its_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, opened = stream_bedrock(monkeypatch, [THROTTLED])

    with pytest.raises(LLMRateLimited):
        asyncio.run(consume(agent))
    assert len(opened) == 3


@pytest.mark.usefixtures("provider_env")
def test_bedrock_stream_error_is_retried_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_error = EventStreamError(
        {"Error": {"Code": "modelStreamErrorException", "Message": "retry"}},
        "ConverseStream",
    )
    agent, opened = stream_bedrock(monkeypatch, [stream_error])

    with pytest.raises(LLMUnavailable):
        asyncio.run(consume(agent))
    assert len(opened) == 3


@pytest.mark.usefixtures("provider_env")
def test_bedrock_throttle_after_the_first_chunk_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, opened = stream_bedrock(monkeypatch, [*ANSWER[:2], THROTTLED])

    with pytest.raises(LLMRateLimited):
        asyncio.run(consume(agent))
    assert len(opened) == 1


# An agent refreshes its model's HTTP client at the start of every run, and a
# provider that made its own client (Google, Mistral) records the loop it ran on.
@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize("model_id", MOCKED_MODELS)
def test_a_finished_threads_loop_is_released_by_every_provider(
    mock_provider: MockProvider, monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    point_at(mock_provider, model_id, monkeypatch)
    mock_provider.mode = "rejected"
    agent = build_agent(use_case(model_id))
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []
    errors: list[BaseException] = []

    def call() -> None:
        try:
            run_sync(agent, "hi")
        except LLMRejected:
            loops.append(weakref.ref(asyncio.get_event_loop()))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    gc.collect()

    assert errors == []
    assert loops[0]() is None


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize(
    ("model_id", "shares_client"),
    [
        ("openai:gpt-4o-mini", False),
        ("google:gemini-3.1-pro-preview", False),
        ("bedrock:us.anthropic.claude-sonnet-5", True),
    ],
)
def test_each_event_loop_gets_its_own_provider(
    model_id: str, *, shares_client: bool
) -> None:
    model = build_model(use_case(model_id))
    assert isinstance(model, PolicyModel)

    async def loop_model() -> Any:  # noqa: ANN401
        return model._loop_model()  # noqa: SLF001

    first, second = (asyncio.run(loop_model()) for _ in range(2))

    assert first.provider is not second.provider
    assert (first.provider.client is second.provider.client) is shares_client
