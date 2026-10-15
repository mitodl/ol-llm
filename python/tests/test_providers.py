"""Provider construction and the single retry layer against a local mock server.

The attempt counts depend on SDK defaults that can change in any release (spec
§6.3), so these tests make real HTTP requests to a server on localhost and count
what arrives.
"""

import json
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from ol_llm import LLMRejected, LLMUnavailable, UseCase
from ol_llm.attribution import USE_CASE_HEADER
from ol_llm.models import PolicyModel, build_agent, build_model
from ol_llm.settings import GatewayRoute, Settings, configure
from pydantic_ai.providers import mistral

HANG_SECONDS = 1.5
TEST_TIMEOUT_SECONDS = "0.3"


def use_case(model: str) -> UseCase:
    return UseCase(app="mit-learn", name="summary", profile="request", model=model)


@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
        "OPENAI_API_VERSION": "2024-10-21",
        "GOOGLE_API_KEY": "test-key",
        "MISTRAL_API_KEY": "test-key",
        "TOGETHER_API_KEY": "test-key",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
    }.items():
        monkeypatch.setenv(name, value)


def provider_client(model_id: str) -> Any:  # noqa: ANN401
    model = build_model(use_case(model_id))
    assert isinstance(model, PolicyModel)
    return model.wrapped.provider.client  # type: ignore[union-attr]


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
def test_bedrock_client_carries_the_timeout_and_no_retries() -> None:
    client = provider_client("bedrock:us.anthropic.claude-sonnet-5")
    config = client._client_config  # noqa: SLF001

    assert config.retries["total_max_attempts"] == 1
    assert config.read_timeout == 60
    assert config.connect_timeout == 60


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize(
    "model_id", ["google:gemini-3-pro-preview", "mistral:mistral-large-latest"]
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


@pytest.mark.parametrize(
    "model_id", ["gpt-4o-mini", "openai:", ":gpt-4o-mini", "groq:llama-3.3-70b"]
)
def test_unsupported_model_ids_are_refused(model_id: str) -> None:
    with pytest.raises(ValueError, match=r"provider:model|does not build"):
        build_model(use_case(model_id))


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
    "google:gemini-3-pro-preview",
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
