import gzip
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Generator
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import ol_llm.catalog as catalog_module
import pytest
from ol_llm import GatewayRoute, UseCase, models, tracing
from ol_llm.catalog import Catalog, CatalogEntry, load_catalog
from ol_llm.settings import Settings, configure
from ol_llm.tracing import TracingConfig, configure_tracing
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opik.api_objects.prompt import prompt_cache

_PROMPTS = "/api/v1/private/prompts/versions"
_ENVIRONMENTS = re.compile(rf"{_PROMPTS}/(?P<id>[^/]+)/environments")


class PromptLibrary:
    """Opik's prompt-version endpoints: what the SDK's fetch, create, and label use.

    Versions are kept per project and prompt name, newest last.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.prompts: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
            self.environments = {"development", "staging", "production"}
            self.retrievals = 0
            self.delay_seconds = 0.0

    def add(  # noqa: PLR0913
        self,
        name: str,
        template: str | list[dict[str, Any]],
        *,
        project_name: str = "mit-learn",
        environments: tuple[str, ...] = (),
        type: str = "mustache",  # noqa: A002
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a version, as someone editing in the Opik UI would."""
        chat = not isinstance(template, str)
        with self._lock:
            return self._create(
                project_name,
                name,
                json.dumps(template) if chat else template,
                type_=type,
                structure="chat" if chat else "text",
                metadata=metadata,
                environments=list(environments),
            )

    def versions(self, name: str, project_name: str = "mit-learn") -> list[str]:
        with self._lock:
            return [v["version_number"] for v in self.prompts[project_name, name]]

    def labels(
        self, name: str, project_name: str = "mit-learn"
    ) -> dict[str, list[str]]:
        with self._lock:
            return {
                v["version_number"]: v["environments"]
                for v in self.prompts[project_name, name]
            }

    def handle(self, method: str, path: str, body: bytes) -> tuple[int, Any] | None:
        """Answer a prompt request, or return ``None`` for any other path."""
        if not path.startswith(_PROMPTS):
            return None
        request = json.loads(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)
        if method == "POST" and path == f"{_PROMPTS}/retrieve":
            time.sleep(self.delay_seconds)
            with self._lock:
                self.retrievals += 1
                return self._retrieve(request)
        with self._lock:
            if method == "POST" and path == _PROMPTS:
                return 200, self._create(
                    request.get("project_name"),
                    request["name"],
                    request["version"]["template"],
                    type_=request["version"].get("type") or "mustache",
                    structure=request.get("template_structure") or "text",
                    metadata=request["version"].get("metadata"),
                    environments=[],
                )
            if method == "PATCH" and (match := _ENVIRONMENTS.fullmatch(path)):
                return self._label(match["id"], request["environments"])
        return 404, {"errors": [f"no fake for {method} {path}"]}

    def _create(  # noqa: PLR0913
        self,
        project_name: str | None,
        name: str,
        template: str,
        *,
        type_: str,
        structure: str,
        metadata: dict[str, Any] | None,
        environments: list[str],
    ) -> dict[str, Any]:
        versions = self.prompts.setdefault((project_name, name), [])
        version_id = str(uuid.uuid4())
        detail = {
            "id": version_id,
            "prompt_id": versions[0]["prompt_id"] if versions else str(uuid.uuid4()),
            "commit": version_id[:8],
            "version_number": f"v{len(versions) + 1}",
            "template": template,
            "type": type_,
            "template_structure": versions[0]["template_structure"]
            if versions
            else structure,
            "metadata": metadata,
            "environments": [],
        }
        versions.append(detail)
        self._label(version_id, environments)
        return detail

    def _retrieve(self, request: dict[str, Any]) -> tuple[int, Any]:
        versions = self.prompts.get((request.get("project_name"), request["name"]), [])
        if environment := request.get("environment"):
            found = [v for v in versions if environment in v["environments"]]
        elif number := request.get("version_number"):
            found = [v for v in versions if v["version_number"] == number]
        else:
            found = versions[-1:]
        if not found:
            return 404, {"errors": ["not found"]}
        return 200, found[0]

    def _label(self, version_id: str, environments: list[str]) -> tuple[int, Any]:
        if set(environments) - self.environments:
            return 404, {"errors": ["environment not found"]}
        for versions in self.prompts.values():
            if not any(v["id"] == version_id for v in versions):
                continue
            for version in versions:
                if version["id"] == version_id:
                    version["environments"] = list(environments)
                else:
                    version["environments"] = [
                        e for e in version["environments"] if e not in environments
                    ]
            return 204, None
        return 404, {"errors": ["version not found"]}


PROMPTS = PromptLibrary()


class AcceptingOpik(BaseHTTPRequestHandler):
    """Accepts every SDK request, so the SDK has nothing to queue and replay."""

    def _accept(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if (answer := PROMPTS.handle(self.command, self.path, body)) is not None:
            status, payload = answer
            data = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(204 if self.command != "GET" else 200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._accept()

    def do_POST(self) -> None:
        self._accept()

    def do_PATCH(self) -> None:
        self._accept()

    def do_PUT(self) -> None:
        self._accept()

    def log_message(self, *_args: object) -> None:
        pass


_opik = ThreadingHTTPServer(("127.0.0.1", 0), AcceptingOpik)
threading.Thread(target=_opik.serve_forever, daemon=True).start()
OPIK_URL = f"http://127.0.0.1:{_opik.server_address[1]}/api"

# The SDK reads these when it first builds its client, which no import does.
# Tests read root traces with opik.record_traces_locally and model spans from the
# in-memory exporter below.
os.environ["OPIK_URL_OVERRIDE"] = OPIK_URL
os.environ["OPIK_WORKSPACE"] = "default"


TRACING = TracingConfig()


@pytest.fixture(autouse=True)
def test_settings() -> Generator[Settings]:
    settings = Settings(environment="ci")
    configure(settings)
    yield settings
    configure(settings)


@pytest.fixture(autouse=True)
def clear_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("OL_LLM_", "OPIK_KEYCLOAK_")):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff waits instead of sleeping."""
    recorded: list[float] = []

    async def record(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(models, "_sleep", record)
    return recorded


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


@pytest.fixture(autouse=True)
def exported(monkeypatch: pytest.MonkeyPatch) -> Generator[InMemorySpanExporter]:
    """Keep model spans in memory in place of sending them to Opik.

    Call ``tracing.flush`` before reading: the batch processor sends in the
    background.
    """
    exporter = InMemorySpanExporter()

    def in_memory(_app: str, _state: object) -> SpanExporter:
        return exporter

    monkeypatch.setattr(tracing, "_exporter", in_memory)
    tracing.reset()
    configure_tracing(TRACING)
    yield exporter
    # Send what the test queued, so a later test's recording does not see it.
    tracing.flush()
    tracing.reset()


# A real provider client against a local server that keeps connections alive, so
# a second call reuses the first call's pooled connection.

RESPONSES_REPLY = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}
GEMINI_REPLY = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 1,
        "candidatesTokenCount": 1,
        "totalTokenCount": 2,
    },
}
PRIMARY = "openai:gpt-4o-mini"
# Not in the released catalog; the fixture adds it.
GEMINI = "google:gemini-3.1-pro-preview"


class KeepAliveProvider(BaseHTTPRequestHandler):
    """Answers OpenAI Responses and Gemini requests over persistent connections.

    Paths under ``/unavailable`` answer 503, so a chain falls back. Gemini's
    streaming endpoint answers with one server-sent event.
    """

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        status, content_type = 200, "application/json"
        if self.path.startswith("/unavailable"):
            status, body = 503, b"{}"
        elif "streamGenerateContent" in self.path:
            content_type = "text/event-stream"
            body = f"data: {json.dumps(GEMINI_REPLY)}\r\n\r\n".encode()
        else:
            reply = GEMINI_REPLY if "generateContent" in self.path else RESPONSES_REPLY
            body = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture(
    params=[(PRIMARY,), (GEMINI,), (PRIMARY, GEMINI)],
    ids=["openai", "google", "openai-then-google"],
)
def real_client(
    provider_env: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Generator[UseCase]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), KeepAliveProvider)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", url)
    gemini = CatalogEntry(
        id=GEMINI,
        kind="chat",
        input_usd_per_mtok=Decimal(1),
        output_usd_per_mtok=Decimal(1),
    )
    catalog = Catalog({**load_catalog().entries, GEMINI: gemini})
    monkeypatch.setattr(catalog_module, "load_catalog", lambda: catalog)
    model, *fallbacks = request.param
    # A chain's primary is unavailable, so every call is answered by a fallback.
    prefix = "/unavailable" if fallbacks else ""
    route = GatewayRoute(base_url=f"{url}{prefix}/v1", api_key="app-key")
    configure(Settings(environment="ci", gateway_routes={"openai": route}))
    yield UseCase(
        app="mit-learn",
        name="summary",
        profile="request",
        model=model,
        fallback_models=tuple(fallbacks),
    )
    server.shutdown()
    server.server_close()


@pytest.fixture
def prompt_library() -> Generator[PromptLibrary]:
    """Opik's prompt library, empty, with the SDK's prompt cache cleared."""
    PROMPTS.reset()
    prompt_cache.get_global_cache().clear()
    yield PROMPTS
    prompt_cache.get_global_cache().clear()
    PROMPTS.reset()
