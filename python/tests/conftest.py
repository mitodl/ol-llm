import os
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from ol_llm import models, tracing
from ol_llm.settings import Settings, configure
from ol_llm.tracing import TracingConfig, configure_tracing
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class AcceptingOpik(BaseHTTPRequestHandler):
    """Accepts every SDK request, so the SDK has nothing to queue and replay."""

    def _accept(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
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
