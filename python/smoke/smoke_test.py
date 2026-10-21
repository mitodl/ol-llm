"""Import-and-trace check for ol-llm installed with the opik SDK slimmed.

Runs as a plain script, not under pytest: pytest is one of the packages the
overrides drop. Exits non-zero on any failure.

What it proves, in an environment shaped like a consumer's runtime image:
- ol_llm and opik import with litellm, openai, pytest, and tree-sitter absent;
- ol_llm.run_sync over an agent from ol_llm.build_agent, ol_llm.complete_sync, and
  ol_llm.run_stream send the SDK root trace and the OTLP model spans to Opik, and
  the spans carry the root trace's id (for a stream, while the caller reads it);
- ol_llm.embed_sync does the same for its embedding span;
- both halves authenticate with a Keycloak client-credentials token, and the span
  exporter fetches a new token after a 401;
- the spans are filed under the app's project, or under the caller's project for
  a run inside the caller's own @opik.track, where Opik files that run's trace;
- ol_llm.get_prompt fetches a prompt by label through the SDK, and the run links
  its version to the root trace;
- importing ol_llm first leaves no Sentry client running and opik's usage
  analytics switched off.

Opik and Keycloak are one fake server in this process, which records what it is
sent. The first OTLP request gets a 401, as an expired token would.
"""

import asyncio
import gzip
import json
import os
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DROPPED = ("litellm", "openai", "pytest", "tree_sitter", "click")
APP = "ol-llm-smoke"
CALLER = "ol-llm-smoke-caller"
OTLP_PATH = "/api/v1/private/otel/v1/traces"
TOKEN_PATH = "/token"  # noqa: S105
PROMPT_PATH = "/api/v1/private/prompts/versions/retrieve"
PROMPT_VERSION = {
    "id": "00000000-0000-7000-8000-000000000002",
    "prompt_id": "00000000-0000-7000-8000-000000000001",
    "commit": "00000002",
    "version_number": "v2",
    "template": "Reply {{word}}.",
    "type": "mustache",
    "template_structure": "text",
    "environments": ["production"],
}


@dataclass
class Request:
    """One request the fake server received."""

    path: str
    headers: dict[str, str]
    body: bytes


class FakeOpik(BaseHTTPRequestHandler):
    """Opik's API and Keycloak's token endpoint, recording every request."""

    received: list[Request] = []  # noqa: RUF012
    tokens_issued = 0
    otlp_refused = False

    def do_GET(self) -> None:
        """Record and accept."""
        self._handle()

    def do_POST(self) -> None:
        """Record and accept, except the first OTLP export."""
        self._handle()

    def do_PATCH(self) -> None:
        """Record and accept."""
        self._handle()

    def do_PUT(self) -> None:
        """Record and accept."""
        self._handle()

    def _handle(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        cls = type(self)
        if self.path == TOKEN_PATH:
            cls.tokens_issued += 1
            self._reply(200, {"access_token": f"token-{cls.tokens_issued}"})
            return
        # HTTP header names are case-insensitive; the OTLP client lowercases them.
        headers = {name.lower(): value for name, value in self.headers.items()}
        cls.received.append(Request(self.path, headers, body))
        if self.path == OTLP_PATH and not cls.otlp_refused:
            cls.otlp_refused = True
            self._reply(401, None)
            return
        if self.path == PROMPT_PATH:
            self._reply(200, PROMPT_VERSION)
            return
        self._reply(204 if self.command != "GET" else 200, None)

    def _reply(self, status: int, payload: dict[str, Any] | None) -> None:
        data = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the server quiet."""


server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpik)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_address[1]}"
os.environ.update(
    {
        "OL_LLM_ENVIRONMENT": "ci",
        "OPIK_URL_OVERRIDE": f"{base}/api",
        "OPIK_WORKSPACE": "default",
        "OPIK_KEYCLOAK_TOKEN_URL": f"{base}{TOKEN_PATH}",
        "OPIK_KEYCLOAK_CLIENT_ID": "ol-llm-smoke",
        "OPIK_KEYCLOAK_CLIENT_SECRET": "smoke",
        # Nothing listens on the discard port, so if the analytics default regresses
        # the events opik reports during the run fail locally instead of reaching
        # Comet from CI.
        "OPIK_ANALYTICS_URL": "http://127.0.0.1:9/",
    }
)

# ol_llm before opik, so OPIK_SENTRY_ENABLE and OPIK_ANALYTICS_ENABLE are set before
# opik reads them.
import ol_llm  # noqa: E402
import opik  # noqa: E402
import sentry_sdk  # noqa: E402
from ol_llm.models import override_models  # noqa: E402
from ol_llm.testing import override_embedding_models  # noqa: E402
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # noqa: E402
    ExportTraceServiceRequest,
)
from pydantic_ai.embeddings import TestEmbeddingModel  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

SMOKE = ol_llm.UseCase(
    app=APP, name="smoke", profile="request", model="openai:gpt-4o-mini"
)
SMOKE_COMPLETE = ol_llm.UseCase(
    app=APP, name="smoke-complete", profile="request", model="openai:gpt-4o-mini"
)
SMOKE_STREAM = ol_llm.UseCase(
    app=APP, name="smoke-stream", profile="request", model="openai:gpt-4o-mini"
)
SMOKE_EMBED = ol_llm.UseCase(
    app=APP,
    name="smoke-embed",
    profile="request",
    model="openai:text-embedding-3-small",
)


async def stream(agent: Any) -> str:  # noqa: ANN401
    """Read a streamed run the way a request handler would."""
    async with ol_llm.run_stream(agent, "ping") as llm_stream:
        text = "".join([chunk async for chunk in llm_stream.stream_text()])
        result = await llm_stream.result()
    return text if text == result.output else f"{text!r} != {result.output!r}"


def ended_traces(name: str) -> list[dict[str, Any]]:
    """Return the traces with this name the SDK sent once they ended."""
    # The SDK sends a trace when it starts and again, complete, when it ends.
    return [t for t in sdk_traces() if t.get("name") == name and "end_time" in t]


def sdk_traces() -> list[dict[str, Any]]:
    """Return the traces the SDK sent, from every batch request."""
    traces = []
    for request in FakeOpik.received:
        if not request.path.endswith("/traces/batch"):
            continue
        body = request.body
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        traces.extend(json.loads(body)["traces"])
    return traces


def check() -> list[str]:  # noqa: C901
    """Run a traced call and return what is wrong with what Opik received."""
    ol_llm.configure_tracing()
    # The test model stands in for the provider; the policy wrapper, the agent's
    # capabilities, and the import graph are the library's own.
    prompt = ol_llm.get_prompt(
        "smoke",
        project_name=APP,
        default=ol_llm.TextTemplate(template="Reply {{word}}."),
        environment="production",
    )
    with override_models(TestModel(custom_output_text="pong")):
        agent = ol_llm.build_agent(SMOKE, instructions=prompt.format(word="pong"))
        stream_agent = ol_llm.build_agent(SMOKE_STREAM)
        completed = ol_llm.complete_sync(SMOKE_COMPLETE, "ping")
    streamed = asyncio.run(stream(stream_agent))
    result = ol_llm.run_sync(
        agent,
        "ping",
        attribution=ol_llm.Attribution(thread_id="smoke-thread"),
        prompt=prompt,
    )
    with override_embedding_models(TestEmbeddingModel(dimensions=8)):
        embedded = ol_llm.embed_sync(SMOKE_EMBED, ["ping"], input_type="query")

    @opik.track(name="caller", project_name=CALLER)
    def caller() -> str:
        return ol_llm.run_sync(agent, "ping").output

    caller()
    if not ol_llm.tracing.flush():
        return ["the span exporter did not finish"]

    failures = []
    outputs = (result.output, completed.output, streamed)
    if outputs != ("pong", "pong", "pong"):
        failures.append(f"outputs were {outputs}")
    if embedded.dimensions != 8:  # noqa: PLR2004
        failures.append(f"embedding dimensions were {embedded.dimensions}")
    names = ("smoke", "smoke-complete", "smoke-stream", "smoke-embed", "caller")
    roots = {name: ended_traces(name) for name in names}
    if wrong := {name: ts for name, ts in roots.items() if len(ts) != 1}:
        return [*failures, f"expected one ended trace for each name, got {wrong}"]
    [trace], [complete_trace], [stream_trace], [embed_trace], callers = roots.values()
    expected = {
        "project_name": APP,
        "environment": "ci",
        "thread_id": "smoke-thread",
    }
    got = {k: trace.get(k) for k in expected}
    if got != expected:
        failures.append(f"root trace fields were {got}")
    if "use_case:smoke" not in trace.get("tags", []):
        failures.append(f"root trace tags were {trace.get('tags')}")
    if (prompt.source, result.prompt_version) != ("opik", "v2"):
        failures.append(f"the prompt was {prompt}, the result named {result}")
    linked = (trace.get("metadata") or {}).get("opik_prompts", [])
    if [p.get("version", {}).get("version_number") for p in linked] != ["v2"]:
        failures.append(f"root trace linked prompts {linked}")

    otlp = [r for r in FakeOpik.received if r.path == OTLP_PATH]
    if [r.headers.get("authorization") for r in otlp[:2]] != [
        "Bearer token-1",
        "Bearer token-2",
    ]:
        failures.append("the span exporter did not retry its 401 with a new token")
    # The first request was refused and sent again, so it is left out.
    trace_ids: dict[str, set[str]] = {}
    for request in otlp[1:]:
        project = request.headers.get("projectname", "")
        ids = {span.get("opik.trace_id", "") for span in _spans(request)}
        trace_ids.setdefault(project, set()).update(ids)
    expected_ids = {
        APP: {
            trace["id"],
            complete_trace["id"],
            stream_trace["id"],
            embed_trace["id"],
        },
        CALLER: {callers[0]["id"]},
    }
    if trace_ids != expected_ids:
        failures.append(
            f"model spans by project and trace were {trace_ids}, not {expected_ids}"
        )
    unauthenticated = [
        r.path
        for r in FakeOpik.received
        if not r.headers.get("authorization", "").startswith("Bearer token-")
    ]
    if unauthenticated:
        failures.append(f"requests without a token: {unauthenticated}")
    return failures


def _spans(request: Request) -> list[dict[str, str]]:
    export = ExportTraceServiceRequest.FromString(request.body)
    return [
        {a.key: a.value.string_value for a in span.attributes}
        for resource in export.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


def main() -> int:
    """Run the checks and return a process exit status."""
    failures = []
    # Before the run, which is what would report events.
    if opik.analytics.reporting_allowed():
        failures.append("opik usage analytics would report after importing ol_llm")
    failures += check()
    if imported := [name for name in DROPPED if name in sys.modules]:
        failures.append(f"dropped packages were imported: {imported}")
    if sentry_sdk.get_client().dsn is not None:
        failures.append("a Sentry client is running after importing opik")

    print(f"ol-llm {ol_llm.__version__}: {len(FakeOpik.received)} requests to Opik")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
