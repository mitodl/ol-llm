"""Import-and-trace check for ol-llm installed with the opik SDK slimmed.

Runs as a plain script, not under pytest: pytest is one of the packages the
overrides drop. Exits non-zero on any failure.

What it proves, in an environment shaped like a consumer's runtime image:
- ol_llm and opik import with litellm, openai, pytest, and tree-sitter absent;
- an @opik.track root with an instrumented agent from ol_llm.build_agent inside it
  produces OTel spans that OpikSpanProcessor attaches to the tracked trace;
- importing ol_llm first leaves no Sentry client running.

No Opik server is needed. The SDK is pointed at a closed port, so its background
sender fails quietly; the assertions read the spans from an in-memory exporter.
"""

import os
import sys

os.environ["OPIK_URL_OVERRIDE"] = "http://127.0.0.1:9/api"
os.environ["OPIK_WORKSPACE"] = "default"

# ol_llm before opik, so OPIK_SENTRY_ENABLE is set before opik reads it.
import ol_llm
import opik
import sentry_sdk
from ol_llm.models import build_agent, override_models
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opik.integrations.otel import OpikSpanProcessor
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

DROPPED = ("litellm", "openai", "pytest", "tree_sitter", "click")
OPIK_TRACE_ID = "opik.trace_id"
SMOKE = ol_llm.UseCase(
    app="ol-llm-smoke", name="smoke", profile="request", model="openai:smoke"
)


def main() -> int:
    """Run the checks and return a process exit status."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(OpikSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    ol_llm.configure(ol_llm.Settings(environment="ci"))
    # The test model stands in for the provider; the policy wrapper, the agent's
    # capabilities, and the import graph are the library's own.
    with override_models(TestModel()):
        agent = build_agent(
            SMOKE,
            capabilities=[
                Instrumentation(
                    settings=InstrumentationSettings(tracer_provider=provider)
                )
            ],
        )

    @opik.track(name="smoke", project_name="ol-llm-smoke")
    def run() -> str:
        return agent.run_sync("ping").output

    output = run()
    spans = exporter.get_finished_spans()
    attached = [s for s in spans if (s.attributes or {}).get(OPIK_TRACE_ID)]
    imported = [name for name in DROPPED if name in sys.modules]
    failures = []
    if not spans:
        failures.append("the agent emitted no OTel spans")
    if len(attached) != len(spans):
        failures.append(
            f"{len(spans) - len(attached)} of {len(spans)} spans lack {OPIK_TRACE_ID}"
        )
    if imported:
        failures.append(f"dropped packages were imported: {imported}")
    if sentry_sdk.get_client().dsn is not None:
        failures.append("a Sentry client is running after importing opik")

    print(f"ol-llm {ol_llm.__version__}: output={output!r} spans={len(spans)}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
