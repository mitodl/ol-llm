"""Decision models through ol-llm's model path: PydanticAI's decision models,
built by ``build_model`` under the use case's policy."""

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal

import pytest
from conftest import record_sleeps
from ol_llm import LLMRejected, LLMUnavailable, UseCase, complete_sync, models
from ol_llm.catalog import load_catalog, model_kind
from ol_llm.models import build_model, classify_failure
from ol_llm.settings import Settings, configure
from pydantic import BaseModel, Field
from pydantic_ai.models.decision import UnsureRoute
from pydantic_ai.models.google_decisions import GoogleDecisionsModel
from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel
from pydantic_ai.models.system_one import SystemOneModel


class Reading(BaseModel):
    """How the learner replied to an answer."""

    reply: Literal["finished", "follow_up", "wants_a_person"] = Field(
        description="What does the learner's reply do?"
    )


ANSWER = {
    "type": "choice",
    "choice": "finished",
    "confidence": 0.9,
    "probabilities": {"finished": 0.93, "follow_up": 0.05, "wants_a_person": 0.02},
}


@dataclass
class Server:
    """A System One server that records each request and answers from a script."""

    url: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list)
    statuses: list[int] = field(default_factory=list)


@pytest.fixture
def server() -> Iterator[Server]:
    state = Server()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state.requests.append(
                {"path": self.path, "headers": dict(self.headers), "body": body}
            )
            status = state.statuses.pop(0) if state.statuses else 200
            reply: dict[str, Any] = {"detail": "unavailable"}
            if status == 200:
                reply = {
                    "model": "microsoft-decision-1.1",
                    "answers": dict.fromkeys(body["questions"], ANSWER),
                    "usage": {"input_tokens": 40, "output_tokens": 0},
                }
            payload = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield state
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def decision_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    return record_sleeps(monkeypatch, models)


def use_case(model: str, **kwargs: Any) -> UseCase:  # noqa: ANN401
    return UseCase(
        app="learn-ai", name="support-reply", profile="request", model=model, **kwargs
    )


def test_a_system_one_model_answers_the_output_type(server: Server) -> None:
    configure(
        Settings(
            environment="ci", systemone_base_url=server.url, systemone_api_key="key"
        )
    )

    result = complete_sync(
        use_case("system-one:laya"),
        "thanks, that worked perfectly!",
        output_type=Reading,
    )

    assert result.output == Reading(reply="finished")
    assert result.provider_details is not None
    assert result.provider_details["confidence"] == {"reply": 0.9}
    assert result.provider_details["probabilities"]["reply"]["finished"] == 0.93
    [request] = server.requests
    assert request["path"] == "/v1/systemone"
    assert request["headers"]["Authorization"] == "Bearer key"
    assert request["body"]["model"] == "laya"
    assert request["body"]["state"] == "thanks, that worked perfectly!"
    assert request["body"]["questions"]["reply"]["type"] == "choice"


def test_an_azure_decision_model_gets_a_new_entra_token_for_each_attempt(
    server: Server, decision_sleeps: list[float]
) -> None:
    tokens = iter(["first", "second"])
    configure(
        Settings(
            environment="ci",
            azure_decision_endpoint=f"{server.url}/",
            azure_ad_token_provider=lambda: next(tokens),
        )
    )
    server.statuses = [503]

    result = complete_sync(
        use_case("azure-decision:Microsoft-Decision-1.1"), "x", output_type=Reading
    )

    assert result.output.reply == "finished"
    assert [r["path"] for r in server.requests] == [
        "/providers/microsoft/v1/systemone"
    ] * 2
    assert [r["headers"]["Authorization"] for r in server.requests] == [
        "Bearer first",
        "Bearer second",
    ]
    assert server.requests[0]["body"]["model"] == "Microsoft-Decision-1.1"
    assert len(decision_sleeps) == 1


def test_a_stalled_entra_token_fetch_fails_at_the_policy_timeout(
    server: Server, decision_sleeps: list[float]
) -> None:
    released = threading.Event()

    def stalls() -> str:
        released.wait(5)
        return "late"

    configure(
        Settings(
            environment="ci",
            azure_decision_endpoint=server.url,
            azure_ad_token_provider=stalls,
        )
    )
    started = time.monotonic()

    try:
        with pytest.raises(LLMUnavailable):
            complete_sync(
                use_case(
                    "azure-decision:Microsoft-Decision-1.1",
                    timeout_seconds=0.2,
                    max_retries=1,
                    policy_reason="a short timeout to keep the test fast",
                ),
                "x",
                output_type=Reading,
            )
    finally:
        released.set()

    # Two attempts at 0.2s each, not the provider's own 5s.
    assert time.monotonic() - started < 2
    assert server.requests == []
    assert len(decision_sleeps) == 1


def test_a_rejected_decision_request_is_not_retried(server: Server) -> None:
    configure(Settings(environment="ci", systemone_base_url=server.url))
    server.statuses = [422]

    with pytest.raises(LLMRejected) as raised:
        complete_sync(use_case("system-one:laya"), "x", output_type=Reading)

    assert raised.value.status_code == 422
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("model", "settings", "message"),
    [
        ("system-one:laya", Settings(environment="ci"), "systemone_base_url"),
        (
            "azure-decision:Microsoft-Decision-1.1",
            Settings(environment="ci"),
            "azure_decision_endpoint",
        ),
        (
            "azure-decision:Microsoft-Decision-1.1",
            Settings(environment="ci", azure_decision_endpoint="https://x"),
            "azure_ad_token_provider",
        ),
    ],
)
def test_an_unconfigured_decision_server_fails_at_build(
    model: str, settings: Settings, message: str
) -> None:
    configure(settings)

    with pytest.raises(ValueError, match=message):
        build_model(use_case(model))


@pytest.mark.parametrize(
    ("model", "settings", "kind", "base_url"),
    [
        (
            "typesafe:jev-latest",
            Settings(environment="ci", typesafe_api_key="ts"),
            SystemOneModel,
            "https://api.typesafe.ai",
        ),
        (
            "openai-decisions:gpt-6-luna",
            Settings(environment="ci"),
            OpenAIDecisionsModel,
            None,
        ),
        (
            "google-decisions:gemini-decision-1",
            Settings(environment="ci"),
            GoogleDecisionsModel,
            None,
        ),
    ],
)
@pytest.mark.usefixtures("provider_env")
def test_each_decision_provider_builds_its_model(
    model: str, settings: Settings, kind: type, base_url: str | None
) -> None:
    configure(settings)

    built = build_model(use_case(model))

    assert isinstance(built.wrapped, kind)  # type: ignore[attr-defined]
    if base_url is not None:
        assert built.wrapped.base_url == base_url  # type: ignore[attr-defined]


def test_a_decision_model_can_fall_back_to_a_chat_model(
    provider_env: None,  # noqa: ARG001
) -> None:
    configure(Settings(environment="ci", systemone_base_url="http://localhost:1"))

    built = build_model(
        use_case("system-one:laya", fallback_models=("openai:gpt-4o-mini",))
    )

    assert [m.catalog_id for m in built.models] == [  # type: ignore[attr-defined]
        "system-one:laya",
        "openai:gpt-4o-mini",
    ]


def test_the_catalog_reads_a_decision_provider_as_a_decision_model() -> None:
    assert model_kind("azure-decision:Microsoft-Decision-1.1") == "decision"
    assert model_kind("openai:gpt-4o-mini") == "chat"
    entry = load_catalog().get("azure-decision:Microsoft-Decision-1.1")
    assert entry is not None
    assert entry.kind == "decision"


def test_a_decision_hand_off_is_not_retried_as_an_outage() -> None:
    unsure = UnsureRoute("laya", "reply", {"reply": 0.4, "other": 0.6}, 0.8)

    assert classify_failure(unsure, use_case="support-reply", model="laya") is None
