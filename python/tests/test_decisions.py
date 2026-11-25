import asyncio
import json
from collections.abc import Callable, Generator
from datetime import date
from decimal import Decimal
from typing import Any

import httpx2
import ol_llm.catalog as catalog_module
import opik
import pytest
from conftest import record_sleeps
from ol_llm import (
    Attribution,
    Choice,
    LLMInvalidOutput,
    LLMRateLimited,
    LLMRejected,
    LLMUnavailable,
    Score,
    UseCase,
    YesNo,
    decide,
    decide_sync,
    decisions,
    tracing,
)
from ol_llm.catalog import Catalog, CatalogEntry, check_decision_model, load_catalog
from ol_llm.decisions import override_decision_transport
from ol_llm.settings import Settings, configure
from ol_llm.trace_destinations.opik_destination import PROJECT_NAME
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

MODEL = "azure:Microsoft-Decision-1.1"
ENDPOINT = "https://ol-decision-learn-ai-ci.cognitiveservices.azure.com/"
REPLY = UseCase(app="learn-ai", name="support-reply", profile="request", model=MODEL)
ATTRIBUTION = Attribution(user_id="user-1", thread_id="thread-1", run_id="run-1")

QUESTIONS = {
    "intent": Choice(
        instructions="What is the learner asking us to do?",
        options={"send": "File the request", "cancel": None, "edit": None},
    ),
    "wants_a_person": YesNo(instructions="Is the learner asking for a person?"),
    "urgency": Score(
        instructions="How urgent is this?", levels=["can wait", "soon", "now"]
    ),
}
ANSWERS = {
    "intent": {
        "type": "choice",
        "choice": "send",
        "confidence": 0.99,
        "probabilities": {"send": 0.8, "cancel": 0.15, "edit": 0.05},
        # Laya's own fields, which are ignored.
        "prediction_set": ["send"],
        "abstain": False,
    },
    "wants_a_person": {"type": "noul", "noul": 0.1},
    "urgency": {
        "type": "score",
        "score": 1.2,
        "confidence": 0.4,
        "legend": {"0": "can wait", "1": "soon", "2": "now"},
        "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
    },
}


def reply_body(answers: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "model": "microsoft-decision-1.1",
        "answers": ANSWERS if answers is None else answers,
        "usage": {"input_tokens": 120, "output_tokens": 3},
    }


Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture(autouse=True)
def azure_settings() -> Settings:
    settings = Settings(
        environment="ci",
        azure_decision_endpoint=ENDPOINT,
        azure_ad_token_provider=lambda: "entra-token",
    )
    configure(settings)
    return settings


@pytest.fixture(autouse=True)
def fresh_catalog_checks() -> Generator[None]:
    """Check each test's use cases against the catalog it patches in."""
    check_decision_model.cache_clear()
    yield
    check_decision_model.cache_clear()


@pytest.fixture
def decision_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    return record_sleeps(monkeypatch, decisions)


@pytest.fixture
def requests() -> list[httpx2.Request]:
    return []


def serving(
    requests: list[httpx2.Request], *responses: httpx2.Response
) -> override_decision_transport:
    """Answer each request with the next response, the last one from then on."""
    queue = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return override_decision_transport(httpx2.MockTransport(handler))


def ok(answers: dict[str, Any] | None = None) -> httpx2.Response:
    return httpx2.Response(200, json=reply_body(answers))


def test_the_request_goes_to_the_foundry_account_with_an_entra_token(
    requests: list[httpx2.Request],
) -> None:
    with serving(requests, ok()):
        decide_sync(REPLY, {"reply": "yeah go ahead"}, QUESTIONS)

    [request] = requests
    assert str(request.url) == (
        "https://ol-decision-learn-ai-ci.cognitiveservices.azure.com"
        "/providers/microsoft/v1/systemone"
    )
    assert request.headers["authorization"] == "Bearer entra-token"
    assert json.loads(request.content) == {
        "model": "Microsoft-Decision-1.1",
        "state": {"reply": "yeah go ahead"},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What is the learner asking us to do?",
                "criteria": {"send": "File the request", "cancel": None, "edit": None},
            },
            "wants_a_person": {
                "type": "noul",
                "instructions": "Is the learner asking for a person?",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["can wait", "soon", "now"],
            },
        },
    }


def test_each_answer_comes_back_typed(requests: list[httpx2.Request]) -> None:
    with serving(requests, ok()):
        decision = decide_sync(REPLY, "yeah go ahead", QUESTIONS)

    intent = decision.choice("intent")
    assert intent.choice == "send"
    assert intent.probabilities == {"send": 0.8, "cancel": 0.15, "edit": 0.05}
    # (0.8 - 1/3) / (1 - 1/3), not the server's own 0.99.
    assert intent.confidence == pytest.approx(0.7)
    assert decision.yes_no("wants_a_person").probability == 0.1
    urgency = decision.score("urgency")
    assert urgency.score == 1.2
    assert urgency.probabilities == (0.2, 0.4, 0.4)
    assert decision.model == MODEL
    assert decision.served_by == "microsoft-decision-1.1"
    assert (decision.input_tokens, decision.output_tokens) == (120, 3)
    with pytest.raises(TypeError, match="ChoiceAnswer"):
        decision.yes_no("intent")


def test_a_choice_without_probabilities_has_no_confidence(
    requests: list[httpx2.Request],
) -> None:
    answers = {"intent": {"type": "choice", "choice": "edit"}}
    with serving(requests, ok(answers)):
        decision = decide_sync(REPLY, "x", {"intent": QUESTIONS["intent"]})

    assert decision.choice("intent").confidence is None
    assert decision.choice("intent").probabilities == {}


@pytest.mark.parametrize(
    ("model", "settings", "url", "authorization"),
    [
        (
            "typesafe:jev-1.14.0",
            Settings(environment="ci", typesafe_api_key="ts-key"),
            "https://api.typesafe.ai/v1/systemone",
            "Bearer ts-key",
        ),
        (
            "systemone:laya",
            Settings(
                environment="ci",
                systemone_base_url="http://laya.internal:8000/",
                systemone_api_key="laya-key",
            ),
            "http://laya.internal:8000/v1/systemone",
            "Bearer laya-key",
        ),
        (
            "systemone:strands-decider-latest",
            Settings(environment="ci", systemone_base_url="http://decider:8000"),
            "http://decider:8000/v1/systemone",
            None,
        ),
    ],
)
def test_each_provider_is_reached_at_its_own_url(
    requests: list[httpx2.Request],
    model: str,
    settings: Settings,
    url: str,
    authorization: str | None,
) -> None:
    configure(settings)
    use_case = UseCase(app="learn-ai", name="route", profile="request", model=model)
    with serving(requests, ok()):
        decide_sync(use_case, "x", QUESTIONS)

    [request] = requests
    assert str(request.url) == url
    assert request.headers.get("authorization") == authorization
    assert json.loads(request.content)["model"] == model.partition(":")[2]


def test_typesafe_reads_its_key_from_the_environment(
    requests: list[httpx2.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(Settings(environment="ci"))
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    use_case = UseCase(
        app="learn-ai", name="route", profile="request", model="typesafe:jev-latest"
    )
    with serving(requests, ok()):
        decide_sync(use_case, "x", QUESTIONS)

    assert requests[0].headers["authorization"] == "Bearer env-key"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (Settings(environment="ci"), "azure_decision_endpoint"),
        (
            Settings(environment="ci", azure_decision_endpoint=ENDPOINT),
            "azure_ad_token_provider",
        ),
    ],
)
def test_an_unconfigured_azure_endpoint_raises_before_any_request(
    requests: list[httpx2.Request], settings: Settings, message: str
) -> None:
    configure(settings)
    with serving(requests, ok()), pytest.raises(ValueError, match=message):
        decide_sync(REPLY, "x", QUESTIONS)

    assert requests == []


def test_a_provider_without_a_decision_api_raises() -> None:
    use_case = UseCase(
        app="learn-ai", name="route", profile="request", model="openai:gpt-4o"
    )
    with pytest.raises(ValueError, match="listed with kind 'chat'"):
        decide_sync(use_case, "x", QUESTIONS)

    unlisted = UseCase(
        app="learn-ai", name="route", profile="request", model="vertex:decider"
    )
    with pytest.raises(ValueError, match="does not call 'vertex'"):
        decide_sync(unlisted, "x", QUESTIONS)


def test_a_decision_use_case_takes_no_fallbacks() -> None:
    use_case = UseCase(
        app="learn-ai",
        name="route",
        profile="request",
        model=MODEL,
        fallback_models=("typesafe:jev-latest",),
    )
    with pytest.raises(ValueError, match="takes none"):
        decide_sync(use_case, "x", QUESTIONS)


def test_a_retired_decision_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    retired = CatalogEntry(
        id=MODEL,
        kind="decision",
        status="retired",
        retires_on=date(2026, 1, 1),
        input_usd_per_mtok=Decimal("0.042"),
        output_usd_per_mtok=Decimal(0),
    )
    catalog = Catalog({**load_catalog().entries, MODEL: retired})
    monkeypatch.setattr(catalog_module, "load_catalog", lambda: catalog)

    with pytest.raises(ValueError, match="retired"):
        decide_sync(REPLY, "x", QUESTIONS)


def test_an_outage_is_retried_under_the_policy(
    requests: list[httpx2.Request], decision_sleeps: list[float]
) -> None:
    with serving(requests, httpx2.Response(503), httpx2.Response(503), ok()):
        decision = decide_sync(REPLY, "x", QUESTIONS)

    assert decision.choice("intent").choice == "send"
    assert len(requests) == 3
    assert len(decision_sleeps) == 2


def test_spent_retries_raise_the_library_error(
    requests: list[httpx2.Request], decision_sleeps: list[float]
) -> None:
    with serving(requests, httpx2.Response(503)), pytest.raises(LLMUnavailable):
        decide_sync(REPLY, "x", QUESTIONS)

    # The request profile's default: two retries.
    assert len(requests) == 3
    assert len(decision_sleeps) == 2


def test_a_rate_limit_waits_as_long_as_it_was_asked(
    requests: list[httpx2.Request], decision_sleeps: list[float]
) -> None:
    limited = httpx2.Response(429, headers={"retry-after": "3"})
    with serving(requests, limited, ok()):
        decide_sync(REPLY, "x", QUESTIONS)

    assert decision_sleeps == [3.0]


def test_a_rate_limit_past_the_budget_raises(
    requests: list[httpx2.Request], decision_sleeps: list[float]
) -> None:
    limited = httpx2.Response(429, headers={"retry-after": "60"})
    with serving(requests, limited), pytest.raises(LLMRateLimited) as raised:
        decide_sync(REPLY, "x", QUESTIONS)

    assert raised.value.retry_after == 60
    assert decision_sleeps == []


def test_a_rejection_is_not_retried(requests: list[httpx2.Request]) -> None:
    rejected = httpx2.Response(422, json={"detail": [{"msg": "bad question"}]})
    with serving(requests, rejected), pytest.raises(LLMRejected) as raised:
        decide_sync(REPLY, "x", QUESTIONS)

    assert raised.value.status_code == 422
    assert len(requests) == 1


def test_a_dropped_connection_is_retried(decision_sleeps: list[float]) -> None:
    calls: list[int] = []

    def flaky(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) == 1:
            msg = "timed out"
            raise httpx2.ReadTimeout(msg, request=request)
        return ok()

    with override_decision_transport(httpx2.MockTransport(flaky)):
        decide_sync(REPLY, "x", QUESTIONS)

    assert len(calls) == 2
    assert len(decision_sleeps) == 1


@pytest.mark.parametrize(
    "answers",
    [
        pytest.param({}, id="missing"),
        pytest.param(
            {**ANSWERS, "intent": {"type": "choice", "choice": "maybe"}},
            id="an option not offered",
        ),
        pytest.param(
            {**ANSWERS, "wants_a_person": {"type": "choice", "choice": "send"}},
            id="the wrong type",
        ),
        pytest.param(
            {**ANSWERS, "wants_a_person": {"type": "noul", "noul": 1.5}},
            id="a probability over 1",
        ),
        pytest.param(
            {
                **ANSWERS,
                "intent": {
                    "type": "choice",
                    "choice": "send",
                    "probabilities": {"send": 0.9, "cancel": 0.6, "edit": 0.1},
                },
            },
            id="probabilities adding up past 1",
        ),
        pytest.param(
            {**ANSWERS, "urgency": {"type": "score", "score": 3.0}},
            id="a score past the last level",
        ),
    ],
)
def test_an_answer_that_does_not_fit_its_question_is_invalid_output(
    requests: list[httpx2.Request], answers: dict[str, Any]
) -> None:
    with serving(requests, ok(answers)), pytest.raises(LLMInvalidOutput):
        decide_sync(REPLY, "x", QUESTIONS)

    assert len(requests) == 1


def test_a_body_that_is_not_a_decision_is_invalid_output(
    requests: list[httpx2.Request],
) -> None:
    not_json = httpx2.Response(200, content=b"<html>gateway</html>")
    with serving(requests, not_json), pytest.raises(LLMInvalidOutput) as raised:
        decide_sync(REPLY, "x", QUESTIONS)

    assert raised.value.__cause__ is not None


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: YesNo(instructions=" "), "needs instructions"),
        (lambda: YesNo(instructions="q", true="yes"), "together"),
        (lambda: Choice(instructions="q", options={"only": None}), "2 to 255"),
        (lambda: Score(instructions="q", levels=["one"]), "2 to 10"),
        (lambda: Score(instructions="q", levels="abc"), "2 to 10"),
    ],
)
def test_a_malformed_question_fails_where_it_is_declared(
    build: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_no_questions_or_state_raises() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        decide_sync(REPLY, "x", {})
    with pytest.raises(ValueError, match="non-empty state"):
        decide_sync(REPLY, "  ", QUESTIONS)


def test_the_call_is_one_trace_with_the_answers(
    requests: list[httpx2.Request], exported: InMemorySpanExporter
) -> None:
    with opik.record_traces_locally() as storage, serving(requests, ok()):
        decide_sync(REPLY, "yeah go ahead", QUESTIONS, attribution=ATTRIBUTION)
        [trace] = storage.trace_trees
    tracing.flush()

    assert trace.name == "support-reply"
    assert trace.project_name == "learn-ai"
    assert trace.thread_id == "thread-1"
    assert trace.input is not None
    assert trace.input["state"] == "yeah go ahead"
    assert trace.output == {
        "model": MODEL,
        "served_by": "microsoft-decision-1.1",
        "answers": {
            "intent": {"choice": "send", "confidence": pytest.approx(0.7)},
            "wants_a_person": {"probability": 0.1},
            "urgency": {"score": 1.2},
        },
    }
    [span] = exported.get_finished_spans()
    attributes = span.attributes or {}
    assert span.name == "decision Microsoft-Decision-1.1"
    assert attributes["opik.trace_id"] == trace.id
    assert attributes[PROJECT_NAME] == "learn-ai"
    assert attributes["gen_ai.provider.name"] == "azure"
    assert attributes["gen_ai.request.model"] == "Microsoft-Decision-1.1"
    assert attributes["gen_ai.response.model"] == "microsoft-decision-1.1"
    assert attributes["gen_ai.usage.input_tokens"] == 120
    assert attributes["gen_ai.usage.output_tokens"] == 3


def test_a_failed_call_marks_its_span_failed(
    requests: list[httpx2.Request], exported: InMemorySpanExporter
) -> None:
    rejected = httpx2.Response(400)
    with serving(requests, rejected), pytest.raises(LLMRejected):
        decide_sync(REPLY, "x", QUESTIONS)
    tracing.flush()

    [span] = exported.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


def test_decide_runs_on_the_callers_loop(requests: list[httpx2.Request]) -> None:
    with serving(requests, ok()):
        decision = asyncio.run(decide(REPLY, "x", QUESTIONS))

    assert decision.yes_no("wants_a_person").probability == 0.1


def raising(exc: Exception) -> override_decision_transport:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise exc

    return override_decision_transport(httpx2.MockTransport(handler))


def test_a_permanent_transport_failure_is_a_rejection(
    decision_sleeps: list[float],
) -> None:
    msg = "proxy refused"
    with raising(httpx2.ProxyError(msg)), pytest.raises(LLMRejected) as raised:
        decide_sync(REPLY, "x", QUESTIONS)

    assert raised.value.status_code is None
    assert decision_sleeps == []


def test_a_body_that_cannot_be_decoded_is_invalid_output() -> None:
    msg = "corrupt gzip"
    with raising(httpx2.DecodingError(msg)), pytest.raises(LLMInvalidOutput):
        decide_sync(REPLY, "x", QUESTIONS)


def test_a_rounded_distribution_a_little_over_1_is_accepted(
    requests: list[httpx2.Request],
) -> None:
    rounded = {"send": 0.34, "cancel": 0.34, "edit": 0.33}
    answers = {
        **ANSWERS,
        "intent": {"type": "choice", "choice": "send", "probabilities": rounded},
    }
    with serving(requests, ok(answers)):
        decision = decide_sync(REPLY, "x", QUESTIONS)

    assert decision.choice("intent").probabilities == rounded


def test_a_score_keeps_each_levels_position_when_a_level_is_left_out(
    requests: list[httpx2.Request],
) -> None:
    answers = {
        **ANSWERS,
        "urgency": {
            "type": "score",
            "score": 1.6,
            "probabilities": {"1": 0.4, "2": 0.6},
        },
    }
    with serving(requests, ok(answers)):
        decision = decide_sync(REPLY, "x", QUESTIONS)

    assert decision.score("urgency").probabilities == (0.0, 0.4, 0.6)


def test_the_token_is_fetched_for_each_attempt(
    requests: list[httpx2.Request], decision_sleeps: list[float]
) -> None:
    tokens = iter(["first", "second"])
    configure(
        Settings(
            environment="ci",
            azure_decision_endpoint=ENDPOINT,
            azure_ad_token_provider=lambda: next(tokens),
        )
    )
    with serving(requests, httpx2.Response(503), ok()):
        decide_sync(REPLY, "x", QUESTIONS)

    assert [r.headers["authorization"] for r in requests] == [
        "Bearer first",
        "Bearer second",
    ]
    assert len(decision_sleeps) == 1


def test_a_token_provider_failure_is_not_a_model_failure(
    requests: list[httpx2.Request],
) -> None:
    def no_token() -> str:
        msg = "managed identity unavailable"
        raise RuntimeError(msg)

    configure(
        Settings(
            environment="ci",
            azure_decision_endpoint=ENDPOINT,
            azure_ad_token_provider=no_token,
        )
    )
    with serving(requests, ok()), pytest.raises(RuntimeError, match="managed"):
        decide_sync(REPLY, "x", QUESTIONS)

    assert requests == []
