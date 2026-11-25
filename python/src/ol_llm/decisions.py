"""The decision entry points: one call to a system-one decision model, traced.

A decision model reads a state (text or JSON) and answers named, typed questions
in one forward pass, with probabilities and no generated text: a yes/no question
gets the probability of yes, a choice gets the option picked and a distribution
over the options, and a score gets the expected level over an ordered scale. They
answer in tens of milliseconds for a fraction of a chat model's price, so they
suit the routing and classification steps an application would otherwise do with
a phrase list or a chat model asked for JSON.

Every model this module calls speaks TypeSafe's System One API
(``POST .../v1/systemone``), which is where the shared shape comes from. The
provider half of a use case's model id picks where the request goes:

- ``azure:<deployment>``: Microsoft-Decision on an Azure Foundry (AIServices)
  account, at ``Settings.azure_decision_endpoint`` plus
  ``/providers/microsoft/v1/systemone``, authenticated with
  ``Settings.azure_ad_token_provider``. The model half is the deployment name.
- ``typesafe:<model>``: TypeSafe's Jev at ``https://api.typesafe.ai``, with
  ``Settings.typesafe_api_key`` or ``TYPESAFE_API_KEY``.
- ``systemone:<model>``: a self-hosted server that copies the API, e.g. Laya's
  ``laya-serve`` or ``strands-decider serve``, at ``Settings.systemone_base_url``
  with ``Settings.systemone_api_key`` as a bearer token when it is set.

Only text and JSON states and the three question types every server shares are
sent: images and TypeSafe's ``box`` question are not. Fields a server adds to its
answers (Laya's ``prediction_set``, Strands' ``latency_ms``) are ignored, and so is
each server's own ``confidence``: they compute it with different formulas, so
``ChoiceAnswer.confidence`` is computed here from the probabilities, the same way
for every model.

``decide`` and ``decide_sync`` run under the use case's call policy (spec §6.2):
the timeout on each attempt, retries with backoff on a 429, a 5xx or a dropped
connection, and the library's error classes (spec §6.7). Each call is one traced
call with the §4 attribution fields and the model's token usage. Needs no extra.
"""

import asyncio
import json
import os
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import httpx2
from opentelemetry.trace import Span
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ol_llm import trace_attributes as attrs
from ol_llm.attribution import NO_ATTRIBUTION, Attribution
from ol_llm.catalog import check_decision_model
from ol_llm.errors import LLMInvalidOutput
from ol_llm.loops import LoopLocal, run_coroutine
from ol_llm.policy import CallPolicy, UseCase, resolve_policy
from ol_llm.retries import RetryBudget, classify_http_failure
from ol_llm.settings import Settings, get_settings
from ol_llm.traces import TracedCall, record_error
from ol_llm.tracing import enabled, include_content, tracer_provider

DECISION_PROVIDERS = frozenset({"azure", "typesafe", "systemone"})
"""The providers ``decide`` sends requests to; see the module docstring."""

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
AZURE_DECISION_PATH = "/providers/microsoft/v1/systemone"
SYSTEMONE_PATH = "/v1/systemone"

# The bounds every server accepts. TypeSafe also takes a one-option choice and a
# one-level score; Strands Decider needs at least two.
MIN_CHOICE_OPTIONS = 2
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# Patched in tests so retries do not wait.
_sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep


@dataclass(frozen=True, kw_only=True)
class YesNo:
    """A yes/no question, ``noul`` on the wire. Answered with P(yes).

    :param instructions: the question, e.g. "Is the learner asking for a person?"
    :param true: what counts as yes, when the question alone leaves it open.
    :param false: what counts as no. Set with ``true`` or not at all.
    :raises ValueError: on empty instructions, or only one of ``true`` and
        ``false``.
    """

    instructions: str
    true: str | None = None
    false: str | None = None

    def __post_init__(self) -> None:
        """Validate the question so a bad one fails where it is declared."""
        _require_instructions(self.instructions)
        if (self.true is None) != (self.false is None):
            msg = "Set YesNo.true and YesNo.false together, or neither"
            raise ValueError(msg)

    def wire(self) -> dict[str, Any]:
        """Return the question as the System One API takes it."""
        question: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.true is not None:
            question["criteria"] = {"true": self.true, "false": self.false}
        return question


@dataclass(frozen=True, kw_only=True)
class Choice:
    """Pick one of a fixed set of options.

    :param instructions: the question, e.g. "What is the learner asking us to do?"
    :param options: option name to what it means, or ``None`` when the name says
        it all. 2 to 255 options.
    :raises ValueError: on empty instructions, or too few or too many options.
    """

    instructions: str
    options: Mapping[str, str | None]

    def __post_init__(self) -> None:
        """Validate the question so a bad one fails where it is declared."""
        _require_instructions(self.instructions)
        if not MIN_CHOICE_OPTIONS <= len(self.options) <= MAX_CHOICE_OPTIONS:
            msg = (
                f"Choice takes {MIN_CHOICE_OPTIONS} to {MAX_CHOICE_OPTIONS} options, "
                f"not {len(self.options)}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "options", dict(self.options))

    def wire(self) -> dict[str, Any]:
        """Return the question as the System One API takes it."""
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.options),
        }


@dataclass(frozen=True, kw_only=True)
class Score:
    """Place the state on an ordered scale. Answered with the expected level.

    :param instructions: the question, e.g. "How urgent is this request?"
    :param levels: the scale from lowest to highest, 2 to 10 levels.
    :raises ValueError: on empty instructions, or too few or too many levels.
    """

    instructions: str
    levels: Sequence[str]

    def __post_init__(self) -> None:
        """Validate the question so a bad one fails where it is declared."""
        _require_instructions(self.instructions)
        if isinstance(self.levels, str) or not (
            MIN_SCORE_LEVELS <= len(self.levels) <= MAX_SCORE_LEVELS
        ):
            msg = (
                f"Score takes a sequence of {MIN_SCORE_LEVELS} to "
                f"{MAX_SCORE_LEVELS} levels"
            )
            raise ValueError(msg)
        object.__setattr__(self, "levels", tuple(self.levels))

    def wire(self) -> dict[str, Any]:
        """Return the question as the System One API takes it."""
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.levels),
        }


Question = YesNo | Choice | Score


def _require_instructions(instructions: str) -> None:
    if not instructions.strip():
        msg = "A decision question needs instructions"
        raise ValueError(msg)


@dataclass(frozen=True)
class YesNoAnswer:
    """The answer to a ``YesNo`` question.

    :param probability: the probability of yes, from 0 to 1.
    """

    probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    """The answer to a ``Choice`` question.

    :param choice: the option the model picked.
    :param probabilities: each option's probability. Empty when the server sent
        none, which Microsoft's documentation does not rule out.
    :param confidence: how far the top probability is above an even split, from 0
        (no better than a guess) to 1 (certain): ``(p_max - 1/n) / (1 - 1/n)``
        over the ``n`` options asked about. ``None`` without probabilities.
    """

    choice: str
    probabilities: Mapping[str, float]
    confidence: float | None


@dataclass(frozen=True)
class ScoreAnswer:
    """The answer to a ``Score`` question.

    :param score: the expected level, from 0 (the first level) to ``n - 1``.
    :param probabilities: each level's probability, lowest level first. Empty when
        the server sent none.
    """

    score: float
    probabilities: tuple[float, ...]


Answer = YesNoAnswer | ChoiceAnswer | ScoreAnswer
_A = TypeVar("_A", YesNoAnswer, ChoiceAnswer, ScoreAnswer)


@dataclass(frozen=True, kw_only=True)
class Decision:
    """What ``decide`` returns.

    :param answers: question name to its answer, one per question asked.
    :param model: the catalog id the use case named.
    :param served_by: the model the server says answered, e.g.
        ``microsoft-decision-1.1`` behind an Azure deployment name.
    :param input_tokens: the input tokens the server reported.
    :param output_tokens: the output tokens the server reported.
    """

    answers: Mapping[str, Answer]
    model: str
    served_by: str | None
    input_tokens: int
    output_tokens: int

    def yes_no(self, name: str) -> YesNoAnswer:
        """Return a ``YesNo`` question's answer.

        :raises KeyError: if no question had the name.
        :raises TypeError: if the question was of another type.
        """
        return _typed(self.answers[name], YesNoAnswer, name)

    def choice(self, name: str) -> ChoiceAnswer:
        """Return a ``Choice`` question's answer.

        :raises KeyError: if no question had the name.
        :raises TypeError: if the question was of another type.
        """
        return _typed(self.answers[name], ChoiceAnswer, name)

    def score(self, name: str) -> ScoreAnswer:
        """Return a ``Score`` question's answer.

        :raises KeyError: if no question had the name.
        :raises TypeError: if the question was of another type.
        """
        return _typed(self.answers[name], ScoreAnswer, name)


def _typed(answer: Answer, kind: type[_A], name: str) -> _A:
    if not isinstance(answer, kind):
        msg = f"Question {name!r} has a {type(answer).__name__}, not a {kind.__name__}"
        raise TypeError(msg)
    return answer


# The response, read leniently: servers add their own fields to it.


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _WireNoul(_Wire):
    type: Literal["noul"]
    noul: float


class _WireChoice(_Wire):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)


class _WireScore(_Wire):
    type: Literal["score"]
    score: float
    probabilities: dict[int, float] = Field(default_factory=dict)


class _WireUsage(_Wire):
    input_tokens: int = 0
    output_tokens: int = 0


class _WireResponse(_Wire):
    model: str | None = None
    answers: dict[str, _WireNoul | _WireChoice | _WireScore]
    usage: _WireUsage = Field(default_factory=_WireUsage)


@dataclass(frozen=True)
class _Endpoint:
    url: str
    model: str
    headers: Mapping[str, str]


def _endpoint(use_case: UseCase, settings: Settings) -> _Endpoint:
    """Return where a use case's request goes and how it authenticates.

    :raises ValueError: for a provider ``decide`` does not call, or one whose
        endpoint or credential is not configured.
    :raises KeyError: for a ``typesafe:`` model with no key in the settings or
        ``TYPESAFE_API_KEY``.
    """
    provider, _, model = use_case.model.partition(":")
    if provider == "azure":
        if not settings.azure_decision_endpoint:
            msg = "azure: decision models need Settings.azure_decision_endpoint"
            raise ValueError(msg)
        if settings.azure_ad_token_provider is None:
            msg = "azure: decision models need Settings.azure_ad_token_provider"
            raise ValueError(msg)
        return _Endpoint(
            url=settings.azure_decision_endpoint.rstrip("/") + AZURE_DECISION_PATH,
            model=model,
            headers={"Authorization": f"Bearer {settings.azure_ad_token_provider()}"},
        )
    if provider == "typesafe":
        key = settings.typesafe_api_key or os.environ["TYPESAFE_API_KEY"]
        return _Endpoint(
            url=TYPESAFE_BASE_URL + SYSTEMONE_PATH,
            model=model,
            headers={"Authorization": f"Bearer {key}"},
        )
    if provider == "systemone":
        if not settings.systemone_base_url:
            msg = "systemone: decision models need Settings.systemone_base_url"
            raise ValueError(msg)
        bearer = settings.systemone_api_key
        return _Endpoint(
            url=settings.systemone_base_url.rstrip("/") + SYSTEMONE_PATH,
            model=model,
            headers={"Authorization": f"Bearer {bearer}"} if bearer else {},
        )
    msg = (
        f"ol-llm does not call {provider!r} decision models; supported providers "
        f"are {sorted(DECISION_PROVIDERS)}"
    )
    raise ValueError(msg)


async def _close_client(client: httpx2.AsyncClient) -> None:
    await client.aclose()


# One client for every decision call on a loop: the URL, credential and timeout
# are set per request.
_clients: LoopLocal[httpx2.AsyncClient] = LoopLocal(httpx2.AsyncClient, _close_client)

# A transport of a test's own; see ``override_decision_transport``.
_transport_override: ContextVar[httpx2.AsyncBaseTransport | None] = ContextVar(
    "ol_llm_decision_transport", default=None
)


@asynccontextmanager
async def _client() -> AsyncIterator[httpx2.AsyncClient]:
    if (transport := _transport_override.get()) is None:
        yield _clients.get()
        return
    async with httpx2.AsyncClient(transport=transport) as client:
        yield client


@contextmanager
def override_decision_transport(
    transport: httpx2.AsyncBaseTransport,
) -> Generator[None]:
    """Send ``decide``'s requests to an ``httpx2`` transport, for tests.

    E.g. ``httpx2.MockTransport(handler)``, with a handler that answers as a
    System One server would. The policy, error classes and tracing still apply.

    :param transport: the transport.
    """
    token = _transport_override.set(transport)
    try:
        yield
    finally:
        _transport_override.reset(token)


def _confidence(probabilities: Mapping[str, float], options: int) -> float | None:
    if not probabilities:
        return None
    even = 1 / options
    return max(0.0, (max(probabilities.values()) - even) / (1 - even))


def _answers(
    response: _WireResponse, questions: Mapping[str, Question], use_case: UseCase
) -> dict[str, Answer]:
    """Check each answer against its question and convert it.

    :raises LLMInvalidOutput: on a missing answer, one of the wrong type, a choice
        that was not offered, or a probability or score out of range.
    """
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        wire = response.answers.get(name)
        answer = None if wire is None else _answer(question, wire)
        if answer is None:
            msg = f"{use_case.model} did not answer {name!r} as it was asked"
            raise LLMInvalidOutput(msg, use_case=use_case.name, model=use_case.model)
        answers[name] = answer
    return answers


def _answer(
    question: Question, wire: _WireNoul | _WireChoice | _WireScore
) -> Answer | None:
    """Return the answer, or ``None`` when it does not fit its question."""
    answer: Answer
    if isinstance(question, YesNo) and isinstance(wire, _WireNoul):
        probabilities: Sequence[float] = (wire.noul,)
        answer = YesNoAnswer(wire.noul)
    elif isinstance(question, Choice) and isinstance(wire, _WireChoice):
        offered = set(question.options)
        if wire.choice not in offered or not set(wire.probabilities) <= offered:
            return None
        probabilities = tuple(wire.probabilities.values())
        answer = ChoiceAnswer(
            wire.choice,
            wire.probabilities,
            _confidence(wire.probabilities, len(offered)),
        )
    elif isinstance(question, Score) and isinstance(wire, _WireScore):
        levels = range(len(question.levels))
        if not 0 <= wire.score <= levels[-1] or not set(wire.probabilities) <= set(
            levels
        ):
            return None
        probabilities = tuple(wire.probabilities.values())
        answer = ScoreAnswer(
            wire.score,
            tuple(wire.probabilities[i] for i in sorted(wire.probabilities)),
        )
    else:
        return None
    if any(not 0 <= p <= 1 for p in probabilities):
        return None
    return answer


def _sent_state(state: str | Mapping[str, Any]) -> str | dict[str, Any]:
    if isinstance(state, str):
        if not state.strip():
            msg = "decide needs a non-empty state"
            raise ValueError(msg)
        return state
    # Fails here, not at the server, on a state JSON cannot carry.
    json.dumps(state)
    return dict(state)


def _decision_input(state: object, questions: object) -> None:
    """Stand in for a traced function, so the trace input names ``decide``'s."""


async def _attempt(
    endpoint: _Endpoint, body: Mapping[str, Any], policy: CallPolicy
) -> bytes:
    async with _client() as client:
        response = await client.post(
            endpoint.url,
            json={**body, "model": endpoint.model},
            headers=dict(endpoint.headers),
            timeout=policy.timeout_seconds,
        )
    response.raise_for_status()
    return response.content


async def _request(
    use_case: UseCase, body: Mapping[str, Any], settings: Settings
) -> _WireResponse:
    """Send the request under the use case's policy and read the response.

    :raises LLMError: once a failure is not retryable or the retries are spent,
        or ``LLMInvalidOutput`` on a body that is not a System One response.
    """
    budget = RetryBudget(policy := resolve_policy(use_case), use_case.profile)
    while True:
        # Per attempt: a token provider hands out a new token near expiry.
        endpoint = _endpoint(use_case, settings)
        try:
            content = await _attempt(endpoint, body, policy)
        except Exception as exc:
            error = classify_http_failure(
                exc, use_case=use_case.name, model=use_case.model
            )
            if error is None:
                raise
            if (delay := budget.next_delay(error)) is None:
                raise error from exc
            await _sleep(delay)
            continue
        try:
            return _WireResponse.model_validate_json(content)
        except ValidationError as exc:
            msg = f"{use_case.model} sent a response that is not a decision: {exc}"
            raise LLMInvalidOutput(
                msg, use_case=use_case.name, model=use_case.model
            ) from exc


async def decide(
    use_case: UseCase,
    state: str | Mapping[str, Any],
    questions: Mapping[str, Question],
    *,
    attribution: Attribution = NO_ATTRIBUTION,
) -> Decision:
    """Ask a decision model questions about a state, under a traced root.

    Read the answers' probabilities against thresholds the application tunes for
    the model, and keep a deterministic path for when the call fails: a decision
    model is fast, but it is still a network call.

    :param use_case: the call site, naming a decision model
        (``azure:<deployment>``, ``typesafe:<model>`` or ``systemone:<model>``).
        It may declare no ``fallback_models``.
    :param state: what the questions are about: text, or a JSON object such as
        ``{"reply": ..., "we_asked": ...}``.
    :param questions: question name to question. The names key the answers.
    :param attribution: the per-call attribution fields.
    :returns: one answer per question.
    :rtype: Decision
    :raises LLMError: on a model failure, or ``LLMInvalidOutput`` on a response
        that does not answer the questions asked, with the original exception as
        its cause.
    :raises ValueError: if there are no questions or no state, the catalog check
        rejects the use case, or the provider or its endpoint is not configured.
    """
    if not questions:
        msg = "decide needs at least one question"
        raise ValueError(msg)
    sent_state = _sent_state(state)
    model_tags = check_decision_model(use_case)
    settings = get_settings()
    # Raises on a provider or endpoint that is not configured, before any trace.
    _endpoint(use_case, settings)
    wire_questions = {name: q.wire() for name, q in questions.items()}
    call = TracedCall(
        use_case=use_case,
        attribution=attribution,
        fields=TracedCall.fields_for(use_case, attribution, model_tags=model_tags),
        traced=enabled(),
    )
    with call.trace_scope(_decision_input, sent_state, wire_questions) as output:
        async with _decision_span(call) as span:
            response = await _request(
                use_case,
                {"state": sent_state, "questions": wire_questions},
                settings,
            )
            decision = Decision(
                answers=_answers(response, questions, use_case),
                model=use_case.model,
                served_by=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            _record(span, decision)
        output.value = _trace_output(decision)
    return decision


def decide_sync(
    use_case: UseCase,
    state: str | Mapping[str, Any],
    questions: Mapping[str, Question],
    *,
    attribution: Attribution = NO_ATTRIBUTION,
) -> Decision:
    """Run ``decide`` from synchronous code, on this thread's ol-llm event loop.

    See ``decide`` for the parameters, return value and errors.
    """
    return run_coroutine(decide(use_case, state, questions, attribution=attribution))


def _trace_output(decision: Decision) -> dict[str, Any]:
    return {
        "model": decision.model,
        "served_by": decision.served_by,
        "answers": {
            name: _answer_output(answer) for name, answer in decision.answers.items()
        },
    }


def _answer_output(answer: Answer) -> dict[str, Any]:
    if isinstance(answer, YesNoAnswer):
        return {"probability": answer.probability}
    if isinstance(answer, ChoiceAnswer):
        return {"choice": answer.choice, "confidence": answer.confidence}
    return {"score": answer.score}


@asynccontextmanager
async def _decision_span(call: TracedCall) -> AsyncIterator[Span | None]:
    """Open the model request's span, with the GenAI request attributes.

    Yields ``None`` when the call is not traced.
    """
    provider = tracer_provider(call.use_case.app) if call.traced else None
    if provider is None:
        yield None
        return
    system, _, model_name = call.use_case.model.partition(":")
    with provider.get_tracer("ol_llm").start_as_current_span(
        f"decision {model_name}",
        attributes={
            attrs.OPERATION_NAME: "decision",
            attrs.PROVIDER_NAME: system,
            attrs.REQUEST_MODEL: model_name,
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            record_error(span, exc, content=include_content())
            raise


def _record(span: Span | None, decision: Decision) -> None:
    if span is None:
        return
    attributes: dict[str, str | int] = {
        attrs.USAGE_INPUT_TOKENS: decision.input_tokens,
        attrs.USAGE_OUTPUT_TOKENS: decision.output_tokens,
    }
    if decision.served_by:
        attributes[attrs.RESPONSE_MODEL] = decision.served_by
    span.set_attributes(attributes)
