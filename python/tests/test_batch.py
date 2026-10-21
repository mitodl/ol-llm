import pickle
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from conftest import PromptLibrary
from ol_llm import (
    BatchFailed,
    BatchKeys,
    ChatTemplate,
    Chunk,
    CircuitBreaker,
    LLMBudgetExceeded,
    LLMContentFiltered,
    LLMError,
    LLMInvalidOutput,
    LLMRateLimited,
    LLMRejected,
    LLMUnavailable,
    ManagedPrompt,
    RowFailure,
    TextTemplate,
    UseCase,
    celery_rate_limit,
    complete_sync,
    dagster_pool,
    get_prompt,
    process_batch,
    run_sync,
)
from ol_llm.models import build_agent, override_models
from pydantic import BaseModel, ConfigDict, create_model
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

PRIMARY = "openai:gpt-4o-mini"
FALLBACK = "bedrock:global.anthropic.claude-haiku-4-5-20251001-v1:0"
SUMMARY = UseCase(
    app="ol-data-platform",
    name="feedback-summary",
    profile="batch",
    model=PRIMARY,
    fallback_models=(FALLBACK,),
)
TEMPLATE = TextTemplate(template="Summarize the feedback.", version="v3")
PROMPT = ManagedPrompt(name="feedback-summary", template=TEMPLATE, source="opik")
KEYS = BatchKeys(model=PRIMARY, prompt=PROMPT)


def rejected(status_code: int | None = 401) -> LLMRejected:
    return LLMRejected("bad key", use_case="s", model=PRIMARY, status_code=status_code)


def row_rejected() -> LLMRejected:
    return rejected(400)


def unavailable() -> LLMUnavailable:
    return LLMUnavailable("503", use_case="s", model=PRIMARY)


def invalid() -> LLMInvalidOutput:
    return LLMInvalidOutput("not a summary", use_case="s", model=PRIMARY)


class Recorder:
    """A persist callback that keeps every chunk it is handed."""

    def __init__(self) -> None:
        self.chunks: list[Chunk[str, str]] = []

    def __call__(self, chunk: Chunk[str, str]) -> None:
        self.chunks.append(chunk)


def failing_on(
    failures: dict[str, LLMError],
) -> tuple[list[str], Callable[[str], str]]:
    called: list[str] = []

    def call(row: str) -> str:
        called.append(row)
        if row in failures:
            raise failures[row]
        return row.upper()

    return called, call


def test_the_key_changes_with_the_input_prompt_and_model() -> None:
    key = KEYS.key("hello")
    edited = replace(PROMPT, template=replace(TEMPLATE, template="Summarize it."))

    assert KEYS.key("hello") == key
    assert KEYS.key("hello!") != key
    assert replace(KEYS, model=FALLBACK).key("hello") != key
    assert replace(KEYS, prompt=replace(PROMPT, name="other")).key("hello") != key
    assert replace(KEYS, prompt=edited).key("hello") != key


def test_the_embedded_default_makes_the_same_key_as_the_fetched_prompt() -> None:
    # The version is left out, so an Opik outage does not reprocess every row.
    embedded = ManagedPrompt(
        name="feedback-summary",
        template=replace(TEMPLATE, version=None),
        source="embedded",
        degraded_reasons=("embedded_prompt",),
    )

    assert replace(KEYS, prompt=embedded).key("hello") == KEYS.key("hello")


def test_a_chat_prompt_is_keyed_on_its_messages() -> None:
    def chat(content: str) -> ManagedPrompt[ChatTemplate]:
        messages = [{"role": "system", "content": content}]
        return ManagedPrompt(
            name="feedback-summary",
            template=ChatTemplate(messages=messages),
            source="opik",
        )

    first = BatchKeys(model=PRIMARY, prompt=chat("Summarize."))
    second = BatchKeys(model=PRIMARY, prompt=chat("Summarize briefly."))

    assert first.key("hello") != second.key("hello")


def test_provenance_keeps_the_prompt_degraded_reason() -> None:
    embedded = ManagedPrompt(
        name="feedback-summary",
        template=TEMPLATE,
        source="embedded",
        degraded_reasons=("embedded_prompt",),
    )
    keys = BatchKeys(model=PRIMARY, prompt=embedded)

    def reply(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("summary")])

    with override_models(FunctionModel(reply)):
        agent = build_agent(SUMMARY)
    # Not passed prompt=, so only the prompt knows it was the embedded default.
    result = run_sync(agent, "hello")

    assert result.degraded_reasons == ()
    assert keys.provenance("hello", result).degraded_reasons == ("embedded_prompt",)


def test_provenance_names_the_model_that_answered(
    prompt_library: PromptLibrary,
) -> None:
    def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(503, "gpt-4o-mini", body="down")

    def reply(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("backup")])

    prompt_library.add(
        "feedback-summary",
        TEMPLATE.template,
        project_name="ol-data-platform",
        environments=("production",),
    )
    prompt = get_prompt(
        "feedback-summary",
        project_name="ol-data-platform",
        default=TEMPLATE,
        environment="production",
    )
    keys = BatchKeys(model=SUMMARY.model, prompt=prompt)
    with override_models(
        {PRIMARY: FunctionModel(fail), FALLBACK: FunctionModel(reply)}
    ):
        agent = build_agent(SUMMARY, instructions=prompt.format())
    result = run_sync(agent, "hello")

    provenance = keys.provenance("hello", result)

    assert provenance.idempotency_key == keys.key("hello")
    assert provenance.model == FALLBACK
    assert provenance.degraded_reasons == ("fallback_model",)
    assert provenance.prompt_name == "feedback-summary"
    assert provenance.prompt_version == "v1"
    assert provenance.prompt_template_hash == TEMPLATE.template_hash
    assert provenance.processed_at.tzinfo is not None


@pytest.mark.parametrize(
    ("error", "retry"),
    [
        (LLMRateLimited("429", use_case="s", model=PRIMARY, retry_after=None), True),
        (unavailable(), True),
        (rejected(), True),
        (rejected(None), True),
        (rejected(404), True),
        (rejected(400), False),
        (rejected(413), False),
        (rejected(422), False),
        (invalid(), False),
        (LLMContentFiltered("filtered", use_case="s", model=PRIMARY), False),
        (LLMBudgetExceeded("limit", use_case="s", model=PRIMARY), False),
    ],
)
def test_failures_about_the_row_are_not_retried_next_run(
    error: LLMError,
    retry: bool,  # noqa: FBT001
) -> None:
    assert RowFailure("row", error).retry_next_run is retry


@pytest.mark.parametrize("status_code", [401, 403, 404, 501, None])
def test_a_rejection_of_the_account_or_model_opens_the_circuit_at_once(
    status_code: int | None,
) -> None:
    breaker = CircuitBreaker()

    breaker.record_failure(rejected(status_code))

    assert breaker.is_open


def test_row_rejections_open_the_circuit_only_at_their_threshold() -> None:
    breaker = CircuitBreaker(max_consecutive_rejections=3)

    breaker.record_failure(row_rejected())
    breaker.record_failure(row_rejected())
    assert not breaker.is_open
    breaker.record_failure(row_rejected())

    assert breaker.is_open


def test_an_accepted_request_resets_the_rejection_count() -> None:
    breaker = CircuitBreaker(max_consecutive_rejections=2)

    breaker.record_failure(row_rejected())
    breaker.record_success()
    breaker.record_failure(row_rejected())
    breaker.record_failure(invalid())
    breaker.record_failure(row_rejected())

    assert not breaker.is_open


def test_an_outage_between_row_rejections_does_not_reset_their_count() -> None:
    breaker = CircuitBreaker(max_consecutive_rejections=2)

    breaker.record_failure(row_rejected())
    breaker.record_failure(unavailable())
    breaker.record_failure(row_rejected())

    assert breaker.is_open


def test_a_row_rejection_resets_the_transient_failure_count() -> None:
    breaker = CircuitBreaker(max_consecutive_failures=2)

    breaker.record_failure(unavailable())
    breaker.record_failure(row_rejected())
    breaker.record_failure(unavailable())

    assert not breaker.is_open


@pytest.mark.parametrize("trailing", ["success", "invalid"])
def test_rows_finishing_after_rejections_open_the_circuit_keep_their_run(
    trailing: str,
) -> None:
    # Rows still in flight on other workers finish after the circuit opens.
    breaker = CircuitBreaker(max_consecutive_rejections=2)
    run = [row_rejected(), row_rejected()]
    for error in run:
        breaker.record_failure(error)

    if trailing == "success":
        breaker.record_success()
    else:
        breaker.record_failure(invalid())
    late = row_rejected()
    breaker.record_failure(late)

    assert breaker.rejection_run == (*run, late)


def test_transient_failures_open_the_circuit_at_the_threshold() -> None:
    breaker = CircuitBreaker(max_consecutive_failures=2)
    first = unavailable()

    breaker.record_failure(first)
    assert not breaker.is_open
    breaker.record_failure(unavailable())

    assert breaker.is_open


def test_a_success_or_a_row_failure_resets_the_count() -> None:
    breaker = CircuitBreaker(max_consecutive_failures=2)

    breaker.record_failure(unavailable())
    breaker.record_success()
    breaker.record_failure(unavailable())
    breaker.record_failure(invalid())
    breaker.record_failure(unavailable())

    assert not breaker.is_open


def test_the_first_error_to_open_the_circuit_is_kept() -> None:
    breaker = CircuitBreaker()
    first = rejected()

    breaker.record_failure(first)
    breaker.record_failure(rejected())

    assert breaker.opened_by is first


@pytest.mark.parametrize(
    "threshold", ["max_consecutive_failures", "max_consecutive_rejections"]
)
def test_a_threshold_below_one_is_refused(threshold: str) -> None:
    with pytest.raises(ValueError, match=f"{threshold} must be at least 1"):
        CircuitBreaker(**{threshold: 0})


def test_each_chunk_is_persisted_with_its_failures() -> None:
    error = invalid()
    _called, call = failing_on({"c": error})
    persist = Recorder()

    report = process_batch(list("abcde"), call, persist, chunk_size=2)

    assert [chunk.number for chunk in persist.chunks] == [1, 2, 3]
    assert [chunk.succeeded for chunk in persist.chunks] == [
        [("a", "A"), ("b", "B")],
        [("d", "D")],
        [("e", "E")],
    ]
    assert persist.chunks[1].failed == [RowFailure("c", error)]
    assert report.succeeded == 4
    assert report.failures == [RowFailure("c", error)]
    assert report.attempted == 5
    assert report.circuit_opened_by is None


def test_a_row_rejected_for_its_input_does_not_stop_the_batch() -> None:
    error = row_rejected()
    called, call = failing_on({"b": error})
    persist = Recorder()

    report = process_batch(list("abcd"), call, persist, chunk_size=2)

    assert called == list("abcd")
    assert report.succeeded == 3
    assert report.failures == [RowFailure("b", error)]
    assert not report.failures[0].retry_next_run
    assert report.circuit_opened_by is None


def test_a_run_of_row_rejections_is_reported_for_clearing() -> None:
    errors = {row: row_rejected() for row in "cde"}
    _called, call = failing_on({"a": row_rejected(), **errors})
    breaker = CircuitBreaker(max_consecutive_rejections=3)

    with pytest.raises(BatchFailed) as caught:
        process_batch(list("abcdef"), call, Recorder(), chunk_size=2, breaker=breaker)

    report = caught.value.report
    # "a" was followed by a success, so it was a rejection of its own row.
    assert report.request_rejections == [RowFailure(row, errors[row]) for row in "cde"]
    assert "3 rejected rows to clear" in str(caught.value)


def test_a_batch_whose_every_row_was_rejected_reports_them_for_clearing() -> None:
    # Too few rows to reach the threshold, but as likely a rejection of the request.
    errors = {row: row_rejected() for row in "abc"}
    _called, call = failing_on(errors)

    with pytest.raises(BatchFailed) as caught:
        process_batch(list("abc"), call, Recorder())

    report = caught.value.report
    assert report.circuit_opened_by is None
    assert report.request_rejections == [RowFailure(row, errors[row]) for row in "abc"]


@pytest.mark.parametrize("other", ["success", "invalid"])
def test_rejected_rows_beside_another_outcome_are_not_reported(other: str) -> None:
    failures: dict[str, LLMError] = {"a": row_rejected(), "b": row_rejected()}
    if other == "invalid":
        failures["c"] = invalid()
    _called, call = failing_on(failures)

    try:
        report = process_batch(list("abc"), call, Recorder())
    except BatchFailed as exc:
        report = exc.report

    assert report.request_rejections == []


def test_no_rows_are_reported_for_clearing_when_something_else_opened_it() -> None:
    _called, call = failing_on({"a": row_rejected(), "b": rejected()})

    with pytest.raises(BatchFailed) as caught:
        process_batch(list("abc"), call, Recorder())

    assert caught.value.report.request_rejections == []


def test_a_rejection_stops_the_batch_after_its_chunk() -> None:
    error = rejected()
    read: list[str] = []

    def rows() -> Iterator[str]:
        for row in "abcdef":
            read.append(row)
            yield row

    called, call = failing_on({"b": error})
    persist = Recorder()

    with pytest.raises(BatchFailed) as caught:
        process_batch(rows(), call, persist, chunk_size=3)

    assert called == ["a", "b"]
    assert read == ["a", "b", "c"]
    assert [chunk.succeeded for chunk in persist.chunks] == [[("a", "A")]]
    assert caught.value.__cause__ is error
    assert caught.value.report.circuit_opened_by is error
    assert "LLMRejected" in str(caught.value)
    assert "opened the circuit" in str(caught.value)


def test_every_row_failing_raises_with_sample_errors() -> None:
    _called, call = failing_on({row: invalid() for row in "abcd"})
    persist = Recorder()

    with pytest.raises(BatchFailed, match=r"4 of 4 rows failed\. Sample errors: "):
        process_batch(list("abcd"), call, persist, chunk_size=2)

    # Row failures do not open the circuit, so every chunk ran and was persisted.
    assert len(persist.chunks) == 2
    assert all(len(chunk.failed) == 2 for chunk in persist.chunks)


def test_an_empty_batch_is_not_a_failure() -> None:
    persist = Recorder()

    report = process_batch([], str.upper, persist)

    assert report.attempted == 0
    assert persist.chunks == []


def test_an_error_that_is_not_a_model_failure_stops_the_batch_unpersisted() -> None:
    def call(row: str) -> str:
        raise KeyError(row)

    persist = Recorder()

    with pytest.raises(KeyError):
        process_batch(["a"], call, persist)

    assert persist.chunks == []


def test_rows_run_concurrently_with_workers() -> None:
    barrier = threading.Barrier(3, timeout=5)

    def call(row: str) -> str:
        barrier.wait()
        return row.upper()

    persist = Recorder()

    report = process_batch(list("abcdef"), call, persist, chunk_size=3, max_workers=3)

    assert report.succeeded == 6
    assert [chunk.succeeded for chunk in persist.chunks] == [
        [("a", "A"), ("b", "B"), ("c", "C")],
        [("d", "D"), ("e", "E"), ("f", "F")],
    ]


@pytest.mark.parametrize("entry_point", ["complete_sync", "run_sync"])
def test_workers_run_real_clients_on_their_own_event_loops(
    real_client: UseCase, entry_point: str
) -> None:
    agent = build_agent(real_client)

    def summarize(row: str) -> str:
        if entry_point == "complete_sync":
            return complete_sync(real_client, row).output
        return run_sync(agent, row).output

    persist = Recorder()

    report = process_batch(
        list("abcdef"), summarize, persist, chunk_size=3, max_workers=3
    )

    assert report.succeeded == 6
    assert [result for chunk in persist.chunks for _, result in chunk.succeeded] == [
        "ok"
    ] * 6


class Deferred(BaseModel):
    """Built on first use, as the openai SDK builds its response models."""

    model_config = ConfigDict(defer_build=True)


def test_workers_first_using_a_deferred_model_all_see_it_built() -> None:
    # Before pydantic 2.14, every thread that found a deferred model unbuilt rebuilt
    # it with no lock, deleting and resetting the model's schema attributes while
    # other threads read them. Here that surfaces as an AttributeError for
    # __pydantic_core_schema__; in the openai SDK, as a response that cannot be
    # constructed. Each trial races a new model; a short switch interval makes the
    # threads interleave inside the rebuild.
    workers = 16
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(workers) as pool:
            for trial in range(200):
                inner = create_model(
                    f"Inner{trial}",
                    __base__=Deferred,
                    **{f"field{n}": (str, "") for n in range(5)},
                )
                model = create_model(
                    f"Outer{trial}", __base__=Deferred, inner=(list[inner], [])
                )
                start = threading.Barrier(workers)

                def first_use(
                    model: type[BaseModel] = model, start: threading.Barrier = start
                ) -> BaseModel:
                    start.wait()
                    return model.model_validate({"inner": [{}]})

                for future in [pool.submit(first_use) for _ in range(workers)]:
                    future.result()
    finally:
        sys.setswitchinterval(interval)


def test_a_rejection_with_workers_stops_later_chunks() -> None:
    called, call = failing_on({"a": rejected()})

    with pytest.raises(BatchFailed):
        process_batch(list("abcd"), call, Recorder(), chunk_size=2, max_workers=2)

    assert sorted(called) == ["a", "b"]


def test_no_row_starts_on_a_worker_once_the_circuit_opens() -> None:
    called: list[int] = []
    lock = threading.Lock()

    def call(row: int) -> str:
        with lock:
            called.append(row)
        time.sleep(0.02)
        raise rejected()

    with pytest.raises(BatchFailed) as caught:
        process_batch(range(200), call, Recorder(), chunk_size=200, max_workers=8)

    # Only the rows already running when the first rejection came back ran.
    assert len(called) <= 8
    assert len(caught.value.report.failures) == len(called)


def test_an_open_breaker_passed_in_stops_the_batch_before_any_row() -> None:
    breaker = CircuitBreaker()
    breaker.record_failure(rejected())
    called, call = failing_on({})

    with pytest.raises(BatchFailed):
        process_batch(["a"], call, Recorder(), breaker=breaker)

    assert called == []


@pytest.mark.parametrize("arguments", [{"chunk_size": 0}, {"max_workers": 0}])
def test_sizes_below_one_are_refused(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        process_batch(["a"], str.upper, Recorder(), **arguments)  # type: ignore[arg-type]


def test_batch_failed_survives_pickling() -> None:
    _called, call = failing_on({"a": invalid()})
    with pytest.raises(BatchFailed) as caught:
        process_batch(["a"], call, Recorder())

    copy = pickle.loads(pickle.dumps(caught.value))  # noqa: S301

    assert str(copy) == str(caught.value)
    assert copy.report.failures[0].row == "a"
    assert isinstance(copy.report.failures[0].error, LLMInvalidOutput)


def test_the_dagster_pool_is_named_for_the_use_case() -> None:
    assert dagster_pool(SUMMARY) == "llm/ol-data-platform/feedback-summary"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"calls_per_minute": 120}, "120/m"),
        ({"calls_per_minute": 100, "rows_per_task": 20, "workers": 2}, "2.5/m"),
    ],
)
def test_the_celery_rate_limit_counts_calls_not_tasks(
    arguments: dict[str, float], expected: str
) -> None:
    assert celery_rate_limit(**arguments) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        {"calls_per_minute": 0},
        {"calls_per_minute": 10, "rows_per_task": 0},
        {"calls_per_minute": 10, "workers": 0},
    ],
)
def test_a_celery_rate_limit_needs_positive_values(arguments: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        celery_rate_limit(**arguments)  # type: ignore[arg-type]
