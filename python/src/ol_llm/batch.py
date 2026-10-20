"""Helpers for batch call sites: Dagster assets and Celery tasks (spec §3.2 rule 7).

``process_batch`` makes one call per row, keeps a failed row from failing its chunk,
hands each chunk to the caller's persist callback as soon as it finishes, and stops
when a ``CircuitBreaker`` opens. ``BatchKeys`` builds the idempotency key that tells a
run which rows a prompt or model change has made stale. ``dagster_pool`` and
``celery_rate_limit`` produce the values those frameworks take, without importing
either.

Needs no extra. Model failures arrive as ``LLMError`` only from models the library
builds, so rows are isolated only from those; anything else a call raises is a bug
and stops the batch.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from itertools import islice
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ol_llm.attribution import DegradedReason
from ol_llm.errors import LLMError, LLMRateLimited, LLMRejected, LLMUnavailable
from ol_llm.policy import UseCase
from ol_llm.prompts import ManagedPrompt

if TYPE_CHECKING:
    from ol_llm.runs import LLMResult

RowT = TypeVar("RowT")
ResultT = TypeVar("ResultT")

# Bounds how many calls a crash can lose; ol-data-platform's summaries use 200.
DEFAULT_CHUNK_SIZE = 200
# Each of these failures has already spent the batch profile's retries, up to
# 3,300 s at the defaults, so a short run of them is an outage, not bad luck.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
# A row rejection costs one attempt and no tokens, so the threshold can be loose;
# it is there for a request every row would get rejected for.
DEFAULT_MAX_CONSECUTIVE_REJECTIONS = 10
SAMPLE_ERROR_COUNT = 3

# Statuses that reject one row's input (too long, malformed, filtered on the way in)
# rather than the account, the model, or the request every row makes.
ROW_REJECTION_STATUSES = frozenset({400, 413, 422})

_COUNTED = (LLMRateLimited, LLMUnavailable)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """How one row's output was produced; ``dataclasses.asdict`` gives the columns.

    :param idempotency_key: the row's key, from ``BatchKeys.key``.
    :param model: the model that answered, which is a fallback when ``degraded``.
    :param degraded_reasons: why the result is degraded; empty when it is not.
    :param prompt_name: the prompt the run used.
    :param prompt_version: the Opik version the run resolved, for linkage.
    :param prompt_template_hash: the prompt's ``template_hash``, which the key
        includes.
    :param processed_at: when the row's result was recorded, in UTC.
    """

    idempotency_key: str
    model: str
    degraded_reasons: tuple[DegradedReason, ...]
    prompt_name: str
    prompt_version: str | None
    prompt_template_hash: str
    processed_at: datetime


@dataclass(frozen=True, kw_only=True)
class BatchKeys:
    """What a row's output depends on besides its input, fixed for one run.

    A batch run fetches its prompt once at the start (lifecycle spec §3 rule 2),
    builds this from it, and skips each row whose stored key equals ``key(input)``.

    The key uses the prompt's ``template_hash``, not its version (lifecycle spec §3
    rule 5): a run that fell back to the embedded default makes the same key as one
    that fetched the same template, so an Opik outage does not reprocess every row.

    :param model: the use case's primary model, ``use_case.model``. A row a
        fallback model answered keeps the primary's key, so the next run does not
        redo it; its provenance names the model that answered.
    :param prompt: the prompt the run fetched with ``get_prompt``.
    """

    model: str
    prompt: ManagedPrompt[Any]

    def key(self, row_input: str) -> str:
        """Return the idempotency key for one row.

        :param row_input: what the row puts into the prompt, e.g. the text to
            summarize, serialized the same way every run.
        :returns: a hex SHA-256 that changes when the input, the prompt name, the
            template, or the model does.
        :rtype: str
        """
        parts = [row_input, self.prompt.name, self.prompt.template_hash, self.model]
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()

    def provenance(self, row_input: str, result: "LLMResult[Any]") -> Provenance:
        """Return the provenance of one row's result.

        The prompt fields, and the prompt's ``embedded_prompt`` reason, come from
        ``prompt`` as well as the result, so they describe the key even for a run
        that was not passed ``prompt=``.

        :param row_input: the value passed to ``key`` for this row.
        :param result: the row's run result.
        :returns: the provenance, stamped with the current time.
        :rtype: Provenance
        """
        return Provenance(
            idempotency_key=self.key(row_input),
            model=result.model,
            degraded_reasons=tuple(
                dict.fromkeys((*result.degraded_reasons, *self.prompt.degraded_reasons))
            ),
            prompt_name=self.prompt.name,
            prompt_version=self.prompt.version,
            prompt_template_hash=self.prompt.template_hash,
            processed_at=datetime.now(UTC),
        )


def rejects_the_row(error: LLMError) -> bool:
    """Whether a rejection is about one row's input rather than the whole batch.

    :param error: what the row's call raised.
    :returns: ``True`` for an ``LLMRejected`` with a status in
        ``ROW_REJECTION_STATUSES``.
    :rtype: bool
    """
    return (
        isinstance(error, LLMRejected) and error.status_code in ROW_REJECTION_STATUSES
    )


@dataclass(frozen=True)
class RowFailure(Generic[RowT]):
    """One entry in the failure ledger.

    :param row: the row whose call failed.
    :param error: what the call raised.
    """

    row: RowT
    error: LLMError

    @property
    def retry_next_run(self) -> bool:
        """Whether the next run should try the row again under the same key.

        A rate limit, an outage, or a rejection of the account or the model says
        nothing about the row: the next run, or the one after the configuration is
        fixed, can succeed. Invalid output, a content filter, an exceeded usage
        limit, or a rejection of the row's input (``rejects_the_row``) comes from
        the input under this prompt and model, so the same key would fail the same
        way; a prompt or model change makes a new key.
        """
        if isinstance(self.error, LLMRejected):
            return not rejects_the_row(self.error)
        return isinstance(self.error, _COUNTED)


@dataclass
class CircuitBreaker:
    """Stops a batch on failures the remaining rows would repeat (spec §6.7).

    An ``LLMRejected`` of the account or the model (a 401, 403, 404, or a transport
    error) opens it at once, because it is our bug or our configuration and every
    later row would hit it too. A rejection of one row's input (a 400, 413, or 422;
    see ``rejects_the_row``) fails that row only, unless
    ``max_consecutive_rejections`` come in a row, which means the request itself is
    wrong (e.g. a parameter the model no longer takes). ``LLMRateLimited`` and
    ``LLMUnavailable`` open it after ``max_consecutive_failures`` in a row.

    A success resets both counts. A row's own failure (invalid output, a content
    filter, an exceeded usage limit) shows the provider is answering and accepted
    the request, so it resets both too. A rate limit or an outage says nothing about
    whether the request is valid and leaves the rejection count alone; a row
    rejection is an answer and resets the failure count.

    :param max_consecutive_failures: rate-limited or unavailable rows in a row that
        open the circuit.
    :param max_consecutive_rejections: rows rejected for their input in a row that
        open the circuit.
    :raises ValueError: if either threshold is less than 1.
    """

    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    max_consecutive_rejections: int = DEFAULT_MAX_CONSECUTIVE_REJECTIONS
    opened_by: LLMError | None = field(default=None, init=False)
    _consecutive: int = field(default=0, init=False, repr=False)
    _rejection_run: list[LLMError] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Refuse a threshold that could never be reached or is always reached."""
        for name in ("max_consecutive_failures", "max_consecutive_rejections"):
            if getattr(self, name) < 1:
                msg = f"CircuitBreaker.{name} must be at least 1"
                raise ValueError(msg)

    @property
    def is_open(self) -> bool:
        """Whether the batch should stop."""
        return self.opened_by is not None

    @property
    def rejection_run(self) -> tuple[LLMError, ...]:
        """The row rejections since the last accepted request, oldest first.

        Once these open the circuit they are more likely rejections of the request
        than of their rows, and rows that fail after the circuit opens (in flight
        on other workers) are still added.
        """
        return tuple(self._rejection_run)

    def record_success(self) -> None:
        """Count a row that succeeded."""
        self._consecutive = 0
        self._rejection_run.clear()

    def record_failure(self, error: LLMError) -> None:
        """Count a row that failed, and open the circuit if it should.

        :param error: what the row's call raised.
        """
        if isinstance(error, _COUNTED):
            self._consecutive += 1
            if self._consecutive >= self.max_consecutive_failures:
                self._open(error)
        elif rejects_the_row(error):
            self._consecutive = 0
            self._rejection_run.append(error)
            if len(self._rejection_run) >= self.max_consecutive_rejections:
                self._open(error)
        elif isinstance(error, LLMRejected):
            self._open(error)
        else:
            self._consecutive = 0
            self._rejection_run.clear()

    def _open(self, error: LLMError) -> None:
        # The first error is the cause; rows still in flight may fail after it.
        if self.opened_by is None:
            self.opened_by = error


@dataclass(frozen=True)
class Chunk(Generic[RowT, ResultT]):
    """One chunk's outcome, handed to the persist callback.

    :param number: the chunk's position in the batch, from 1.
    :param succeeded: each successful row with its call's result.
    :param failed: the ledger entries for the rows that failed.
    """

    number: int
    succeeded: Sequence[tuple[RowT, ResultT]]
    failed: Sequence[RowFailure[RowT]]


@dataclass(frozen=True, kw_only=True)
class BatchReport(Generic[RowT]):
    """What a batch did.

    :param succeeded: rows whose call succeeded.
    :param failures: the failure ledger, in the order rows were processed.
    :param circuit_opened_by: the error that opened the circuit, if it opened.
    :param request_rejections: when a run of row rejections opened the circuit,
        the failures in that run. The request was most likely wrong for every row
        (e.g. a parameter the model no longer takes), but each was persisted as a
        row not to retry, and fixing the request does not change the row's key.
        Clear their skip marks once the request is fixed.
    """

    succeeded: int
    failures: Sequence[RowFailure[RowT]]
    circuit_opened_by: LLMError | None
    request_rejections: Sequence[RowFailure[RowT]] = ()

    @property
    def attempted(self) -> int:
        """Rows whose call ran, successful or not."""
        return self.succeeded + len(self.failures)

    def describe(self) -> str:
        """Summarize the outcome with a few sample errors, for logs and failures.

        :returns: one line naming the counts, the circuit's cause, and up to
            ``SAMPLE_ERROR_COUNT`` errors.
        :rtype: str
        """
        summary = f"{len(self.failures)} of {self.attempted} rows failed"
        if (error := self.circuit_opened_by) is not None:
            summary += (
                f"; stopped after {type(error).__name__} from {error.model} opened "
                f"the circuit for use case {error.use_case!r}"
            )
        if self.request_rejections:
            summary += (
                f"; {len(self.request_rejections)} rejected rows to clear and retry "
                "once the request is fixed"
            )
        samples = "; ".join(
            f"{type(failure.error).__name__}: {failure.error}"
            for failure in self.failures[:SAMPLE_ERROR_COUNT]
        )
        return f"{summary}. Sample errors: {samples}" if samples else summary


class BatchFailed(Exception):  # noqa: N818
    """The batch stopped on an open circuit, or every attempted row failed.

    Either would otherwise look like a run with nothing to do. Every chunk processed
    before the stop has already been persisted.

    :param message: the report's description.
    :param report: what the batch did before it stopped.
    """

    def __init__(self, message: str, report: BatchReport[Any]) -> None:
        """Keep the report for callers that record counts on failure."""
        super().__init__(message)
        self.report = report

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with the report, for Celery and process pools."""
        return (type(self), (str(self), self.report))


def _chunks(
    rows: Iterable[RowT], size: int, breaker: CircuitBreaker
) -> Iterator[list[RowT]]:
    """Yield chunks of rows, reading none past the chunk in which the breaker opens."""
    iterator = iter(rows)
    while not breaker.is_open and (chunk := list(islice(iterator, size))):
        yield chunk


def _inline(
    rows: list[RowT], call: Callable[[RowT], ResultT], breaker: CircuitBreaker
) -> Iterator[tuple[int, Callable[[], ResultT]]]:
    for index, row in enumerate(rows):
        if breaker.is_open:
            return
        yield index, partial(call, row)


def _threaded(
    rows: list[RowT],
    call: Callable[[RowT], ResultT],
    breaker: CircuitBreaker,
    executor: ThreadPoolExecutor,
    max_workers: int,
) -> Iterator[tuple[int, Callable[[], ResultT]]]:
    """Yield outcomes as rows finish, starting a row only while the breaker is closed.

    At most ``max_workers`` rows are in flight, so once the breaker opens nothing
    new starts; rows already running are still yielded and recorded.
    """
    unstarted = iter(enumerate(rows))
    running: dict[Future[ResultT], int] = {}
    while True:
        while (
            len(running) < max_workers
            and not breaker.is_open
            and (item := next(unstarted, None)) is not None
        ):
            running[executor.submit(call, item[1])] = item[0]
        if not running:
            return
        finished, _ = wait(running, return_when=FIRST_COMPLETED)
        for future in finished:
            yield running.pop(future), future.result


def _run_chunk(
    rows: list[RowT],
    outcomes: Iterator[tuple[int, Callable[[], ResultT]]],
    breaker: CircuitBreaker,
    log: logging.Logger,
) -> tuple[list[tuple[RowT, ResultT]], list[RowFailure[RowT]]]:
    succeeded: list[tuple[int, tuple[RowT, ResultT]]] = []
    failed: list[tuple[int, RowFailure[RowT]]] = []
    for index, outcome in outcomes:
        row = rows[index]
        try:
            result = outcome()
        except LLMError as error:
            failure = RowFailure(row, error)
            failed.append((index, failure))
            breaker.record_failure(error)
            log.warning(
                "Row failed with %s (%s next run): %s",
                type(error).__name__,
                "retry" if failure.retry_next_run else "no retry",
                error,
            )
        else:
            succeeded.append((index, (row, result)))
            breaker.record_success()
    # Threaded rows finish out of order; persist sees them in row order.
    return (
        [pair for _, pair in sorted(succeeded, key=itemgetter(0))],
        [failure for _, failure in sorted(failed, key=itemgetter(0))],
    )


def _request_rejections(
    breaker: CircuitBreaker, failures: Sequence[RowFailure[RowT]]
) -> list[RowFailure[RowT]]:
    """Return the failures in the rejection run that opened the circuit, if one did."""
    if breaker.opened_by is None or not rejects_the_row(breaker.opened_by):
        return []
    run = {id(error) for error in breaker.rejection_run}
    return [failure for failure in failures if id(failure.error) in run]


def process_batch(  # noqa: PLR0913
    rows: Iterable[RowT],
    call: Callable[[RowT], ResultT],
    persist: Callable[[Chunk[RowT, ResultT]], None],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_workers: int = 1,
    breaker: CircuitBreaker | None = None,
    logger: logging.Logger | None = None,
) -> BatchReport[RowT]:
    """Call ``call`` on every row, persisting each chunk as soon as it finishes.

    A row whose call raises an ``LLMError`` goes in the failure ledger and does not
    fail its chunk. ``persist`` gets every chunk, including one where every row
    failed, so it can record the failures that should not be retried. A crash
    loses at most the chunk in progress; the next run's key check skips what was
    persisted. The batch stops after the chunk in which the breaker opens.

    :param rows: the rows to process, read one chunk at a time.
    :param call: makes the model call for one row, e.g. with
        ``ol_llm.complete_sync``.
    :param persist: writes one chunk, e.g. an upsert keyed on the row.
    :param chunk_size: rows per chunk, so per ``persist`` call.
    :param max_workers: rows of a chunk called at once, on threads. Use
        ``complete_sync`` from threads: an agent built once and passed to
        ``run_sync`` needs ``max_workers=1`` until agents work across event loops.
        Threads do not inherit the caller's Opik context, so with more than one
        each row's run is its own root trace rather than a span of the caller's
        trace.
    :param breaker: the circuit breaker; a new ``CircuitBreaker`` when omitted.
    :param logger: where to log chunk progress and row failures, e.g. Dagster's
        ``context.log``; this module's logger when omitted.
    :returns: what the batch did.
    :rtype: BatchReport
    :raises BatchFailed: if the circuit opened, or every attempted row failed.
    :raises ValueError: if ``chunk_size`` or ``max_workers`` is less than 1.
    """
    if chunk_size < 1 or max_workers < 1:
        msg = "chunk_size and max_workers must be at least 1"
        raise ValueError(msg)
    breaker = CircuitBreaker() if breaker is None else breaker
    log = _logger if logger is None else logger
    succeeded = 0
    failures: list[RowFailure[RowT]] = []
    pool = ThreadPoolExecutor(max_workers) if max_workers > 1 else None
    with pool or nullcontext():
        chunks = _chunks(rows, chunk_size, breaker)
        for number, chunk_rows in enumerate(chunks, start=1):
            outcomes = (
                _inline(chunk_rows, call, breaker)
                if pool is None
                else _threaded(chunk_rows, call, breaker, pool, max_workers)
            )
            done, failed = _run_chunk(chunk_rows, outcomes, breaker, log)
            persist(Chunk(number, done, failed))
            succeeded += len(done)
            failures.extend(failed)
            log.info(
                "Persisted chunk %d: %d succeeded, %d failed",
                number,
                len(done),
                len(failed),
            )
    report = BatchReport(
        succeeded=succeeded,
        failures=failures,
        circuit_opened_by=breaker.opened_by,
        request_rejections=_request_rejections(breaker, failures),
    )
    if report.circuit_opened_by is not None or (failures and not succeeded):
        raise BatchFailed(report.describe(), report) from report.circuit_opened_by
    return report


def dagster_pool(use_case: UseCase) -> str:
    """Return the Dagster pool for every asset or op that calls this use case.

    One pool per use case, so the limit set on the instance (``dagster instance
    concurrency set <pool> <limit>``) bounds the use case across assets and code
    locations. A step holds one slot however many rows it calls at once, so the
    most concurrent calls is the pool's limit times ``process_batch``'s
    ``max_workers``.

    :param use_case: the call site.
    :returns: the pool name, e.g. ``llm/ol-data-platform/feedback-summary``.
    :rtype: str
    """
    return f"llm/{use_case.app}/{use_case.name}"


def celery_rate_limit(
    calls_per_minute: float, *, rows_per_task: int = 1, workers: int = 1
) -> str:
    """Return a Celery ``rate_limit`` that holds a task to a number of model calls.

    Celery limits task starts per worker instance, not calls across the fleet,
    so this divides the call budget by the calls each task makes and by the
    workers consuming the task. The policy carries no throughput figure; batch
    throughput is the orchestrator's to set (usage policies spec §1.3).

    :param calls_per_minute: model calls per minute across all workers.
    :param rows_per_task: rows each task processes, one call per row.
    :param workers: Celery worker instances that consume the task.
    :returns: the rate, e.g. ``2.5/m``.
    :rtype: str
    :raises ValueError: if a value is not positive.
    """
    if calls_per_minute <= 0 or rows_per_task < 1 or workers < 1:
        msg = "calls_per_minute, rows_per_task, and workers must be positive"
        raise ValueError(msg)
    return f"{calls_per_minute / rows_per_task / workers:g}/m"
