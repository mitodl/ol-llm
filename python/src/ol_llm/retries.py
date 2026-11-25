"""The retry budget every model call runs under (spec §3.2 rule 1, §6.3).

Kept free of PydanticAI so the LangChain wrapper (``ol_llm.langchain``) applies
the same budget and error classes as the other entry points without the
``pydantic-ai`` extra.
"""

import email.utils
import random
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import httpx2

from ol_llm.errors import LLMError, LLMRateLimited, LLMRejected, LLMUnavailable
from ol_llm.policy import CallPolicy, Profile

_T = TypeVar("_T")

RETRYABLE_STATUSES = frozenset({408, 500, 502, 503, 504, 529})
"""HTTP statuses that mean the provider is unavailable (spec §6.3)."""

UNAVAILABLE_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx2.TimeoutException,
    httpx2.NetworkError,
    httpx2.RemoteProtocolError,
)
"""Transport failures that mean no answer came back."""

TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    httpx.TransportError,
    httpx2.TransportError,
)
"""Every transport failure. Those not in ``UNAVAILABLE_TRANSPORT_ERRORS`` are
permanent (a bad URL scheme, a proxy failure) and are not retried."""

RATE_LIMITED_STATUS = 429
INITIAL_BACKOFF_SECONDS = 0.5
MILLISECONDS_PER_SECOND = 1000
UNMAPPED_ERROR_STATUS = 500
"""The status litellm gives an error it cannot map to a provider response."""


@dataclass
class RetryBudget:
    """The retries and waits one call has left under its policy.

    :param policy: the resolved policy.
    :param profile: the use case's profile, which decides how backoff is capped.
    """

    policy: CallPolicy
    profile: Profile
    retries: int = 0
    waited: float = 0

    def next_delay(self, error: LLMError) -> float | None:
        """Return how long to wait before the next attempt, or None to give up."""
        if not error.retryable or self.retries >= self.policy.max_retries:
            return None
        backoff = INITIAL_BACKOFF_SECONDS * 2**self.retries
        # Jitter spreads out callers that failed together; it is not security.
        delay = random.uniform(backoff / 2, backoff)  # noqa: S311
        retry_after = error.retry_after if isinstance(error, LLMRateLimited) else None
        if retry_after is not None:
            delay = max(delay, retry_after)
        if self.profile == "request":
            # A total budget: a Retry-After longer than what is left ends the call
            # so the caller can tell the user instead of holding a request open.
            # Once the budget is spent, retrying with no wait would hammer a
            # provider that is already failing.
            remaining = self.policy.max_backoff_seconds - self.waited
            if remaining <= 0 or (retry_after is not None and retry_after > remaining):
                return None
            delay = min(delay, remaining)
        else:
            delay = min(delay, self.policy.max_backoff_seconds)
        self.retries += 1
        self.waited += delay
        return delay


Classify = Callable[[Exception], LLMError | None]
"""Maps a failed attempt to its library error, or ``None`` to raise it as is."""


def delay_or_raise(exc: Exception, budget: RetryBudget, classify: Classify) -> float:
    """Return the wait before the next attempt, or raise what the caller sees.

    Callers catch ``Exception`` and hand it here.

    :param exc: what the attempt raised.
    :param budget: the call's retry budget.
    :param classify: the call path's classifier.
    :returns: the wait in seconds.
    :rtype: float
    :raises LLMError: once the failure is not retryable or the retries are spent,
        with ``exc`` as its cause. ``exc`` itself when it is not a model failure.
    """
    if (error := classify(exc)) is None:
        raise exc
    if (delay := budget.next_delay(error)) is None:
        raise error from exc
    return delay


async def retry_attempts(
    attempt: Callable[[], Awaitable[_T]],
    *,
    budget: RetryBudget,
    delay: Callable[[Exception, RetryBudget], float],
    sleep: Callable[[float], Awaitable[Any]],
) -> _T:
    """Make attempts until one succeeds or ``delay`` gives up.

    :param attempt: makes one attempt.
    :param budget: the call's retry budget.
    :param delay: the call path's ``delay_or_raise``, with its classifier bound.
    :param sleep: waits between attempts; tests pass one that records instead.
    :returns: what the first successful attempt returned.
    :rtype: _T
    :raises LLMError: as ``delay_or_raise`` does.
    """
    while True:
        try:
            return await attempt()
        except Exception as exc:  # noqa: BLE001
            wait = delay(exc, budget)
        await sleep(wait)


def classify_http_failure(
    exc: BaseException, *, use_case: str, model: str
) -> LLMError | None:
    """Map a failed HTTP model request to the library error for its class.

    For clients ol-llm does not reach through PydanticAI: a LangChain chat model,
    and the decision entry points' own ``httpx2`` client. The status comes from
    the HTTP response in the exception's chain (``original_exception``,
    ``__cause__``, ``__context__``) when there is one, else from an httpx timeout
    or disconnect there, else from the exception's ``status_code``. A bare 500 is
    a rejection: litellm gives 500 to whatever it cannot map, e.g. an error object
    inside a 200 stream or a failing credential hook, and nothing in it says the
    provider is down.

    :param exc: what the request raised.
    :param use_case: the use case name, for the error.
    :param model: the model id, for the error.
    :returns: the library error, or ``None`` when ``exc`` is not a model request
        failure.
    :rtype: LLMError | None
    """
    message = str(exc)
    chain = list(_exception_chain(exc))
    status = next(
        (
            link.response.status_code
            for link in chain
            if isinstance(link, httpx.HTTPStatusError | httpx2.HTTPStatusError)
        ),
        None,
    )
    if status is None:
        if any(isinstance(link, UNAVAILABLE_TRANSPORT_ERRORS) for link in chain):
            return LLMUnavailable(message, use_case=use_case, model=model)
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int) or isinstance(status, bool):
            return None
        if status == UNMAPPED_ERROR_STATUS:
            return LLMRejected(
                message, use_case=use_case, model=model, status_code=None
            )
    if status == RATE_LIMITED_STATUS:
        return LLMRateLimited(
            message, use_case=use_case, model=model, retry_after=_retry_after(chain)
        )
    if status in RETRYABLE_STATUSES:
        return LLMUnavailable(message, use_case=use_case, model=model)
    return LLMRejected(message, use_case=use_case, model=model, status_code=status)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and every exception it wraps, each once."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        link = pending.pop(0)
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        yield link
        original = getattr(link, "original_exception", None)
        pending.extend(
            (
                original if isinstance(original, BaseException) else None,
                link.__cause__,
                link.__context__,
            )
        )


def _retry_after(chain: Sequence[BaseException]) -> float | None:
    """Read the seconds a rate-limited response asked to wait, if it said."""
    for link in chain:
        if headers := getattr(link, "litellm_response_headers", None):
            return _parse_retry_after(headers)
    for link in chain:
        response = getattr(link, "response", None)
        if isinstance(response, httpx.Response | httpx2.Response) and response.headers:
            return _parse_retry_after(response.headers)
    return None


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    if (milliseconds := lowered.get("retry-after-ms")) is not None:
        try:
            return float(milliseconds) / MILLISECONDS_PER_SECOND
        except ValueError:
            pass
    if (value := lowered.get("retry-after")) is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(when.timestamp() - time.time(), 0.0)
