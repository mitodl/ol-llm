"""The retry budget every model call runs under (spec §3.2 rule 1, §6.3).

Kept free of PydanticAI so the LangChain wrapper (``ol_llm.langchain``) and the
decision entry points (``ol_llm.decisions``) apply the same budget and error
classes as the other entry points without the ``pydantic-ai`` extra.
"""

import email.utils
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import httpx
import httpx2

from ol_llm.errors import LLMError, LLMRateLimited, LLMRejected, LLMUnavailable
from ol_llm.policy import CallPolicy, Profile

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
