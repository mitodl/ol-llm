"""The retry budget every model call runs under (spec §3.2 rule 1, §6.3).

Kept free of PydanticAI so the LangChain wrapper (``ol_llm.langchain``) applies
the same budget as the entry points without the ``pydantic-ai`` extra.
"""

import random
from dataclasses import dataclass

import httpx
import httpx2

from ol_llm.errors import LLMError, LLMRateLimited
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
