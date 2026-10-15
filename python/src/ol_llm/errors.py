"""The typed errors callers see for model failures (spec §3.2 rule 5, §6.7).

Each carries the exception that caused it as ``__cause__``. Request-time surfaces map
the class to their own user-facing text; the provider's message stays on the
exception and the trace and is never shown to users.
"""

import copy
from typing import Any, ClassVar, Self


class LLMError(Exception):
    """Base class for every model failure the library raises.

    :param message: what went wrong, for logs and traces.
    :param use_case: the name of the use case that made the call.
    :param model: the model id that failed.
    """

    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, use_case: str, model: str) -> None:
        """Record the call the failure belongs to."""
        super().__init__(message)
        self.use_case = use_case
        self.model = model

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by attributes, because the keyword-only fields defeat the default.

        Celery and process pools pickle exceptions to hand them back to the caller;
        without this the caller gets a wrapper its ``except LLMRateLimited`` misses.
        """
        return (_rebuild, (type(self), self.args, self.__dict__))

    def copy(self) -> Self:
        """Return a copy that can take a different ``__cause__``.

        Fallback re-raises the primary model's error with the whole exception group
        as its cause; copying keeps the original's own cause intact inside the group.
        """
        return copy.copy(self)


def _rebuild(
    cls: type[LLMError], args: tuple[object, ...], state: dict[str, object]
) -> LLMError:
    error = cls.__new__(cls)
    error.args = args
    error.__dict__.update(state)
    return error


class LLMRateLimited(LLMError):
    """The provider answered 429.

    :param retry_after: seconds the provider asked us to wait, when it said.
    """

    retryable = True

    def __init__(
        self, message: str, *, use_case: str, model: str, retry_after: float | None
    ) -> None:
        """Record the provider's requested wait."""
        super().__init__(message, use_case=use_case, model=model)
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """No answer came back: a timeout, a dropped connection, or a 408 or 5xx."""

    retryable = True


class LLMRejected(LLMError):
    """The provider refused the request, which is our bug or our configuration.

    :param status_code: the HTTP status, or ``None`` for a transport error that
        retrying cannot fix (a bad URL scheme, a proxy failure).
    """

    def __init__(
        self, message: str, *, use_case: str, model: str, status_code: int | None
    ) -> None:
        """Record the status the provider answered with."""
        super().__init__(message, use_case=use_case, model=model)
        self.status_code = status_code


class LLMContentFiltered(LLMError):
    """The provider's content filter stopped the response."""


class LLMInvalidOutput(LLMError):
    """The model did not produce valid output once its output retries were spent."""


class LLMBudgetExceeded(LLMError):
    """A usage limit on the run (requests, tokens, or cost) was reached."""
