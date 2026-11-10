"""The typed errors memory operations raise, in the style of the §3 contract.

Each carries the exception that caused it as ``__cause__`` and says whether a
retry can help. Failures are classified by status and exception type, never by
message text. The service's message stays on the exception for logs and traces and
is never shown to users.
"""

from typing import Any, ClassVar

import httpx
import httpx2

from ol_llm.errors import _rebuild

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
"""HTTP statuses that mean the service did not answer and a retry can help."""

# The Hindsight client and the token exchange send with httpx2; client-credentials
# tokens come from the cache shared with Opik, which sends with httpx.
_TIMEOUT_ERRORS = (httpx.TimeoutException, httpx2.TimeoutException)
_UNAVAILABLE_TRANSPORT_ERRORS = (
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx2.NetworkError,
    httpx2.RemoteProtocolError,
)


class MemoryStoreError(Exception):
    """Base class for every memory failure the library raises.

    :param message: what went wrong, for logs and traces.
    :param operation: the operation that failed, e.g. ``put`` or ``forget``.
    :param status_code: the HTTP status the service answered with, if any.
    """

    retryable: ClassVar[bool] = False

    def __init__(
        self, message: str, *, operation: str, status_code: int | None = None
    ) -> None:
        """Record the operation the failure belongs to."""
        super().__init__(message)
        self.operation = operation
        self.status_code = status_code

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by attributes, as ``LLMError`` does, for Celery and process pools."""
        return (_rebuild, (type(self), self.args, self.__dict__))


class MemoryNotFound(MemoryStoreError):
    """No such note, or a bank the caller may not see (another learner's is a 404)."""


class MemoryForbidden(MemoryStoreError):
    """The caller may not do this, e.g. write general memory without the grant."""


class MemoryFenced(MemoryStoreError):
    """The learner's memory is being forgotten, and writes are refused until it is.

    The fence outlasts the forget (an hour on the deployed service), so retrying
    the write soon will not help.
    """


class MemoryUnauthenticated(MemoryStoreError):
    """The token was missing, expired, or not one the service accepts.

    A learner token comes from the session at request start; exchange again from a
    fresh session token rather than retrying with the same one.
    """


class MemoryUnavailable(MemoryStoreError):
    """No answer came back: a dropped connection, a 408, 429 or 5xx."""

    retryable = True


class MemoryTimeout(MemoryUnavailable):
    """The request ran past its timeout."""


class MemoryRejected(MemoryStoreError):
    """The service refused the request as malformed, or the client is misconfigured.

    Our bug or configuration: a 400 or 422, an unexpected status, or a transport
    error retrying cannot fix (a bad URL scheme, a proxy failure).
    """


class MemoryUnsupported(MemoryStoreError):
    """The configured backend does not have this capability."""


_BY_STATUS: dict[int, type[MemoryStoreError]] = {
    401: MemoryUnauthenticated,
    403: MemoryForbidden,
    404: MemoryNotFound,
    409: MemoryFenced,
}


def from_status(status_code: int, message: str, *, operation: str) -> MemoryStoreError:
    """Return the error for an HTTP status the service answered with.

    :param status_code: the status.
    :param message: the service's message.
    :param operation: the operation that failed.
    :returns: the error.
    :rtype: MemoryStoreError
    """
    if status_code in RETRYABLE_STATUSES:
        kind: type[MemoryStoreError] = MemoryUnavailable
    else:
        kind = _BY_STATUS.get(status_code, MemoryRejected)
    return kind(message, operation=operation, status_code=status_code)


def from_transport(
    exc: httpx.TransportError | httpx2.TransportError, *, operation: str
) -> MemoryStoreError:
    """Return the error for a request that got no answer.

    :param exc: what ``httpx`` or ``httpx2`` raised.
    :param operation: the operation that failed.
    :returns: the error.
    :rtype: MemoryStoreError
    """
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, _TIMEOUT_ERRORS):
        return MemoryTimeout(message, operation=operation)
    if isinstance(exc, _UNAVAILABLE_TRANSPORT_ERRORS):
        return MemoryUnavailable(message, operation=operation)
    return MemoryRejected(message, operation=operation)
