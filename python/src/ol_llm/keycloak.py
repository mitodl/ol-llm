"""Keycloak client-credentials tokens, and the auth that sends them.

Our Opik sits behind Keycloak at APISIX, which validates a client-credentials JWT
on every ``/api`` request. One ``KeycloakTokens`` cache serves every client that
sends there: the Opik SDK's ``httpx`` client (``HttpxBearerAuth``), the span
exporter's ``requests`` session (``RequestsBearerAuth``), and the memory service's
token exchange. ``ol_llm.tracing`` builds the cache from ``OPIK_KEYCLOAK_*``.

A forked child gets a new HTTP client for the token endpoint, whose pooled
sockets the parent still uses, and a new lock.
"""

import threading
import time
import weakref
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import httpx
import requests
from requests.auth import AuthBase

from ol_llm.loops import register_at_fork

# Renew this long before the stated expiry, so a token is not used as it expires.
REFRESH_SKEW_SECONDS = 30
# A hung Keycloak must not wedge the SDK's sender threads or the span exporter.
TOKEN_REQUEST_TIMEOUT_SECONDS = 10
# Used when the token response carries no expires_in.
DEFAULT_TOKEN_LIFETIME_SECONDS = 60


@dataclass(frozen=True, kw_only=True)
class KeycloakClient:
    """A Keycloak client that gets Opik tokens by the client-credentials grant.

    :param token_url: the realm's token endpoint.
    :param client_id: the client.
    :param client_secret: the client's secret.
    :param scope: the scope to request, when the client needs one.
    """

    token_url: str
    client_id: str
    client_secret: str = field(repr=False)
    scope: str | None = None


class KeycloakTokens:
    """One cached Keycloak access token, shared by every client that sends to Opik.

    The token is renewed ``REFRESH_SKEW_SECONDS`` before it expires, and on demand
    after a 401. Thread-safe: the SDK's sender threads and the span exporter's
    thread share it.
    """

    def __init__(
        self, client: KeycloakClient, *, http: httpx.Client | None = None
    ) -> None:
        """Hold the client; no token is fetched until one is needed.

        :param client: the Keycloak client.
        :param http: the HTTP client for the token endpoint. Never one that sends
            to Opik, so a token request does not recurse through its own auth.
        """
        self._client = client
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=TOKEN_REQUEST_TIMEOUT_SECONDS)
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0
        _token_caches.add(self)

    def _after_fork(self) -> None:
        # A sender thread may have held the lock when the parent forked.
        self._lock = threading.Lock()
        if self._owns_http:
            self._http = httpx.Client(timeout=TOKEN_REQUEST_TIMEOUT_SECONDS)

    def token(self, *, refresh: bool = False) -> str:
        """Return a current access token.

        :param refresh: fetch a new token even if the cached one has not expired.
        :returns: the token.
        :rtype: str
        :raises httpx.HTTPError: if Keycloak cannot be reached or refuses.
        """
        with self._lock:
            if refresh or self._token is None or time.monotonic() >= self._expires_at:
                self._fetch()
            if self._token is None:  # pragma: no cover - _fetch sets it or raises
                raise AssertionError
            return self._token

    def _fetch(self) -> None:
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client.client_id,
            "client_secret": self._client.client_secret,
        }
        if self._client.scope:
            data["scope"] = self._client.scope
        response = self._http.post(self._client.token_url, data=data)
        response.raise_for_status()
        payload = response.json()
        lifetime = float(payload.get("expires_in", DEFAULT_TOKEN_LIFETIME_SECONDS))
        self._token = payload["access_token"]
        self._expires_at = time.monotonic() + max(0.0, lifetime - REFRESH_SKEW_SECONDS)

    def close(self) -> None:
        """Close the token endpoint's HTTP client."""
        self._http.close()


class HttpxBearerAuth(httpx.Auth):
    """Bearer auth for the Opik SDK's ``httpx`` client, retried once on a 401.

    The SDK takes the auth once, when it builds its client, so ``tokens`` and
    ``host`` can be replaced afterwards; with no tokens, requests go out without
    credentials. Only requests to the Opik host get the token: the SDK builds its
    client for presigned S3 uploads from the same hooks.
    """

    def __init__(self, tokens: KeycloakTokens | None, *, host: str) -> None:
        """Use the given token cache for requests to one host.

        :param tokens: the shared token cache.
        :param host: the Opik API's host.
        """
        self.tokens = tokens
        self.host = host

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response]:
        """Send with the cached token; after a 401, once more with a new one."""
        if (tokens := self.tokens) is None or request.url.host != self.host:
            yield request
            return
        request.headers["Authorization"] = f"Bearer {tokens.token()}"
        response = yield request
        if response.status_code != httpx.codes.UNAUTHORIZED:
            return
        try:
            _ = request.content
        except httpx.RequestNotRead:
            # A streamed body (the SDK's file uploads) cannot be sent twice.
            return
        # httpx reads and closes the 401 itself before sending the retry; closing
        # it here first makes that read raise StreamClosed.
        request.headers["Authorization"] = f"Bearer {tokens.token(refresh=True)}"
        yield request


class RequestsBearerAuth(AuthBase):
    """Bearer auth for the span exporter's ``requests`` session, retried on a 401.

    The OTLP exporter takes a ``requests.Session`` and only static headers
    otherwise, so this is where an expiring token can be attached.
    """

    def __init__(self, tokens: KeycloakTokens) -> None:
        """Use the given token cache.

        :param tokens: the shared token cache.
        """
        self._tokens = tokens

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        """Attach the cached token and the hook that retries a 401."""
        request.headers["Authorization"] = f"Bearer {self._tokens.token()}"
        request.register_hook("response", self._retry_unauthorized)
        return request

    def _retry_unauthorized(
        self,
        response: requests.Response,
        **kwargs: Any,  # noqa: ANN401
    ) -> requests.Response:
        if response.status_code != requests.codes.unauthorized:
            return response
        # Drain the body so the connection can be reused for the retry.
        _ = response.content
        response.close()
        retry = response.request.copy()
        # copy() shares the hook lists; removing ours from the shared list would
        # skip whatever hook requests runs after it on this response.
        retry.hooks = {
            event: [hook for hook in hooks if hook != self._retry_unauthorized]
            for event, hooks in retry.hooks.items()
        }
        retry.headers["Authorization"] = f"Bearer {self._tokens.token(refresh=True)}"
        retried: requests.Response = response.connection.send(retry, **kwargs)
        retried.history.append(response)
        retried.request = retry
        return retried


_token_caches: "weakref.WeakSet[KeycloakTokens]" = weakref.WeakSet()


def _after_fork_in_child() -> None:
    for tokens in list(_token_caches):
        tokens._after_fork()  # noqa: SLF001


register_at_fork(after_in_child=_after_fork_in_child)
