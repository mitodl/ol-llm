"""Tokens for the memory service, from Keycloak.

A learner call carries a token exchanged from the learner's own session token by
standard token exchange (RFC 8693), as the app's own exchange client (learn-ai's
``ol-learn-ai-client``, mit-learn's ``ol-mitlearn-memory-client``). Exchanged
tokens last 300 s and APISIX may forward a session token close to expiry, so
exchange at the start of each request and do not keep the result.

General-memory writes and the platform's forget path use client-credentials
tokens, cached and renewed by the same code that authenticates the Opik exporter
(``ol_llm.keycloak.KeycloakTokens``).

A backend without Keycloak (in-memory, local development) takes neither.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import httpx2

from ol_llm.keycloak import (
    TOKEN_REQUEST_TIMEOUT_SECONDS,
    KeycloakClient,
    KeycloakTokens,
)
from ol_llm.loops import LoopLocal
from ol_llm.memory.errors import (
    MemoryRejected,
    MemoryStoreError,
    MemoryUnauthenticated,
    from_status,
    from_transport,
)

TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"  # noqa: S105
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"  # noqa: S105
MEMORY_AUDIENCE = "ol-memory-client"
"""The memory service's Keycloak client, which exchanged tokens are issued for."""
MEMORY_AUDIENCE_SCOPE = "ol-memory-audience"
"""The client scope an exchange requests; it adds the audience and the ``sub``."""
GENERAL_SCOPE = "ol-memory-general"
"""The client scope that grants general-memory writes to a client-credentials token."""

_CLIENT_ERRORS = frozenset({400, 401, 403})


class TokenExchange(Protocol):
    """Turns a learner's session token into one for the memory service."""

    async def exchange(self, subject_token: str) -> str:
        """Return a memory-service token for the session token's learner.

        :param subject_token: the learner's session access token, as the gateway
            forwards it (APISIX's ``X-Access-Token``).
        :returns: the exchanged token.
        :rtype: str
        """
        ...


@dataclass(frozen=True, kw_only=True)
class KeycloakExchangeClient:
    """The app's Keycloak client for standard token exchange.

    :param token_url: the realm's token endpoint.
    :param client_id: the app's exchange client, one only the app holds.
    :param client_secret: its secret.
    :param audience: the memory service's client.
    :param scope: the scope that puts the audience and ``sub`` on the token.
    """

    token_url: str
    client_id: str
    client_secret: str = field(repr=False)
    audience: str = MEMORY_AUDIENCE
    scope: str = MEMORY_AUDIENCE_SCOPE


class KeycloakTokenExchange:
    """Standard token exchange (RFC 8693) of learner session tokens."""

    def __init__(
        self, client: KeycloakExchangeClient, *, http: httpx2.AsyncClient | None = None
    ) -> None:
        """Exchange as ``client``.

        :param client: the app's exchange client.
        :param http: the HTTP client for the token endpoint; by default one per
            event loop, closed with the loop when ol-llm owns it.
        """
        self._client = client
        self._given = http
        self._http = LoopLocal(
            lambda: httpx2.AsyncClient(timeout=TOKEN_REQUEST_TIMEOUT_SECONDS),
            _close_http,
        )

    async def exchange(self, subject_token: str) -> str:
        """Exchange the learner's session token for a memory-service token.

        :param subject_token: the learner's session access token.
        :returns: the exchanged token.
        :rtype: str
        :raises MemoryUnauthenticated: if Keycloak refuses the session token, e.g.
            because it expired.
        :raises MemoryRejected: if Keycloak refuses the exchange client itself.
        :raises MemoryUnavailable: if Keycloak cannot be reached.
        """
        http = self._given or self._http.get()
        try:
            response = await http.post(
                self._client.token_url,
                data={
                    "grant_type": TOKEN_EXCHANGE_GRANT,
                    "client_id": self._client.client_id,
                    "client_secret": self._client.client_secret,
                    "subject_token": subject_token,
                    "subject_token_type": ACCESS_TOKEN_TYPE,
                    "requested_token_type": ACCESS_TOKEN_TYPE,
                    "audience": self._client.audience,
                    "scope": self._client.scope,
                },
            )
        except httpx2.TransportError as exc:
            raise from_transport(exc, operation="exchange") from exc
        if response.is_success:
            token: str = response.json()["access_token"]
            return token
        raise _exchange_error(response)


async def _close_http(http: httpx2.AsyncClient) -> None:
    await http.aclose()


def _exchange_error(response: httpx2.Response) -> MemoryStoreError:
    try:
        code = str(response.json().get("error", ""))
    except ValueError:
        code = ""
    message = f"token exchange refused: {response.status_code} {code}".rstrip()
    if code == "invalid_client":
        return MemoryRejected(
            message, operation="exchange", status_code=response.status_code
        )
    if response.status_code in _CLIENT_ERRORS:
        return MemoryUnauthenticated(
            message, operation="exchange", status_code=response.status_code
        )
    return from_status(response.status_code, message, operation="exchange")


class ClientCredentialsTokens:
    """Client-credentials tokens for a ``ServiceCredential``, cached and renewed.

    For the general writer, request ``ol-memory-audience ol-memory-general`` as the
    scope unless the client has both as default scopes; the admin client needs
    none.

    It sends with ``httpx``, not ``httpx2`` as the rest of the memory package does,
    because the token cache it wraps is the one the Opik SDK's ``httpx`` hooks use.
    """

    def __init__(
        self, client: KeycloakClient, *, http: httpx.Client | None = None
    ) -> None:
        """Get tokens as ``client``.

        :param client: the Keycloak client.
        :param http: the HTTP client for the token endpoint.
        """
        self._tokens = KeycloakTokens(client, http=http)

    async def token(self, *, refresh: bool = False) -> str:
        """Return a current token, fetched on a worker thread when it is not cached.

        :param refresh: fetch a new token even if the cached one has not expired.
        :returns: the token.
        :rtype: str
        :raises MemoryUnauthenticated: if Keycloak refuses the client.
        :raises MemoryUnavailable: if Keycloak cannot be reached.
        """
        try:
            return await asyncio.to_thread(self._tokens.token, refresh=refresh)
        except httpx.HTTPStatusError as exc:
            message = f"client-credentials grant refused: {exc.response.status_code}"
            raise from_status(
                exc.response.status_code, message, operation="token"
            ) from exc
        except httpx.TransportError as exc:
            raise from_transport(exc, operation="token") from exc
