import asyncio
from collections.abc import Awaitable
from typing import TypeVar
from urllib.parse import parse_qs

import httpx
import httpx2
import pytest
from ol_llm.memory import (
    ClientCredentialsTokens,
    KeycloakExchangeClient,
    KeycloakTokenExchange,
    MemoryRejected,
    MemoryTimeout,
    MemoryUnauthenticated,
    MemoryUnavailable,
)
from ol_llm.tracing import KeycloakClient

_T = TypeVar("_T")
# The exchange sends with httpx2; client-credentials tokens with httpx.
Response = httpx.Response | httpx2.Response

TOKEN_URL = "https://sso.example/realms/olapps/protocol/openid-connect/token"  # noqa: S105
EXCHANGE_CLIENT = KeycloakExchangeClient(
    token_url=TOKEN_URL, client_id="ol-learn-ai-client", client_secret="s3cret"
)


def run(coroutine: Awaitable[_T]) -> _T:
    async def wait() -> _T:
        return await coroutine

    return asyncio.run(wait())


class FakeKeycloak:
    """The realm's token endpoint: records each form and answers in turn."""

    def __init__(self, *responses: Response) -> None:
        self.responses = list(responses)
        self.forms: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request | httpx2.Request) -> Response:
        assert str(request.url) == TOKEN_URL
        form = parse_qs(request.content.decode())
        self.forms.append({key: value for key, [value] in form.items()})
        return self.responses.pop(0)


def exchange(keycloak: FakeKeycloak, subject: str = "session-token") -> str:
    async def go() -> str:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(keycloak)) as http:
            return await KeycloakTokenExchange(EXCHANGE_CLIENT, http=http).exchange(
                subject
            )

    return run(go())


def test_exchange_sends_the_standard_token_exchange_form() -> None:
    keycloak = FakeKeycloak(httpx2.Response(200, json={"access_token": "memory-token"}))

    assert exchange(keycloak) == "memory-token"
    assert keycloak.forms == [
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": "ol-learn-ai-client",
            "client_secret": "s3cret",
            "subject_token": "session-token",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "audience": "ol-memory-client",
            "scope": "ol-memory-audience",
        }
    ]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            httpx2.Response(400, json={"error": "invalid_token"}),
            MemoryUnauthenticated,
        ),
        (
            httpx2.Response(
                400,
                json={
                    "error": "invalid_request",
                    "error_description": "Client is not within the token audience",
                },
            ),
            MemoryUnauthenticated,
        ),
        (httpx2.Response(401, json={"error": "invalid_client"}), MemoryRejected),
        (httpx2.Response(503, text="maintenance"), MemoryUnavailable),
        (httpx2.Response(418, text="teapot"), MemoryRejected),
    ],
)
def test_a_refused_exchange_raises_by_class(
    response: httpx2.Response, error: type[Exception]
) -> None:
    with pytest.raises(error) as raised:
        exchange(FakeKeycloak(response))
    assert type(raised.value) is error
    assert "session-token" not in str(raised.value)


def test_an_exchange_that_times_out_is_retryable() -> None:
    def hang(request: httpx2.Request) -> httpx2.Response:
        msg = "slow"
        raise httpx2.ReadTimeout(msg, request=request)

    async def go() -> str:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(hang)) as http:
            return await KeycloakTokenExchange(EXCHANGE_CLIENT, http=http).exchange("x")

    with pytest.raises(MemoryTimeout) as raised:
        run(go())
    assert raised.value.retryable
    assert raised.value.operation == "exchange"


def test_the_exchange_client_keeps_its_secret_out_of_repr() -> None:
    assert "s3cret" not in repr(EXCHANGE_CLIENT)


def test_client_credentials_tokens_are_cached_and_refreshed() -> None:
    keycloak = FakeKeycloak(
        httpx.Response(200, json={"access_token": "first", "expires_in": 300}),
        httpx.Response(200, json={"access_token": "second", "expires_in": 300}),
    )
    tokens = ClientCredentialsTokens(
        KeycloakClient(
            token_url=TOKEN_URL,
            client_id="catalog-indexer",
            client_secret="general-secret",
            scope="ol-memory-audience ol-memory-general",
        ),
        http=httpx.Client(transport=httpx.MockTransport(keycloak)),
    )

    assert run(tokens.token()) == "first"
    assert run(tokens.token()) == "first"
    assert run(tokens.token(refresh=True)) == "second"
    assert [form["grant_type"] for form in keycloak.forms] == [
        "client_credentials",
        "client_credentials",
    ]
    assert keycloak.forms[0]["scope"] == "ol-memory-audience ol-memory-general"


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            httpx.Response(401, json={"error": "unauthorized_client"}),
            MemoryUnauthenticated,
        ),
        (httpx.Response(502), MemoryUnavailable),
    ],
)
def test_a_refused_client_credentials_grant_raises_by_class(
    response: httpx.Response, error: type[Exception]
) -> None:
    tokens = ClientCredentialsTokens(
        KeycloakClient(
            token_url=TOKEN_URL, client_id="ol-memory-admin-client", client_secret="s"
        ),
        http=httpx.Client(transport=httpx.MockTransport(FakeKeycloak(response))),
    )
    with pytest.raises(error):
        run(tokens.token())


def test_an_unreachable_keycloak_for_client_credentials_is_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        msg = "refused"
        raise httpx.ConnectError(msg, request=request)

    tokens = ClientCredentialsTokens(
        KeycloakClient(
            token_url=TOKEN_URL, client_id="ol-memory-admin-client", client_secret="s"
        ),
        http=httpx.Client(transport=httpx.MockTransport(refuse)),
    )
    with pytest.raises(MemoryUnavailable) as raised:
        run(tokens.token())
    assert raised.value.operation == "token"
