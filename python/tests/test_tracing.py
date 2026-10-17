from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import httpx
import opik
import pytest
import requests
from ol_llm import tracing
from ol_llm.testing import tracing_disabled
from ol_llm.tracing import (
    PROJECT_ATTRIBUTE,
    HttpxBearerAuth,
    KeycloakClient,
    KeycloakTokens,
    ProjectExporter,
    RequestsBearerAuth,
    TracingConfig,
    configure_tracing,
    tracer_provider,
)
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opik.hooks import HttpxClientHook
from requests.adapters import BaseAdapter

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

KEYCLOAK = KeycloakClient(
    token_url="https://sso.example/token",
    client_id="ol-opik-client",
    client_secret="secret",
    scope="opik",
)
KEYCLOAK_ENV = {
    "OPIK_KEYCLOAK_TOKEN_URL": KEYCLOAK.token_url,
    "OPIK_KEYCLOAK_CLIENT_ID": KEYCLOAK.client_id,
    "OPIK_KEYCLOAK_CLIENT_SECRET": KEYCLOAK.client_secret,
}


def token_server(
    lifetimes: list[float],
) -> tuple[httpx.Client, list[dict[str, str]]]:
    """Return a client for a Keycloak that issues token-1, token-2, ..."""
    requests_seen: list[dict[str, str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        form = dict(httpx.QueryParams(request.content.decode()))
        requests_seen.append(form)
        return httpx.Response(
            200,
            json={
                "access_token": f"token-{len(requests_seen)}",
                "expires_in": lifetimes[len(requests_seen) - 1],
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handle)), requests_seen


def test_from_env_reads_the_sdk_config_and_keycloak() -> None:
    config = TracingConfig.from_env({**KEYCLOAK_ENV, "OPIK_KEYCLOAK_SCOPE": "opik"})

    assert config == TracingConfig(
        enabled=True, keycloak=KEYCLOAK, include_content=True
    )


def test_from_env_without_keycloak_sends_no_credentials() -> None:
    assert TracingConfig.from_env({}).keycloak is None


def test_from_env_follows_opik_track_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPIK_TRACK_DISABLE", "true")

    assert not TracingConfig.from_env({}).enabled


def test_from_env_rejects_a_partial_keycloak_client() -> None:
    env = {**KEYCLOAK_ENV}
    del env["OPIK_KEYCLOAK_CLIENT_SECRET"]

    with pytest.raises(ValueError, match="OPIK_KEYCLOAK_CLIENT_SECRET must be set"):
        TracingConfig.from_env(env)


def test_from_env_rejects_keycloak_with_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPIK_API_KEY", "static")

    with pytest.raises(ValueError, match="OPIK_API_KEY"):
        TracingConfig.from_env(KEYCLOAK_ENV)


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("1", True)])
def test_from_env_reads_trace_content(value: str, expected: bool) -> None:  # noqa: FBT001
    config = TracingConfig.from_env({"OL_LLM_TRACE_CONTENT": value})

    assert config.include_content is expected


def test_from_env_rejects_a_trace_content_value_that_is_not_boolean() -> None:
    with pytest.raises(ValueError, match="OL_LLM_TRACE_CONTENT"):
        TracingConfig.from_env({"OL_LLM_TRACE_CONTENT": "sometimes"})


def test_tokens_are_cached_until_they_near_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1000.0]
    monkeypatch.setattr(tracing.time, "monotonic", lambda: now[0])
    http, seen = token_server([300, 300])
    tokens = KeycloakTokens(KEYCLOAK, http=http)

    assert tokens.token() == "token-1"
    now[0] += 300 - tracing.REFRESH_SKEW_SECONDS - 1
    assert tokens.token() == "token-1"
    now[0] += 1
    assert tokens.token() == "token-2"
    assert seen[0] == {
        "grant_type": "client_credentials",
        "client_id": "ol-opik-client",
        "client_secret": "secret",
        "scope": "opik",
    }


def test_tokens_refresh_on_demand() -> None:
    http, _ = token_server([300, 300])
    tokens = KeycloakTokens(KEYCLOAK, http=http)

    assert tokens.token() == "token-1"
    assert tokens.token(refresh=True) == "token-2"


def test_tokens_raise_when_keycloak_refuses() -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401)))

    with pytest.raises(httpx.HTTPStatusError):
        KeycloakTokens(KEYCLOAK, http=http).token()


def opik_server(statuses: list[int]) -> tuple[Callable[..., httpx.Response], list[str]]:
    """Return a handler answering with the given statuses, recording each token."""
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(statuses[len(seen) - 1])

    return handle, seen


def test_httpx_auth_sends_the_token_and_retries_a_401_once() -> None:
    http, _ = token_server([300, 300])
    handle, seen = opik_server([401, 200])
    client = httpx.Client(
        transport=httpx.MockTransport(handle),
        auth=HttpxBearerAuth(KeycloakTokens(KEYCLOAK, http=http), host="opik.example"),
    )

    response = client.get("https://opik.example/api/v1/private/projects")

    assert response.status_code == 200
    assert seen == ["Bearer token-1", "Bearer token-2"]


def test_httpx_auth_returns_a_second_401() -> None:
    http, _ = token_server([300, 300])
    handle, seen = opik_server([401, 401])
    client = httpx.Client(
        transport=httpx.MockTransport(handle),
        auth=HttpxBearerAuth(KeycloakTokens(KEYCLOAK, http=http), host="opik.example"),
    )

    assert client.get("https://opik.example/api").status_code == 401
    assert len(seen) == 2


class RecordingAdapter(BaseAdapter):
    """A requests transport answering with the given statuses."""

    def __init__(self, statuses: list[int]) -> None:
        super().__init__()
        self.statuses = statuses
        self.seen: list[str] = []

    def send(
        self,
        request: requests.PreparedRequest,
        *_args: object,
        **_kwargs: object,
    ) -> requests.Response:
        self.seen.append(str(request.headers["Authorization"]))
        response = requests.Response()
        response.status_code = self.statuses[len(self.seen) - 1]
        response.request = request
        response.connection = self
        response._content = b""  # noqa: SLF001
        return response

    def close(self) -> None:
        pass


def test_requests_auth_sends_the_token_and_retries_a_401_once() -> None:
    http, _ = token_server([300, 300])
    adapter = RecordingAdapter([401, 200])
    session = requests.Session()
    session.mount("https://", adapter)
    session.auth = RequestsBearerAuth(KeycloakTokens(KEYCLOAK, http=http))

    response = session.post("https://opik.example/api/v1/private/otel/v1/traces")

    assert response.status_code == 200
    assert [r.status_code for r in response.history] == [401]
    assert adapter.seen == ["Bearer token-1", "Bearer token-2"]


def test_requests_auth_returns_a_second_401() -> None:
    http, _ = token_server([300, 300])
    adapter = RecordingAdapter([401, 401])
    session = requests.Session()
    session.mount("https://", adapter)
    session.auth = RequestsBearerAuth(KeycloakTokens(KEYCLOAK, http=http))

    assert session.post("https://opik.example/api").status_code == 401
    assert len(adapter.seen) == 2


OTHER_KEYCLOAK = KeycloakClient(
    token_url="https://other/token", client_id="c", client_secret="s"
)


@pytest.fixture
def hooks(monkeypatch: pytest.MonkeyPatch) -> list[HttpxClientHook]:
    """Record SDK hooks in place of registering them on the real SDK."""
    registered: list[HttpxClientHook] = []
    monkeypatch.setattr(tracing, "add_httpx_client_hook", registered.append)
    monkeypatch.setattr(tracing, "_sdk_auth", None)
    tracing.reset()
    return registered


def sdk_auth(hooks: list[HttpxClientHook]) -> HttpxBearerAuth:
    [hook] = hooks
    auth = hook.update_init_arguments({})["auth"]
    assert isinstance(auth, HttpxBearerAuth)
    return auth


def test_keycloak_hooks_the_sdk_client_once(hooks: list[HttpxClientHook]) -> None:
    configure_tracing(TracingConfig(keycloak=KEYCLOAK))
    first = sdk_auth(hooks).tokens
    configure_tracing(TracingConfig(keycloak=OTHER_KEYCLOAK))

    assert len(hooks) == 1
    assert sdk_auth(hooks).tokens is not first
    assert sdk_auth(hooks).tokens is tracing._state().tokens  # noqa: SLF001


def test_configuration_can_change_until_it_is_used(
    hooks: list[HttpxClientHook],  # noqa: ARG001
) -> None:
    configure_tracing(TracingConfig(keycloak=KEYCLOAK))
    configure_tracing(TracingConfig(keycloak=KEYCLOAK, include_content=False))

    assert not tracing.include_content()


def test_the_same_configuration_again_is_accepted_after_use(
    hooks: list[HttpxClientHook],  # noqa: ARG001
) -> None:
    configure_tracing(TracingConfig(keycloak=KEYCLOAK))
    provider = tracer_provider("mit-learn")

    configure_tracing(TracingConfig(keycloak=KEYCLOAK))

    assert tracer_provider("mit-learn") is provider


def test_a_different_configuration_after_use_is_refused_and_not_kept(
    hooks: list[HttpxClientHook],  # noqa: ARG001
) -> None:
    configure_tracing(TracingConfig(keycloak=KEYCLOAK))
    provider = tracer_provider("mit-learn")

    with pytest.raises(RuntimeError, match="before building agents"):
        configure_tracing(TracingConfig(keycloak=KEYCLOAK, include_content=False))

    assert tracing.include_content()
    assert tracer_provider("mit-learn") is provider


def test_disabled_tracing_builds_no_provider_and_no_hook(
    hooks: list[HttpxClientHook],
) -> None:
    configure_tracing(TracingConfig(enabled=False, keycloak=KEYCLOAK))

    assert tracer_provider("mit-learn") is None
    assert hooks == []


def test_each_app_gets_its_own_provider() -> None:
    first = tracer_provider("mit-learn")

    assert first is tracer_provider("mit-learn")
    assert first is not tracer_provider("learn-ai")


def test_spans_are_dropped_while_opik_tracing_is_off(
    exported: InMemorySpanExporter,
) -> None:
    provider = tracer_provider("mit-learn")
    assert provider is not None
    tracer = provider.get_tracer("test")

    with tracing_disabled(), tracer.start_as_current_span("dropped"):
        pass
    with tracer.start_as_current_span("kept"):
        pass
    tracing.flush()

    assert [span.name for span in exported.get_finished_spans()] == ["kept"]


def test_spans_inside_a_tracked_function_carry_its_trace_id(
    exported: InMemorySpanExporter,
) -> None:
    provider = tracer_provider("mit-learn")
    assert provider is not None
    tracer = provider.get_tracer("test")

    @opik.track(project_name="mit-learn")
    def tracked() -> str:
        with tracer.start_as_current_span("model"):
            pass
        return opik.opik_context.get_current_trace_data().id  # type: ignore[union-attr]

    with opik.record_traces_locally():
        trace_id = tracked()
    tracing.flush()

    [span] = exported.get_finished_spans()
    assert (span.attributes or {})["opik.trace_id"] == trace_id


def test_spans_record_the_project_of_their_trace(
    exported: InMemorySpanExporter,
) -> None:
    provider = tracer_provider("mit-learn")
    assert provider is not None
    tracer = provider.get_tracer("test")

    @opik.track(project_name="ol-data-platform")
    def tracked() -> None:
        with tracer.start_as_current_span("agent"), tracer.start_as_current_span("llm"):
            pass

    with opik.record_traces_locally():
        tracked()
    with tracer.start_as_current_span("untraced"):
        pass
    tracing.flush()

    projects = {
        span.name: (span.attributes or {}).get(PROJECT_ATTRIBUTE)
        for span in exported.get_finished_spans()
    }
    assert projects == {
        "agent": "ol-data-platform",
        "llm": "ol-data-platform",
        "untraced": None,
    }


def test_spans_are_exported_per_project() -> None:
    by_project: dict[str, InMemorySpanExporter] = {}

    def make(project: str) -> InMemorySpanExporter:
        return by_project.setdefault(project, InMemorySpanExporter())

    exporter = ProjectExporter("mit-learn", make)
    provider = tracer_provider("mit-learn")
    assert provider is not None
    tracer = provider.get_tracer("test")
    spans: list[ReadableSpan] = []
    for project in ("ol-data-platform", None):
        with tracer.start_as_current_span("s") as span:
            if project:
                span.set_attribute(PROJECT_ATTRIBUTE, project)
        spans.append(span)  # type: ignore[arg-type]

    assert exporter.export(spans) == SpanExportResult.SUCCESS
    assert {p: len(e.get_finished_spans()) for p, e in by_project.items()} == {
        "ol-data-platform": 1,
        "mit-learn": 1,
    }


class StreamingTransport(httpx.BaseTransport):
    """Streams the request body and the response, as httpx's real transport does.

    MockTransport reads the whole request and response, which would hide both a
    body that cannot be resent and a response closed before httpx reads it.
    """

    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.bodies: list[bytes] = []
        self.tokens: list[str | None] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(b"".join(request.stream))  # type: ignore[arg-type]
        self.tokens.append(request.headers.get("Authorization"))
        return httpx.Response(
            self.statuses[len(self.bodies) - 1], stream=httpx.ByteStream(b"{}")
        )


def test_httpx_auth_retries_a_401_whose_body_is_unread() -> None:
    http, _ = token_server([300, 300])
    transport = StreamingTransport([401, 200])
    client = httpx.Client(
        transport=transport,
        auth=HttpxBearerAuth(KeycloakTokens(KEYCLOAK, http=http), host="opik.example"),
    )

    response = client.post("https://opik.example/api/v1/private/traces", json={})

    assert response.status_code == 200
    assert transport.tokens == ["Bearer token-1", "Bearer token-2"]


def test_httpx_auth_sends_the_token_only_to_the_opik_host() -> None:
    http, seen = token_server([300])
    transport = StreamingTransport([200])
    client = httpx.Client(
        transport=transport,
        auth=HttpxBearerAuth(KeycloakTokens(KEYCLOAK, http=http), host="opik.example"),
    )

    client.put("https://bucket.s3.amazonaws.com/upload?X-Amz-Signature=x", content=b"a")

    assert transport.tokens == [None]
    assert seen == []


def test_httpx_auth_does_not_resend_a_streamed_body() -> None:
    http, _ = token_server([300, 300])
    transport = StreamingTransport([401, 200])
    client = httpx.Client(
        transport=transport,
        auth=HttpxBearerAuth(KeycloakTokens(KEYCLOAK, http=http), host="opik.example"),
    )

    def body() -> Iterator[bytes]:
        yield b"attachment"

    response = client.post("https://opik.example/api/upload", content=body())

    assert response.status_code == 401
    assert transport.bodies == [b"attachment"]


def test_httpx_auth_without_tokens_sends_no_credentials() -> None:
    seen: list[str | None] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200)

    client = httpx.Client(
        transport=httpx.MockTransport(handle),
        auth=HttpxBearerAuth(None, host="opik.example"),
    )

    client.get("https://opik.example/api")

    assert seen == [None]
