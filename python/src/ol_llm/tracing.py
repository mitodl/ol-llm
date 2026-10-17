"""Opik tracing wiring: SDK authentication and the OTel pipeline for model spans.

Every run has two halves (spec §2.2). The root trace comes from the Opik SDK
(``@opik.track``), which sends over the SDK's own ``httpx`` client. The model spans
come from PydanticAI's OpenTelemetry instrumentation, which this module exports over
OTLP/HTTP to Opik's ``/v1/private/otel`` endpoint. ``OpikSpanProcessor`` stamps
those spans with the surrounding root trace's ids so they land inside it.

Configuration is the environment the SDK already reads, so the SDK and the
exporter always agree on the instance: ``OPIK_URL_OVERRIDE``, ``OPIK_WORKSPACE``,
and ``OPIK_TRACK_DISABLE``. Our Opik sits behind Keycloak at APISIX, which validates
a client-credentials JWT on every ``/api`` request; ``OPIK_KEYCLOAK_TOKEN_URL``,
``OPIK_KEYCLOAK_CLIENT_ID``, ``OPIK_KEYCLOAK_CLIENT_SECRET``, and optionally
``OPIK_KEYCLOAK_SCOPE`` turn that on for both halves from one token cache.

``OPIK_TRACK_DISABLE=true`` is the no-op mode for tests and local development: the
SDK skips ``@track`` and the library builds no exporter.

Spans are filed under the Opik project of the trace they belong to. That is the
app's project for a run the library opened, and the caller's project for a run
inside the caller's own ``@opik.track``, where Opik puts the run's span too.
"""

import os
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import opik
import requests
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import (
    ReadableSpan,
    Span,
    SpanProcessor,
    TracerProvider,
)
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import (
    INVALID_SPAN,
    Link,
    SpanKind,
    TraceState,
    get_current_span,
)
from opentelemetry.util.types import Attributes
from opik import opik_context
from opik.config import OpikConfig
from opik.hooks import HttpxClientHook, add_httpx_client_hook
from opik.integrations.otel import OpikSpanProcessor
from requests.auth import AuthBase

_ENV_PREFIX = "OPIK_KEYCLOAK"
_CONTENT_ENV = "OL_LLM_TRACE_CONTENT"

# Renew this long before the stated expiry, so a token is not used as it expires.
REFRESH_SKEW_SECONDS = 30
# A hung Keycloak must not wedge the SDK's sender threads or the span exporter.
TOKEN_REQUEST_TIMEOUT_SECONDS = 10
# Used when the token response carries no expires_in.
DEFAULT_TOKEN_LIFETIME_SECONDS = 60

OTLP_TRACES_PATH = "/v1/private/otel/v1/traces"
"""Opik's OTLP/HTTP traces endpoint, relative to the SDK's API base URL."""

PROJECT_HEADER = "projectName"
"""Opik files OTLP spans under the project this header names, ``Default Project``
when it is absent, whatever trace the spans attach to."""

WORKSPACE_HEADER = "Comet-Workspace"

PROJECT_ATTRIBUTE = "ol_llm.opik_project"
"""The span attribute that carries the project of the trace a span belongs to."""


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


@dataclass(frozen=True, kw_only=True)
class TracingConfig:
    """Whether runs are traced, how they authenticate, and what they carry.

    The Opik URL and workspace are not here: the SDK reads them from its own
    environment (``OPIK_URL_OVERRIDE``, ``OPIK_WORKSPACE``) or config file, and the
    span exporter reads them from the same place so the two cannot disagree.

    :param enabled: false turns tracing off: runs open no root trace and export no
        spans.
    :param keycloak: the client to authenticate with; ``None`` sends no
        credentials, for an Opik with nothing in front of it.
    :param include_content: whether traces carry prompts, completions, and tool
        arguments. On until the data-handling policy says otherwise.
    """

    enabled: bool = True
    keycloak: KeycloakClient | None = None
    include_content: bool = True

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "TracingConfig":
        """Read the configuration the Opik SDK reads, plus the Keycloak client.

        :param environ: the environment to read; ``os.environ`` when omitted. The
            SDK's own settings always come from the process environment and the
            SDK's config file, since that is where the SDK reads them.
        :returns: the configuration.
        :rtype: TracingConfig
        :raises ValueError: if the Keycloak variables are partly set, if they are
            set alongside ``OPIK_API_KEY``, or if ``OL_LLM_TRACE_CONTENT`` is not a
            boolean.
        """
        env = os.environ if environ is None else environ
        opik_config = OpikConfig()
        keycloak = _keycloak_from_env(env)
        if keycloak is not None and opik_config.api_key:
            # The SDK would send the key as a static Authorization header in place
            # of the refreshed token.
            msg = f"Set either OPIK_API_KEY or the {_ENV_PREFIX}_* variables, not both"
            raise ValueError(msg)
        return cls(
            enabled=not opik_config.track_disable,
            keycloak=keycloak,
            include_content=_parse_bool(env.get(_CONTENT_ENV, "true"), _CONTENT_ENV),
        )


def _keycloak_from_env(env: Mapping[str, str]) -> KeycloakClient | None:
    required = ("TOKEN_URL", "CLIENT_ID", "CLIENT_SECRET")
    values = {name: env.get(f"{_ENV_PREFIX}_{name}") for name in required}
    if not any(values.values()):
        return None
    if missing := [f"{_ENV_PREFIX}_{name}" for name, v in values.items() if not v]:
        msg = f"{', '.join(missing)} must be set with the other {_ENV_PREFIX}_* values"
        raise ValueError(msg)
    return KeycloakClient(
        token_url=env[f"{_ENV_PREFIX}_TOKEN_URL"],
        client_id=env[f"{_ENV_PREFIX}_CLIENT_ID"],
        client_secret=env[f"{_ENV_PREFIX}_CLIENT_SECRET"],
        scope=env.get(f"{_ENV_PREFIX}_SCOPE") or None,
    )


def _parse_bool(value: str, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    msg = f"{name} must be true or false, not {value!r}"
    raise ValueError(msg)


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
        self._http = http or httpx.Client(timeout=TOKEN_REQUEST_TIMEOUT_SECONDS)
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0

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

    The SDK takes the auth once, when it builds its client, so ``tokens`` can be
    replaced afterwards; with none, requests go out without credentials.
    """

    def __init__(self, tokens: KeycloakTokens | None) -> None:
        """Use the given token cache.

        :param tokens: the shared token cache.
        """
        self.tokens = tokens

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response]:
        """Send with the cached token; after a 401, once more with a new one."""
        if (tokens := self.tokens) is None:
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
        response.close()
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


class OpikActiveSampler(Sampler):
    """Records spans only while Opik tracing is active.

    The SDK skips ``@track`` when tracing is off at runtime
    (``opik.set_tracing_active(False)``); this keeps model spans from being exported
    on their own as unattached traces in the meantime.
    """

    def should_sample(  # noqa: PLR0913, PLR0917
        self,
        parent_context: Context | None,  # noqa: ARG002
        trace_id: int,  # noqa: ARG002
        name: str,  # noqa: ARG002
        kind: SpanKind | None = None,  # noqa: ARG002
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,  # noqa: ARG002
        trace_state: TraceState | None = None,  # noqa: ARG002
    ) -> SamplingResult:
        """Sample every span while tracing is active, none otherwise."""
        if not opik.is_tracing_active():
            return SamplingResult(Decision.DROP)
        return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

    def get_description(self) -> str:
        """Name the sampler."""
        return "OpikActiveSampler"


class ProjectStampProcessor(SpanProcessor):
    """Records on each span the Opik project of the trace it belongs to.

    A span inherits its parent's project; a span with none takes the project of
    the current ``@opik.track`` trace. Spans outside any trace get nothing, and
    the exporter files them under the app's project.
    """

    def __init__(self, default_project: str) -> None:
        """Use the SDK's default project for a trace that names none.

        :param default_project: where Opik files a trace with no project.
        """
        self._default_project = default_project

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Copy the project from the parent span or the current trace."""
        parent = get_current_span(parent_context)
        if parent is not INVALID_SPAN:
            attributes = getattr(parent, "attributes", None) or {}
            if (project := attributes.get(PROJECT_ATTRIBUTE)) is not None:
                span.set_attribute(PROJECT_ATTRIBUTE, project)
                return
        trace = opik_context.get_current_trace_data()
        if trace is not None:
            span.set_attribute(
                PROJECT_ATTRIBUTE, trace.project_name or self._default_project
            )


class ProjectExporter(SpanExporter):
    """Exports each span to the Opik project recorded on it, else the app's.

    Opik files OTLP spans under the project named on the export request, whatever
    trace they attach to, so each project's spans go in their own request.
    """

    def __init__(self, app: str, exporter: Callable[[str], SpanExporter]) -> None:
        """Build one exporter per project with ``exporter(project)``.

        :param app: the project for spans that record none.
        :param exporter: makes the exporter for a project.
        """
        self._app = app
        self._make = exporter
        self._lock = threading.Lock()
        self._exporters: dict[str, SpanExporter] = {}

    def _for(self, project: str) -> SpanExporter:
        with self._lock:
            if (exporter := self._exporters.get(project)) is None:
                exporter = self._exporters[project] = self._make(project)
            return exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Send each project's spans in one request."""
        by_project: dict[str, list[ReadableSpan]] = {}
        for span in spans:
            project = (span.attributes or {}).get(PROJECT_ATTRIBUTE, self._app)
            by_project.setdefault(str(project), []).append(span)
        results = [self._for(p).export(group) for p, group in by_project.items()]
        if all(result == SpanExportResult.SUCCESS for result in results):
            return SpanExportResult.SUCCESS
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shut down every project's exporter."""
        with self._lock:
            exporters = list(self._exporters.values())
        for exporter in exporters:
            exporter.shutdown()


@dataclass
class _Tracing:
    config: TracingConfig
    url: str
    workspace: str
    default_project: str
    tokens: KeycloakTokens | None
    providers: dict[str, TracerProvider] = field(default_factory=dict)
    used: bool = False

    def close(self) -> None:
        for provider in self.providers.values():
            provider.shutdown()
        if self.tokens is not None:
            self.tokens.close()


_lock = threading.RLock()
_tracing: _Tracing | None = None
# Registered on the SDK at most once: the SDK reads its hooks only when it builds
# its client, so a later configuration swaps the tokens this auth uses.
_sdk_auth: HttpxBearerAuth | None = None


def configure_tracing(config: TracingConfig | None = None) -> None:
    """Set up tracing for this process.

    Call it at startup, before anything uses the Opik SDK (the SDK reads its client
    hooks when it first builds its HTTP client) and before building agents (each
    agent keeps the instrumentation it was built with). Without this call, tracing
    reads the environment the first time it is needed.

    :param config: the configuration; read from the environment when omitted.
    :raises ValueError: if the environment does not parse.
    :raises RuntimeError: if tracing was already used under a different
        configuration. The configuration in effect stays as it was.
    """
    global _tracing  # noqa: PLW0603
    config = config if config is not None else TracingConfig.from_env()
    with _lock:
        if _tracing is not None:
            if _tracing.config == config:
                return
            if _tracing.used:
                msg = (
                    "Tracing was already used with a different configuration; "
                    "call configure_tracing before building agents or running them"
                )
                raise RuntimeError(msg)
            _tracing.close()
        _tracing = _start(config)


def _start(config: TracingConfig) -> _Tracing:
    global _sdk_auth  # noqa: PLW0603
    opik_config = OpikConfig()
    tokens = KeycloakTokens(config.keycloak) if config.keycloak else None
    sdk_tokens = tokens if config.enabled else None
    if _sdk_auth is not None:
        _sdk_auth.tokens = sdk_tokens
    elif sdk_tokens is not None:
        _sdk_auth = HttpxBearerAuth(sdk_tokens)
        add_httpx_client_hook(
            HttpxClientHook(
                client_modifier=None, client_init_arguments={"auth": _sdk_auth}
            )
        )
    return _Tracing(
        config=config,
        url=opik_config.url_override,
        workspace=opik_config.workspace,
        default_project=opik_config.project_name,
        tokens=tokens,
    )


def _state() -> _Tracing:
    """Return the tracing state, marking the configuration as used."""
    global _tracing  # noqa: PLW0603
    with _lock:
        if _tracing is None:
            _tracing = _start(TracingConfig.from_env())
        _tracing.used = True
        return _tracing


def get_tracing_config() -> TracingConfig:
    """Return the tracing configuration, reading the environment on first use.

    :returns: the configuration.
    :rtype: TracingConfig
    :raises ValueError: if nothing was configured and the environment does not
        parse.
    """
    return _state().config


def enabled() -> bool:
    """Return whether runs are traced.

    :returns: the configured value.
    :rtype: bool
    """
    return _state().config.enabled


def tracer_provider(app: str) -> TracerProvider | None:
    """Return the tracer provider for an app's model spans.

    :param app: the app; spans outside any Opik trace go to its project.
    :returns: the provider, or ``None`` when tracing is off.
    :rtype: TracerProvider | None
    """
    with _lock:
        state = _state()
        if not state.config.enabled:
            return None
        if (provider := state.providers.get(app)) is None:
            provider = state.providers[app] = _build_provider(app, state)
        return provider


def _exporter(app: str, state: _Tracing) -> SpanExporter:
    def otlp(project: str) -> SpanExporter:
        session = requests.Session()
        if state.tokens is not None:
            session.auth = RequestsBearerAuth(state.tokens)
        return OTLPSpanExporter(
            endpoint=state.url.rstrip("/") + OTLP_TRACES_PATH,
            headers={PROJECT_HEADER: project, WORKSPACE_HEADER: state.workspace},
            session=session,
        )

    return ProjectExporter(app, otlp)


def _build_provider(app: str, state: _Tracing) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({"service.name": app}),
        sampler=OpikActiveSampler(),
    )
    provider.add_span_processor(OpikSpanProcessor())
    provider.add_span_processor(ProjectStampProcessor(state.default_project))
    provider.add_span_processor(BatchSpanProcessor(_exporter(app, state)))
    return provider


def include_content() -> bool:
    """Return whether traces carry prompts, completions, and tool arguments.

    :returns: the configured value.
    :rtype: bool
    """
    return _state().config.include_content


def flush(timeout_millis: int = 30_000) -> bool:
    """Send every buffered root trace and span now.

    For short-lived processes (a Celery task, a CLI) that may exit before the
    background senders run. Long-running processes need not call it.

    :param timeout_millis: how long to wait for the span exporters.
    :returns: whether every exporter finished within the timeout.
    :rtype: bool
    """
    opik.flush_tracker()
    with _lock:
        providers = list(_tracing.providers.values()) if _tracing else []
    return all(provider.force_flush(timeout_millis) for provider in providers)


def reset() -> None:
    """Forget the configuration and shut down every provider. For tests."""
    global _tracing  # noqa: PLW0603
    with _lock:
        if _tracing is not None:
            _tracing.close()
        if _sdk_auth is not None:
            _sdk_auth.tokens = None
        _tracing = None
