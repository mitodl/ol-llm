"""Tracing wiring: the configuration, and the OpenTelemetry pipeline every span uses.

Every call is recorded as OpenTelemetry spans (library spec §8.2 principle 5): the
call span ``ol_llm.traces`` opens, with the §4 attribution as attributes named in
``ol_llm.trace_attributes``, and PydanticAI's, embedding, and memory spans under
it. Each app gets a tracer provider whose span processors send every span to each
configured trace destination (``ol_llm.trace_destinations``), Opik by default.

Configuration is the environment: ``OL_LLM_TRACE_DESTINATIONS`` names the
destinations, ``OL_LLM_TRACE_CONTENT`` says whether traces carry content, and
``OPIK_TRACK_DISABLE`` turns tracing off, as it turns off the Opik SDK. Our Opik
sits behind Keycloak at APISIX, which validates a client-credentials JWT on every
``/api`` request; ``OPIK_KEYCLOAK_TOKEN_URL``, ``OPIK_KEYCLOAK_CLIENT_ID``,
``OPIK_KEYCLOAK_CLIENT_SECRET``, and optionally ``OPIK_KEYCLOAK_SCOPE`` turn that on
for the Opik SDK and the span exporter from one token cache.

``OPIK_TRACK_DISABLE=true`` is the no-op mode for tests and local development: the
library opens no call spans and builds no exporter. It does not turn off Keycloak:
SDK calls the app makes itself (prompt sync, datasets, experiments) still send the
token.

Apps configure tracing at startup, often before a prefork worker pool forks. A
forked child gets new HTTP clients for the token endpoint and the span exporters,
whose pooled sockets the parent still uses. OpenTelemetry restarts its own batch
thread in the child.
"""

import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import Link, SpanKind, TraceState
from opentelemetry.util.types import Attributes

import ol_llm.trace_destinations.opik_destination as _opik
from ol_llm.keycloak import (
    DEFAULT_TOKEN_LIFETIME_SECONDS,
    REFRESH_SKEW_SECONDS,
    TOKEN_REQUEST_TIMEOUT_SECONDS,
    HttpxBearerAuth,
    KeycloakClient,
    KeycloakTokens,
    RequestsBearerAuth,
)
from ol_llm.loops import register_at_fork
from ol_llm.trace_destinations import (
    DEFAULT_DESTINATIONS,
    DestinationSettings,
    TraceDestination,
    build_destinations,
    tracing_disabled,
)
from ol_llm.trace_destinations.interface import parse_bool

__all__ = [
    "DEFAULT_TOKEN_LIFETIME_SECONDS",
    "REFRESH_SKEW_SECONDS",
    "TOKEN_REQUEST_TIMEOUT_SECONDS",
    "ActiveSampler",
    "HttpxBearerAuth",
    "KeycloakClient",
    "KeycloakTokens",
    "RequestsBearerAuth",
    "TracingConfig",
    "active",
    "configure_tracing",
    "destinations",
    "enabled",
    "flush",
    "get_tracing_config",
    "include_content",
    "opik_client",
    "reset",
    "tracer_provider",
]

_ENV_PREFIX = "OPIK_KEYCLOAK"
_CONTENT_ENV = "OL_LLM_TRACE_CONTENT"
_DESTINATIONS_ENV = "OL_LLM_TRACE_DESTINATIONS"


@dataclass(frozen=True, kw_only=True)
class TracingConfig:
    """Whether runs are traced, where to, how they authenticate, what they carry.

    The Opik URL and workspace are not here: the SDK reads them from its own
    environment (``OPIK_URL_OVERRIDE``, ``OPIK_WORKSPACE``) or config file, and the
    Opik destination reads them from the same place so the two cannot disagree.

    :param enabled: false turns tracing off: runs open no call span and export no
        spans. SDK calls outside a run still authenticate with ``keycloak``.
    :param keycloak: the client to authenticate with, whether or not tracing is
        enabled; ``None`` sends no credentials, for an Opik with nothing in front
        of it.
    :param include_content: whether traces carry prompts, completions, and tool
        arguments. On until the data-handling policy says otherwise.
    :param destinations: the trace destinations every span goes to, by name.
    """

    enabled: bool = True
    keycloak: KeycloakClient | None = None
    include_content: bool = True
    destinations: tuple[str, ...] = DEFAULT_DESTINATIONS

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "TracingConfig":
        """Read the configuration from the environment.

        Tracing is off when a named destination's backend turns it off, e.g.
        ``OPIK_TRACK_DISABLE`` for ``opik``.

        :param environ: the environment to read; ``os.environ`` when omitted. The
            SDK's own settings always come from the process environment and the
            SDK's config file, since that is where the SDK reads them.
        :returns: the configuration.
        :rtype: TracingConfig
        :raises ValueError: if the Keycloak variables are partly set, if
            ``OL_LLM_TRACE_CONTENT`` is not a boolean, or if
            ``OL_LLM_TRACE_DESTINATIONS`` names none or one not registered.
        """
        env = os.environ if environ is None else environ
        destinations = _parse_names(env.get(_DESTINATIONS_ENV))
        return cls(
            enabled=not tracing_disabled(destinations, env),
            keycloak=_keycloak_from_env(env),
            include_content=parse_bool(env.get(_CONTENT_ENV, "true"), _CONTENT_ENV),
            destinations=destinations,
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


def _parse_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_DESTINATIONS
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        msg = f"{_DESTINATIONS_ENV} names no trace destination"
        raise ValueError(msg)
    return names


class ActiveSampler(Sampler):
    """Records spans only while every destination has tracing on.

    A destination with a runtime switch, e.g. ``opik.set_tracing_active(False)``,
    turns recording off; a run opens no call span meanwhile, and this keeps its
    model spans from being exported on their own as unattached traces.
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
        if not active():
            return SamplingResult(Decision.DROP)
        return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

    def get_description(self) -> str:
        """Name the sampler."""
        return "ActiveSampler"


@dataclass
class _Tracing:
    config: TracingConfig
    tokens: KeycloakTokens | None
    destinations: tuple[TraceDestination, ...]
    providers: dict[str, TracerProvider] = field(default_factory=dict)
    used: bool = False
    sdk_client_used: bool = False

    def close(self) -> None:
        for provider in self.providers.values():
            provider.shutdown()
        for destination in self.destinations:
            destination.shutdown()
        if self.sdk_client_used:
            _opik.release_sdk_auth(self.tokens)
        if self.tokens is not None:
            self.tokens.close()


_lock = threading.RLock()
_tracing: _Tracing | None = None


def configure_tracing(config: TracingConfig | None = None) -> None:
    """Set up tracing for this process.

    Call it at startup, before anything uses the Opik SDK (the SDK reads its client
    hooks when it first builds its HTTP client) and before building agents (each
    agent keeps the instrumentation it was built with). Without this call, tracing
    reads the environment the first time it is needed.

    :param config: the configuration; read from the environment when omitted.
    :raises ValueError: if the environment does not parse, or a destination is
        not registered.
    :raises RuntimeError: if tracing was already used under a different
        configuration. The configuration in effect stays as it was.
    """
    global _tracing
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
        # Start the new state before closing the old, so a failure keeps the old.
        previous, _tracing = _tracing, _start(config)
        if previous is not None:
            previous.close()


def _start(config: TracingConfig) -> _Tracing:
    tokens = KeycloakTokens(config.keycloak) if config.keycloak else None
    destinations = build_destinations(
        config.destinations, DestinationSettings(environ=os.environ, tokens=tokens)
    )
    return _Tracing(config=config, tokens=tokens, destinations=destinations)


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


def opik_client() -> _opik.OpikClient:
    """Return the Opik SDK's global client, with the configuration settled first.

    For the Opik adapters of the other seams (prompts, evaluation). Settling the
    configuration registers the Keycloak hook, which the SDK reads only when it
    builds its client, so every SDK call outside a traced run gets the same token
    as the runs, with tracing on or off.

    :returns: the client.
    :rtype: opik.Opik
    :raises ValueError: if Keycloak is configured alongside ``OPIK_API_KEY``.
    """
    with _lock:
        state = _state()
        if not state.sdk_client_used:
            _opik.install_sdk_auth(state.tokens)
            state.sdk_client_used = True
    return _opik.global_client()


def enabled() -> bool:
    """Return whether runs are traced.

    :returns: the configured value.
    :rtype: bool
    """
    return _state().config.enabled


def destinations() -> tuple[TraceDestination, ...]:
    """Return the configured trace destinations.

    :returns: the destinations, in configured order.
    :rtype: tuple[TraceDestination, ...]
    """
    return _state().destinations


def active() -> bool:
    """Return whether calls are traced right now.

    That is, tracing is enabled and no destination's runtime switch is off.

    :returns: whether to trace.
    :rtype: bool
    """
    state = _state()
    return state.config.enabled and all(d.active() for d in state.destinations)


def tracer_provider(app: str) -> TracerProvider | None:
    """Return the tracer provider for an app's spans.

    :param app: the app; a span outside any call goes to its project.
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


def _build_provider(app: str, state: _Tracing) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({"service.name": app}),
        sampler=ActiveSampler(),
    )
    for destination in state.destinations:
        provider.add_span_processor(destination.span_processor(app))
    return provider


def include_content() -> bool:
    """Return whether traces carry prompts, completions, and tool arguments.

    :returns: the configured value.
    :rtype: bool
    """
    return _state().config.include_content


def flush(timeout_millis: int = 30_000) -> bool:
    """Send every buffered call trace and span now.

    For short-lived processes (a Celery task, a CLI) that may exit before the
    background senders run. Long-running processes need not call it. Also sends
    what each destination buffers itself, and what the Opik SDK buffered for the
    other seams (scores, experiments) once they used it through ``opik_client``.

    :param timeout_millis: how long to wait for the span exporters.
    :returns: whether every exporter finished within the timeout.
    :rtype: bool
    """
    with _lock:
        state = _tracing
    if state is None:
        return True
    flushed = [d.flush(timeout_millis) for d in state.destinations]
    if state.sdk_client_used:
        _opik.flush_sdk()
    with _lock:
        providers = list(state.providers.values())
    flushed.extend(provider.force_flush(timeout_millis) for provider in providers)
    return all(flushed)


def reset() -> None:
    """Forget the configuration and shut down every provider. For tests."""
    global _tracing  # noqa: PLW0603
    with _lock:
        if _tracing is not None:
            _tracing.close()
        _tracing = None


def _after_fork_in_child() -> None:
    global _lock  # noqa: PLW0603
    _lock = threading.RLock()


register_at_fork(after_in_child=_after_fork_in_child)
