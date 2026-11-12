"""Choosing and configuring the memory backend, by name rather than by import.

The library registers ``hindsight`` and ``in-memory``. Another package adds a
backend with an entry point in the ``ol_llm.memory_backends`` group, naming a
factory that takes a ``MemoryConfig`` and returns a ``MemoryBackend``, or in
code with ``register_backend``.
"""

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

from ol_llm.memory.in_memory import InMemoryBackend
from ol_llm.memory.interface import DEFAULT_TIMEOUT_SECONDS, MemoryBackend
from ol_llm.memory.service import Memory
from ol_llm.memory.tokens import (
    GENERAL_SCOPE,
    MEMORY_AUDIENCE,
    MEMORY_AUDIENCE_SCOPE,
    ClientCredentialsTokens,
    KeycloakExchangeClient,
    KeycloakTokenExchange,
)
from ol_llm.plugins import NamedFactories
from ol_llm.tracing import KeycloakClient

if TYPE_CHECKING:
    from ol_llm.memory.hindsight import HindsightBackend

ENTRY_POINT_GROUP = "ol_llm.memory_backends"
# What the memory extra installs that the Hindsight backend imports directly.
_MEMORY_EXTRA_MODULES = frozenset({"hindsight_client_api", "aiohttp"})
_ENV_PREFIX = "OL_LLM_MEMORY"

DEFAULT_GENERAL_SCOPE = f"{MEMORY_AUDIENCE_SCOPE} {GENERAL_SCOPE}"


@dataclass(frozen=True, kw_only=True)
class MemoryConfig:
    """Which memory backend an app uses, and the Keycloak clients it calls as.

    :param backend: the registered backend name, e.g. ``hindsight`` or
        ``in-memory``.
    :param app: the app's name as the service knows it, e.g. ``learn-ai``.
    :param url: the service's URL, for a backend that is a service.
    :param timeout_seconds: bound on each request to the service.
    :param exchange: the app's token-exchange client, for learner calls. ``None``
        hands session tokens to the backend as they are.
    :param general_client: the client holding the general-memory grant, for
        general writes. Request ``ol-memory-audience ol-memory-general`` as its
        scope unless the client has both by default.
    :param admin_client: the platform admin client, for forget.
    """

    backend: str
    app: str
    url: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    exchange: KeycloakExchangeClient | None = None
    general_client: KeycloakClient | None = None
    admin_client: KeycloakClient | None = None

    def __post_init__(self) -> None:
        """Reject an empty backend or app and a timeout that is not positive.

        :raises ValueError: if any is invalid.
        """
        if not self.backend or not self.app:
            msg = "MemoryConfig.backend and MemoryConfig.app must not be empty"
            raise ValueError(msg)
        if not (math.isfinite(self.timeout_seconds) and self.timeout_seconds > 0):
            msg = "MemoryConfig.timeout_seconds must be a positive number"
            raise ValueError(msg)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "MemoryConfig":
        """Read the configuration from ``OL_LLM_MEMORY_*`` environment variables.

        ``OL_LLM_MEMORY_BACKEND`` and ``OL_LLM_MEMORY_APP`` are required;
        ``OL_LLM_MEMORY_URL`` and ``OL_LLM_MEMORY_TIMEOUT_SECONDS`` are optional.
        Each Keycloak client is set by a ``CLIENT_ID`` and ``CLIENT_SECRET`` pair,
        under ``OL_LLM_MEMORY_EXCHANGE_``, ``OL_LLM_MEMORY_GENERAL_`` and
        ``OL_LLM_MEMORY_ADMIN_``, and all of them use ``OL_LLM_MEMORY_TOKEN_URL``.
        ``OL_LLM_MEMORY_AUDIENCE``, ``OL_LLM_MEMORY_AUDIENCE_SCOPE`` and
        ``OL_LLM_MEMORY_GENERAL_SCOPE`` override the deployed service's names.

        :param environ: the environment to read; ``os.environ`` when omitted.
        :returns: the configuration.
        :rtype: MemoryConfig
        :raises ValueError: if a required variable is unset, a client's id or
            secret is set without the other, a client is set without the token
            URL, or the timeout does not parse.
        """
        env = os.environ if environ is None else environ
        missing = [
            f"{_ENV_PREFIX}_{name}"
            for name in ("BACKEND", "APP")
            if not env.get(f"{_ENV_PREFIX}_{name}")
        ]
        if missing:
            msg = f"{' and '.join(missing)} must be set"
            raise ValueError(msg)
        token_url = env.get(f"{_ENV_PREFIX}_TOKEN_URL") or None
        exchange = _client_from_env(env, "EXCHANGE", token_url)
        general = _client_from_env(env, "GENERAL", token_url)
        admin = _client_from_env(env, "ADMIN", token_url)
        timeout_name = f"{_ENV_PREFIX}_TIMEOUT_SECONDS"
        try:
            timeout = float(env.get(timeout_name, DEFAULT_TIMEOUT_SECONDS))
        except ValueError as exc:
            msg = f"{timeout_name}={env[timeout_name]!r} is not a number"
            raise ValueError(msg) from exc
        return cls(
            backend=env[f"{_ENV_PREFIX}_BACKEND"],
            app=env[f"{_ENV_PREFIX}_APP"],
            url=env.get(f"{_ENV_PREFIX}_URL") or None,
            timeout_seconds=timeout,
            exchange=None
            if exchange is None
            else KeycloakExchangeClient(
                token_url=exchange.token_url,
                client_id=exchange.client_id,
                client_secret=exchange.client_secret,
                audience=env.get(f"{_ENV_PREFIX}_AUDIENCE", MEMORY_AUDIENCE),
                scope=env.get(f"{_ENV_PREFIX}_AUDIENCE_SCOPE", MEMORY_AUDIENCE_SCOPE),
            ),
            general_client=None
            if general is None
            else KeycloakClient(
                token_url=general.token_url,
                client_id=general.client_id,
                client_secret=general.client_secret,
                scope=env.get(f"{_ENV_PREFIX}_GENERAL_SCOPE", DEFAULT_GENERAL_SCOPE),
            ),
            admin_client=admin,
        )


def _client_from_env(
    env: Mapping[str, str], role: str, token_url: str | None
) -> KeycloakClient | None:
    prefix = f"{_ENV_PREFIX}_{role}"
    client_id = env.get(f"{prefix}_CLIENT_ID")
    secret = env.get(f"{prefix}_CLIENT_SECRET")
    if not client_id and not secret:
        return None
    if not client_id or not secret:
        msg = f"{prefix}_CLIENT_ID and {prefix}_CLIENT_SECRET must be set together"
        raise ValueError(msg)
    if token_url is None:
        msg = f"{_ENV_PREFIX}_TOKEN_URL must be set with {prefix}_CLIENT_ID"
        raise ValueError(msg)
    return KeycloakClient(
        token_url=token_url, client_id=client_id, client_secret=secret
    )


BackendFactory = Callable[[MemoryConfig], MemoryBackend]


def hindsight_backend() -> "type[HindsightBackend]":
    """Import the Hindsight backend, which needs the ``memory`` extra.

    :returns: the backend class.
    :rtype: type[HindsightBackend]
    :raises ImportError: naming the extra, if it is not installed.
    """
    try:
        module = import_module("ol_llm.memory.hindsight")
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] not in _MEMORY_EXTRA_MODULES:
            raise
        msg = f"the hindsight memory backend needs the ol-llm[memory] extra: {exc}"
        raise ImportError(msg) from exc
    backend: type[HindsightBackend] = module.HindsightBackend
    return backend


def _hindsight(config: MemoryConfig) -> MemoryBackend:
    if config.url is None:
        msg = "the hindsight memory backend needs a url (OL_LLM_MEMORY_URL)"
        raise ValueError(msg)
    return hindsight_backend()(
        config.url, app=config.app, timeout_seconds=config.timeout_seconds
    )


def _in_memory(config: MemoryConfig) -> MemoryBackend:
    return InMemoryBackend(config.app)


_backends: NamedFactories[BackendFactory] = NamedFactories(
    group=ENTRY_POINT_GROUP,
    kind="memory backend",
    builtins={"hindsight": _hindsight, "in-memory": _in_memory},
)


def register_backend(name: str, factory: BackendFactory) -> None:
    """Make a backend available by name, in place of any registered before.

    :param name: what ``MemoryConfig.backend`` names it as.
    :param factory: builds the backend from the configuration.
    """
    _backends.register(name, factory)


def backend_factory(name: str) -> BackendFactory:
    """Return the factory registered under ``name``, loading entry points if needed.

    :param name: the backend name.
    :returns: the factory.
    :rtype: BackendFactory
    :raises ValueError: if no backend has that name.
    """
    return _backends.get(name)


def build_memory(config: MemoryConfig | None = None) -> Memory:
    """Build the app's memory client from configuration.

    Build one at startup and share it; ``Memory.learner`` is what each request
    calls.

    :param config: the configuration; read from the environment when omitted.
    :returns: the client.
    :rtype: Memory
    :raises ValueError: if the configuration is incomplete or names no backend.
    """
    config = config if config is not None else MemoryConfig.from_env()
    backend = backend_factory(config.backend)(config)
    return Memory(
        backend,
        exchange=None
        if config.exchange is None
        else KeycloakTokenExchange(config.exchange),
        general_tokens=None
        if config.general_client is None
        else ClientCredentialsTokens(config.general_client),
        admin_tokens=None
        if config.admin_client is None
        else ClientCredentialsTokens(config.admin_client),
    )
