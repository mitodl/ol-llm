"""Deployment configuration: the environment name and how each provider is reached.

The library reads ``OL_LLM_*`` environment variables by default. A host that keeps
its configuration elsewhere (Django settings, a Dagster resource) builds a
``Settings`` itself and passes it to ``configure`` at startup.

Provider credentials are not read here. Each provider SDK reads its own variables
(``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``AZURE_OPENAI_ENDPOINT``,
``GOOGLE_API_KEY``, ``MISTRAL_API_KEY``, ``TOGETHER_API_KEY``), and Bedrock and
Vertex use the default credential chain, which covers IRSA and workload identity.
"""

import inspect
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

_ENV_PREFIX = "OL_LLM"

GATEWAY_ROUTE_FAMILIES = frozenset({"openai", "anthropic"})
"""The API shapes a gateway route can speak. OpenAI-format routes serve the
``openai``, ``openai-chat``, and ``openai-responses`` providers; Anthropic-format
routes serve ``anthropic``."""


@dataclass(frozen=True, kw_only=True)
class GatewayRoute:
    """A gateway route standing in for a provider's API.

    :param base_url: the route's base URL, used where the SDK would use the
        vendor's.
    :param api_key: the application's credential for the route.
    """

    base_url: str
    api_key: str = field(repr=False)


@dataclass(frozen=True, kw_only=True)
class Settings:
    """How this deployment calls models.

    :param environment: the deployment environment, e.g. ``ci``, ``qa``,
        ``production``. Goes on every trace.
    :param gateway_routes: route family (``openai`` or ``anthropic``) to the
        gateway route that replaces the vendor API. A family with no route is
        called directly.
    :param azure_ad_token_provider: a callable returning an Entra ID token for
        Azure OpenAI, for federated credentials in place of an API key. Only
        settable in code, e.g. from ``azure.identity.get_bearer_token_provider``.
        It must be synchronous: every event loop's Azure client shares it, and an
        ``azure.identity.aio`` credential binds its session and locks to the first
        loop that awaits it.
    """

    environment: str
    gateway_routes: Mapping[str, GatewayRoute] = field(default_factory=dict)
    azure_ad_token_provider: Callable[[], str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Reject an empty environment and routes for unknown families.

        :raises ValueError: if either is invalid, or the Azure token provider is
            asynchronous.
        """
        if inspect.iscoroutinefunction(self.azure_ad_token_provider):
            msg = (
                "Settings.azure_ad_token_provider must be synchronous; use "
                "azure.identity.get_bearer_token_provider, not azure.identity.aio"
            )
            raise ValueError(msg)
        if not self.environment:
            msg = "Settings.environment must not be empty"
            raise ValueError(msg)
        if unknown := set(self.gateway_routes) - GATEWAY_ROUTE_FAMILIES:
            msg = (
                f"No gateway route family named {sorted(unknown)}; "
                f"expected one of {sorted(GATEWAY_ROUTE_FAMILIES)}"
            )
            raise ValueError(msg)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Read settings from ``OL_LLM_*`` environment variables.

        ``OL_LLM_ENVIRONMENT`` is required. A gateway route for a family is set by
        ``OL_LLM_GATEWAY_<FAMILY>_BASE_URL`` and ``OL_LLM_GATEWAY_<FAMILY>_API_KEY``
        together.

        :param environ: the environment to read; ``os.environ`` when omitted.
        :returns: the settings.
        :rtype: Settings
        :raises ValueError: if ``OL_LLM_ENVIRONMENT`` is unset, or a route has a base
            URL without a key or the reverse.
        """
        env = os.environ if environ is None else environ
        if not (environment := env.get(f"{_ENV_PREFIX}_ENVIRONMENT")):
            msg = f"{_ENV_PREFIX}_ENVIRONMENT must be set"
            raise ValueError(msg)
        routes: dict[str, GatewayRoute] = {}
        for family in sorted(GATEWAY_ROUTE_FAMILIES):
            prefix = f"{_ENV_PREFIX}_GATEWAY_{family.upper()}"
            base_url = env.get(f"{prefix}_BASE_URL")
            api_key = env.get(f"{prefix}_API_KEY")
            if base_url is None and api_key is None:
                continue
            if not base_url or not api_key:
                msg = f"{prefix}_BASE_URL and {prefix}_API_KEY must be set together"
                raise ValueError(msg)
            routes[family] = GatewayRoute(base_url=base_url, api_key=api_key)
        return cls(environment=environment, gateway_routes=routes)


_lock = threading.Lock()
_settings: Settings | None = None


def configure(settings: Settings) -> None:
    """Set the settings for this process, in place of reading the environment.

    Call it once at startup, before the first model is built.

    :param settings: the settings to use.
    """
    global _settings  # noqa: PLW0603
    with _lock:
        _settings = settings


def get_settings() -> Settings:
    """Return the configured settings, reading the environment on first use.

    :returns: the settings.
    :rtype: Settings
    :raises ValueError: if nothing was configured and the environment is
        incomplete.
    """
    global _settings  # noqa: PLW0603
    with _lock:
        if _settings is None:
            _settings = Settings.from_env()
        return _settings
