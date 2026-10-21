import pytest
from ol_llm.attribution import Attribution, degraded_metadata, trace_fields
from ol_llm.policy import CallPolicy, UseCase
from ol_llm.settings import GatewayRoute, Settings, configure, get_settings

TUTOR = UseCase(
    app="learn-ai", name="tutor", profile="request", model="openai:gpt-4o-mini"
)
POLICY = CallPolicy(timeout_seconds=60, max_retries=2, max_backoff_seconds=10)


def test_environment_is_required() -> None:
    with pytest.raises(ValueError, match="OL_LLM_ENVIRONMENT"):
        Settings.from_env({})


def test_gateway_routes_are_read_per_family() -> None:
    settings = Settings.from_env(
        {
            "OL_LLM_ENVIRONMENT": "qa",
            "OL_LLM_GATEWAY_OPENAI_BASE_URL": "https://gateway.example/openai/v1",
            "OL_LLM_GATEWAY_OPENAI_API_KEY": "app-key",
        }
    )

    assert settings.environment == "qa"
    assert settings.gateway_routes == {
        "openai": GatewayRoute(
            base_url="https://gateway.example/openai/v1", api_key="app-key"
        )
    }


def test_route_without_a_key_is_refused() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        Settings.from_env(
            {
                "OL_LLM_ENVIRONMENT": "qa",
                "OL_LLM_GATEWAY_ANTHROPIC_BASE_URL": "https://gateway.example",
            }
        )


def test_route_for_an_unknown_family_is_refused() -> None:
    with pytest.raises(ValueError, match="No gateway route family"):
        Settings(
            environment="qa",
            gateway_routes={"bedrock": GatewayRoute(base_url="u", api_key="k")},
        )


def test_route_key_is_not_in_the_repr() -> None:
    assert "app-key" not in repr(GatewayRoute(base_url="u", api_key="app-key"))


def test_configured_settings_win_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OL_LLM_ENVIRONMENT", "production")
    configure(Settings(environment="qa"))

    assert get_settings().environment == "qa"


def test_trace_fields_follow_the_schema() -> None:
    fields = trace_fields(
        TUTOR,
        environment="production",
        policy=POLICY,
        attribution=Attribution(user_id="anon-abc", thread_id="t-1", run_id="r-1"),
    )

    assert fields.project_name == "learn-ai"
    assert fields.environment == "production"
    assert fields.thread_id == "t-1"
    assert fields.tags == ("app:learn-ai", "use_case:tutor", "env:production")
    assert fields.metadata == {
        "app": "learn-ai",
        "use_case": "tutor",
        "environment": "production",
        "policy_timeout_seconds": 60,
        "policy_max_retries": 2,
        "policy_max_backoff_seconds": 10,
        "user_id": "anon-abc",
        "run_id": "r-1",
    }


def test_identifiers_never_reach_tags() -> None:
    fields = trace_fields(
        TUTOR,
        environment="ci",
        policy=POLICY,
        attribution=Attribution(user_id="u-1", thread_id="t-1", run_id="r-1"),
    )

    assert not any(
        identifier in tag for tag in fields.tags for identifier in ("u-1", "t-1", "r-1")
    )


def test_unset_identifiers_are_left_out() -> None:
    fields = trace_fields(TUTOR, environment="ci", policy=POLICY)

    assert "user_id" not in fields.metadata
    assert "run_id" not in fields.metadata
    assert fields.thread_id is None


def test_degraded_metadata() -> None:
    assert degraded_metadata(()) == {}
    assert degraded_metadata(
        ("fallback_model", "partial_output", "fallback_model")
    ) == {
        "degraded": True,
        "degraded_reason": "fallback_model,partial_output",
    }


def test_an_async_azure_token_provider_is_refused() -> None:
    async def token() -> str:
        return "token"

    with pytest.raises(ValueError, match="must be synchronous"):
        Settings(environment="ci", azure_ad_token_provider=token)  # type: ignore[arg-type]
