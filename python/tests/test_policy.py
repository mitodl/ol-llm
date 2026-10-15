import pytest
from ol_llm.policy import CallPolicy, UseCase, resolve_policy


def use_case(**overrides: object) -> UseCase:
    fields: dict[str, object] = {
        "app": "mit-learn",
        "name": "course-translation",
        "profile": "batch",
        "model": "openai:gpt-4o-mini",
    }
    fields.update(overrides)
    return UseCase(**fields)  # type: ignore[arg-type]


def test_profile_defaults() -> None:
    assert resolve_policy(use_case(profile="request"), {}) == CallPolicy(
        timeout_seconds=60, max_retries=2, max_backoff_seconds=10
    )
    assert resolve_policy(use_case(profile="batch"), {}) == CallPolicy(
        timeout_seconds=300, max_retries=5, max_backoff_seconds=300
    )


def test_use_case_value_overrides_profile_default() -> None:
    declared = use_case(timeout_seconds=120, policy_reason="documented 120 s limit")

    assert resolve_policy(declared, {}).timeout_seconds == 120
    assert resolve_policy(declared, {}).max_retries == 5


def test_zero_retries_on_the_use_case_is_kept() -> None:
    declared = use_case(max_retries=0, policy_reason="caller retries itself")

    assert resolve_policy(declared, {}).max_retries == 0


def test_override_without_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="policy_reason"):
        use_case(max_retries=1)


@pytest.mark.parametrize("name", ["request", "batch"])
def test_profile_names_are_reserved(name: str) -> None:
    with pytest.raises(ValueError, match="profile name"):
        use_case(name=name)


def test_unknown_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="profile"):
        use_case(profile="interactive")


def test_environment_overrides_resolve_in_order() -> None:
    declared = use_case(timeout_seconds=120, policy_reason="documented 120 s limit")
    env = {
        "OL_LLM_BATCH_TIMEOUT_SECONDS": "200",
        "OL_LLM_BATCH_MAX_RETRIES": "3",
        "OL_LLM_COURSE_TRANSLATION_MAX_RETRIES": "1",
    }

    assert resolve_policy(declared, env) == CallPolicy(
        timeout_seconds=200, max_retries=1, max_backoff_seconds=300
    )


def test_other_profiles_variables_are_ignored() -> None:
    env = {"OL_LLM_REQUEST_TIMEOUT_SECONDS": "5"}

    assert resolve_policy(use_case(profile="batch"), env).timeout_seconds == 300


def test_use_case_name_is_upper_cased_for_the_environment() -> None:
    assert use_case(name="feedback.summary v2").env_name == "FEEDBACK_SUMMARY_V2"


def test_unparseable_environment_value_names_the_variable() -> None:
    env = {"OL_LLM_COURSE_TRANSLATION_MAX_RETRIES": "two"}

    with pytest.raises(ValueError, match="OL_LLM_COURSE_TRANSLATION_MAX_RETRIES"):
        resolve_policy(use_case(), env)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("OL_LLM_BATCH_TIMEOUT_SECONDS", "0"),
        ("OL_LLM_BATCH_TIMEOUT_SECONDS", "nan"),
        ("OL_LLM_BATCH_MAX_RETRIES", "-1"),
        ("OL_LLM_BATCH_MAX_BACKOFF_SECONDS", "-1"),
    ],
)
def test_out_of_range_environment_value_is_refused(variable: str, value: str) -> None:
    with pytest.raises(ValueError, match="resolved policy"):
        resolve_policy(use_case(), {variable: value})


def test_out_of_range_declared_value_is_refused() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        use_case(timeout_seconds=0, policy_reason="r")


def test_fallbacks_are_stored_as_a_tuple() -> None:
    declared = use_case(fallback_models=["anthropic:claude-haiku-4-5"])

    assert declared.fallback_models == ("anthropic:claude-haiku-4-5",)
    assert declared.model_ids == ("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5")
    assert hash(declared) == hash(
        use_case(fallback_models=("anthropic:claude-haiku-4-5",))
    )
