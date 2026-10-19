import os
from collections.abc import Generator

import pytest
from ol_llm import models
from ol_llm.settings import Settings, configure


@pytest.fixture(autouse=True)
def test_settings() -> Generator[Settings]:
    settings = Settings(environment="ci")
    configure(settings)
    yield settings
    configure(settings)


@pytest.fixture(autouse=True)
def clear_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("OL_LLM_"):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff waits instead of sleeping."""
    recorded: list[float] = []

    async def record(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(models, "_sleep", record)
    return recorded


@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
        "OPENAI_API_VERSION": "2024-10-21",
        "GOOGLE_API_KEY": "test-key",
        "MISTRAL_API_KEY": "test-key",
        "TOGETHER_API_KEY": "test-key",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
    }.items():
        monkeypatch.setenv(name, value)
