import json
import logging
from collections.abc import Callable, Mapping
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx
import opik
import pytest
from ol_llm import UseCase, models, run_sync
from ol_llm.catalog import (
    TOKENS_PER_MTOK,
    Catalog,
    CatalogEntry,
    check_model_ids,
    load_catalog,
    main,
    price_rules,
)
from ol_llm.models import (
    PolicyFallbackModel,
    PolicyModel,
    build_agent,
    build_model,
    override_models,
)
from ol_llm.testing import policy_model
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

# The tests build real provider models, which need credentials to exist.
pytestmark = pytest.mark.usefixtures("provider_env")

CATALOG = load_catalog()
# The chat models build_model will build.
CHAT_ENTRIES = [
    e for e in CATALOG.entries.values() if e.kind == "chat" and e.status != "retired"
]

TEST_CATALOG = """
[[model]]
id = "openai:gpt-4o-mini"
kind = "chat"
input_usd_per_mtok = "0.15"
output_usd_per_mtok = "0.60"

[[model]]
id = "openai:gpt-4"
kind = "chat"
status = "retired"
input_usd_per_mtok = "30"
output_usd_per_mtok = "60"

[[model]]
id = "openai:gpt-4-turbo"
kind = "chat"
status = "deprecated"
retires_on = 2027-01-01
input_usd_per_mtok = "10"
output_usd_per_mtok = "30"

[[model]]
id = "openai:gpt-5"
kind = "chat"
route = "gateway"
route_falls_back = true
input_usd_per_mtok = "1.25"
output_usd_per_mtok = "10"

[[model]]
id = "openai:text-embedding-3-small"
kind = "embedding"
input_usd_per_mtok = "0.02"
output_usd_per_mtok = "0"
"""


def use_case(model: str, *fallbacks: str) -> UseCase:
    return UseCase(
        app="mit-learn",
        name="summary",
        profile="request",
        model=model,
        fallback_models=fallbacks,
    )


@pytest.fixture
def test_catalog(monkeypatch: pytest.MonkeyPatch) -> Catalog:
    catalog = Catalog.from_toml(TEST_CATALOG)
    monkeypatch.setattr(models, "load_catalog", lambda: catalog)
    return catalog


def test_released_catalog_has_both_kinds() -> None:
    kinds = {entry.kind for entry in CATALOG.entries.values()}

    assert kinds == {"chat", "embedding"}


def test_unlisted_model_builds_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="ol_llm.models"):
        model = build_model(use_case("openai:gpt-3.5-turbo"))

    assert isinstance(model, PolicyModel)
    assert model.catalog_id == "openai:gpt-3.5-turbo"
    assert model.catalog_entry is None
    assert "openai:gpt-3.5-turbo" in caplog.text
    assert "does not list" in caplog.text


def test_unlisted_fallback_builds() -> None:
    model = build_model(use_case("openai:gpt-4o-mini", "openai:gpt-3.5-turbo"))

    assert isinstance(model, PolicyFallbackModel)
    assert [m.catalog_entry for m in model.models] == [  # type: ignore[attr-defined]
        CATALOG.entries["openai:gpt-4o-mini"],
        None,
    ]


@pytest.mark.parametrize("model_id", ["gpt-4o-mini", "openai:", ":gpt-4o-mini"])
def test_build_model_refuses_a_malformed_id(model_id: str) -> None:
    with pytest.raises(ValueError, match="not in provider:model form"):
        build_model(use_case(model_id))


@pytest.mark.usefixtures("test_catalog")
def test_build_model_refuses_a_retired_model() -> None:
    with pytest.raises(ValueError, match="retired"):
        build_model(use_case("openai:gpt-4"))


@pytest.mark.usefixtures("test_catalog")
def test_build_model_refuses_an_embedding_model() -> None:
    with pytest.raises(ValueError, match="kind 'embedding', not 'chat'"):
        build_model(use_case("openai:text-embedding-3-small"))


@pytest.mark.usefixtures("test_catalog")
def test_deprecated_model_builds_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="ol_llm.models"):
        model = build_model(use_case("openai:gpt-4-turbo"))

    assert isinstance(model, PolicyModel)
    assert model.catalog_entry.status == "deprecated"
    assert "openai:gpt-4-turbo" in caplog.text
    assert "2027-01-01" in caplog.text


@pytest.mark.usefixtures("test_catalog")
def test_route_that_falls_back_takes_no_fallback_models() -> None:
    assert isinstance(build_model(use_case("openai:gpt-5")), PolicyModel)
    with pytest.raises(ValueError, match="already falls back"):
        build_model(use_case("openai:gpt-5", "openai:gpt-4o-mini"))


@pytest.mark.usefixtures("test_catalog")
def test_unlisted_primary_takes_fallback_models() -> None:
    model = build_model(use_case("openai:gpt-5.9", "openai:gpt-4o-mini"))

    assert isinstance(model, PolicyFallbackModel)


@pytest.mark.usefixtures("test_catalog")
@pytest.mark.parametrize(
    ("models_named", "tags"),
    [
        (("openai:gpt-4o-mini",), []),
        (("openai:gpt-4-turbo",), ["model_status:deprecated"]),
        (("openai:gpt-4o-mini", "openai:gpt-5.9"), ["model_status:uncataloged"]),
        (
            ("openai:gpt-5.9", "openai:gpt-4-turbo"),
            ["model_status:deprecated", "model_status:uncataloged"],
        ),
    ],
)
def test_a_run_tags_its_trace_with_the_catalog_status(
    models_named: tuple[str, ...], tags: list[str]
) -> None:
    with override_models(TestModel(custom_output_text="ok")):
        agent = build_agent(use_case(*models_named))

    with opik.record_traces_locally() as storage:
        run_sync(agent, "summarize")
        [trace] = storage.trace_trees

    assert [t for t in trace.tags or [] if t.startswith("model_status:")] == tags


def test_each_chain_member_carries_its_entry() -> None:
    model = build_model(
        use_case("openai:gpt-4o-mini", "bedrock:us.anthropic.claude-sonnet-5")
    )

    assert isinstance(model, PolicyFallbackModel)
    assert [m.catalog_entry for m in model.models] == [  # type: ignore[attr-defined]
        CATALOG.entries["openai:gpt-4o-mini"],
        CATALOG.entries["bedrock:us.anthropic.claude-sonnet-5"],
    ]


@pytest.mark.parametrize(
    "model_id", ["openai-chat:gpt-4o-mini", "openai-responses:gpt-4o-mini"]
)
def test_api_shape_providers_share_the_base_entry(model_id: str) -> None:
    model = build_model(use_case(model_id))

    assert isinstance(model, PolicyModel)
    assert model.catalog_id == model_id
    assert model.catalog_entry is CATALOG.entries["openai:gpt-4o-mini"]


@pytest.mark.usefixtures("test_catalog")
def test_a_nested_run_tags_the_callers_trace() -> None:
    with override_models(TestModel(custom_output_text="ok")):
        agent = build_agent(use_case("openai:gpt-5.9"))

    @opik.track(project_name="mit-learn")
    def handler() -> None:
        run_sync(agent, "summarize")

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees

    assert "model_status:uncataloged" in (trace.tags or [])


def test_policy_model_checks_the_catalog() -> None:
    assert (
        policy_model(use_case("openai:gpt-3.5-turbo"), TestModel()).catalog_entry
        is None
    )
    with pytest.raises(ValueError, match="retired"):
        policy_model(use_case("google:gemini-3-pro-preview"), TestModel())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.replace('"0.15"', "0.15"), "quoted decimal"),
        (
            lambda t: t.replace('kind = "chat"', 'kind = "chat"\ncolor = "red"', 1),
            "Extra inputs",
        ),
        (lambda t: t.replace('kind = "chat"', 'kind = "vision"', 1), "'vision'"),
        (lambda t: t.replace("retires_on = 2027-01-01\n", ""), "needs retires_on"),
        (lambda t: t.replace('route = "gateway"', 'route = "direct"'), "gateway route"),
        (lambda t: t.replace('"openai:gpt-4"', '"openai:gpt-4o-mini"'), "listed twice"),
        (lambda t: t.replace('"openai:gpt-4"', '"gpt-4"'), "provider:model"),
        (lambda t: t.replace('"30"', '"-30"', 1), "negative"),
        (
            lambda t: t.replace('kind = "chat"', 'kind = "chat"\nopik_model = "x"', 1),
            "together",
        ),
        (lambda t: t.replace('kind = "chat"\n', "", 1), "kind"),
        (
            lambda t: t.replace("route_falls_back = true", 'route_falls_back = "true"'),
            "bool",
        ),
        (
            lambda t: t.replace("retires_on = 2027-01-01", 'retires_on = "2027-01-01"'),
            "date",
        ),
    ],
)
def test_malformed_catalog_is_refused(
    change: Callable[[str], str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Catalog.from_toml(change(TEST_CATALOG))


def test_check_model_ids(test_catalog: Catalog) -> None:
    result = check_model_ids(
        [
            "openai:gpt-4o-mini",
            "openai-chat:gpt-4o-mini",
            "openai:gpt-4-turbo",
            "openai:gpt-4",
            "gpt-4o-mini",
            "openai:gpt-3.5-turbo",
            "openai:gpt-4",
        ],
        test_catalog,
    )

    assert not result.ok
    assert result.errors == (
        "openai:gpt-4: retired",
        "gpt-4o-mini: not in provider:model form",
    )
    assert result.warnings == (
        "openai:gpt-4-turbo: deprecated, retiring on 2027-01-01",
        "openai:gpt-3.5-turbo: not in the model catalog, so it has no price",
    )


def test_check_model_ids_passes_a_clean_list() -> None:
    assert check_model_ids(["openai:gpt-4o-mini", "azure:gpt-5-mini"]).ok


def test_check_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ids = tmp_path / "models.txt"
    ids.write_text("# learn-ai\nopenai:gpt-4o-mini\n\nazure:gpt-4o  # tutor\n")

    assert main(["check", "--file", str(ids), "mistral:mistral-large-latest"]) == 0
    assert main(["check", "--file", str(ids), "openai:gpt-3.5-turbo"]) == 0
    assert (
        "warning: openai:gpt-3.5-turbo: not in the model catalog"
        in capsys.readouterr().err
    )
    assert main(["check", "google:gemini-3-pro-preview"]) == 1
    assert "error: google:gemini-3-pro-preview: retired" in capsys.readouterr().err


def test_check_command_needs_ids() -> None:
    with pytest.raises(SystemExit):
        main(["check"])


def test_price_rules_record_a_price_per_token() -> None:
    rules = price_rules()["groups"][0]["rules"]
    by_series = {
        (r["record"], r["labels"]["provider"], r["labels"]["model"]): r for r in rules
    }

    assert len(rules) == 2 * len(CATALOG.entries)
    rule = by_series[("ol_llm_model_input_usd_per_token", "openai", "gpt-4o-mini")]
    assert rule["expr"] == "vector(0.00000015)"


def test_prices_command_prints_the_rules(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["prices"]) == 0
    assert json.loads(capsys.readouterr().out) == price_rules()


@pytest.mark.parametrize("entry", CHAT_ENTRIES, ids=lambda e: e.id)
def test_pydantic_ai_can_price_every_chat_model(entry: CatalogEntry) -> None:
    """A cost_limit on a model genai-prices cannot price enforces nothing (§6.8).

    The price is looked up the way PydanticAI does it for a run, from the
    provider model build_model makes.
    """
    model = build_model(use_case(entry.id))
    assert isinstance(model, PolicyModel)
    response = ModelResponse(
        parts=[],
        usage=RequestUsage(input_tokens=1000, output_tokens=1000),
        model_name=model.wrapped.model_name,
        provider_name=model.wrapped.system,
        provider_url=model.wrapped.base_url,
    )

    assert response.cost().total_price > 0


# Opik prices a span from its own table, which ships in the backend image rather
# than the SDK. The SDK and the backend are released together under one version.
OPIK_PRICE_FILES = (
    "model_prices_and_context_window.json",
    "model_prices_overrides.json",
)
OPIK_PRICES_URL = (
    "https://raw.githubusercontent.com/comet-ml/opik/{version}/"
    "apps/opik-backend/src/main/resources/{name}"
)
# The litellm_provider values Opik files under each provider name it prices by:
# CostService.PROVIDERS_MAPPING in apps/opik-backend/.../domain/cost/CostService.java.
OPIK_PROVIDERS: Mapping[str, frozenset[str]] = {
    "openai": frozenset({"openai"}),
    "azure": frozenset({"azure", "microsoft"}),
    "anthropic": frozenset({"anthropic"}),
    "bedrock": frozenset({"bedrock", "bedrock_converse"}),
    "google_ai": frozenset({"gemini"}),
    "google_vertexai": frozenset({"vertex_ai-language-models"}),
    "mistral": frozenset({"mistral"}),
}


def _opik_key(name: str, provider: str) -> tuple[str, str] | None:
    canonical = next((p for p, raw in OPIK_PROVIDERS.items() if provider in raw), None)
    if canonical is None or (canonical == "bedrock" and "/" in name):
        return None
    return name.partition("/")[2] or name, canonical


@pytest.fixture(scope="module")
def opik_prices() -> dict[tuple[str, str], dict[str, Any]]:
    """Opik's prices by (model, provider), resolved as the backend loads them."""
    upstream, overrides = (
        httpx.get(
            OPIK_PRICES_URL.format(version=version("opik"), name=name),
            follow_redirects=True,
            timeout=60,
        )
        .raise_for_status()
        .json(parse_float=Decimal)
        for name in OPIK_PRICE_FILES
    )
    prices: dict[tuple[str, str], dict[str, Any]] = {}
    for name, row in [*upstream.items(), *overrides.items()]:
        target = upstream.get(row["alias_of"]) if "alias_of" in row else row
        if target and (key := _opik_key(name, target.get("litellm_provider", ""))):
            prices[key] = target
    return prices


@pytest.mark.network
@pytest.mark.parametrize("entry", CATALOG.entries.values(), ids=lambda e: e.id)
def test_opik_prices_every_model_at_the_catalog_price(
    entry: CatalogEntry, opik_prices: dict[tuple[str, str], dict[str, Any]]
) -> None:
    if entry.opik_model is None:
        # Leaving the Opik name out is only for a model Opik cannot price, so
        # this fails once Opik adds it and the entry should name it.
        priced = [key for key in opik_prices if key[0] == entry.model_name]
        assert not priced, f"Opik prices {entry.id} as {priced}; name it"
        return
    row = opik_prices.get((entry.opik_model, str(entry.opik_provider)))

    assert row is not None, f"Opik has no price for {entry.opik_model}"
    opik_input = row.get("input_cost_per_token", Decimal(0)) * TOKENS_PER_MTOK
    opik_output = row.get("output_cost_per_token", Decimal(0)) * TOKENS_PER_MTOK
    assert (opik_input, opik_output) == (
        entry.input_usd_per_mtok,
        entry.output_usd_per_mtok,
    )
