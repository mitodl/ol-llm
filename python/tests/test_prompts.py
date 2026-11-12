from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import opik
import pytest
from conftest import OPIK_KEYCLOAK, TOKEN_REPLY, PromptLibrary, wait_for_fetches
from ol_llm import (
    ChatTemplate,
    PromptVariablesError,
    TextTemplate,
    create_prompt,
    get_prompt,
    promote_prompt,
    prompts,
    tracing,
)
from ol_llm.pointers import render_export
from ol_llm.prompt_registry import (
    FilePromptRegistry,
    OpikPromptRegistry,
    PromptRegistry,
    RegistryUnavailableError,
    configure_prompt_registry,
    prompt_registry,
    register_prompt_registry,
)
from ol_llm.tracing import (
    KeycloakClient,
    KeycloakTokens,
    TracingConfig,
    configure_tracing,
)
from opik.api_objects.opik_client import reset_global_client
from opik.api_objects.prompt import PromptTemplate, PromptType, mask_context
from opik.rest_api.core import ApiError

PROJECT = "mit-learn"
DEFAULT = TextTemplate(template="Summarize {{text}}.", version="v1")
CHAT = [
    {"role": "system", "content": "You summarize {{subject}}."},
    {"role": "user", "content": "{{text}}"},
]


def fetch(
    name: str = "summary", default: TextTemplate = DEFAULT, **kwargs: str
) -> prompts.ManagedPrompt[TextTemplate]:
    return get_prompt(name, app=PROJECT, default=default, **kwargs)


def failing(exc: Exception) -> Callable[..., None]:
    """A replacement for Opik's retrieval that fails with ``exc``."""

    def fail(*_args: object, **_kwargs: object) -> None:
        raise exc

    return fail


# The reads every registry serves: the version production runs, by its label and
# by its version. Each test runs against Opik and against committed exports.

Publish = Callable[[str, "str | list[dict[str, Any]]", str], None]


@pytest.fixture(params=["opik", "file"])
def publish(
    request: pytest.FixtureRequest, prompt_library: PromptLibrary, tmp_path: Path
) -> Publish:
    """Make a prompt's production version ``v1`` in the parametrized registry."""

    def to_opik(name: str, template: str | list[dict[str, Any]], kind: str) -> None:
        prompt_library.add(name, template, type=kind, environments=("production",))

    def to_files(name: str, template: str | list[dict[str, Any]], kind: str) -> None:
        exported = (
            TextTemplate(template=template, type=kind, version="v1")  # type: ignore[arg-type]
            if isinstance(template, str)
            else ChatTemplate(messages=template, type=kind, version="v1")  # type: ignore[arg-type]
        )
        (tmp_path / f"{name}.prompt.yaml").write_text(
            render_export(name, PROJECT, exported), encoding="utf-8"
        )

    if request.param == "file":
        configure_prompt_registry(FilePromptRegistry(tmp_path))
        return to_files
    return to_opik


def test_fetches_the_production_version(publish: Publish) -> None:
    publish("summary", "Summarize {{text}}.", "mustache")

    prompt = fetch(environment="production")

    assert (prompt.source, prompt.version) == ("registry", "v1")
    assert prompt.degraded_reasons == ()
    assert prompt.format(text="this") == "Summarize this."


def test_fetches_a_pinned_version(publish: Publish) -> None:
    publish("summary", "Summarize {{text}}.", "mustache")

    assert fetch(version="v1").template.template == "Summarize {{text}}."
    with pytest.raises(LookupError, match="version v2"):
        fetch(version="v2")


def test_a_label_on_no_version_raises(publish: Publish) -> None:
    publish("summary", "Summarize {{text}}.", "mustache")

    with pytest.raises(LookupError, match="environment qa"):
        fetch(environment="qa")


def test_another_apps_prompt_is_not_found(publish: Publish) -> None:
    publish("summary", "Summarize {{text}}.", "mustache")

    with pytest.raises(LookupError, match="'learn-ai'"):
        get_prompt("summary", app="learn-ai", default=DEFAULT, environment="production")


def test_a_chat_prompt_renders_every_message(publish: Publish) -> None:
    publish("tutor", CHAT, "mustache")

    prompt = get_prompt(
        "tutor",
        app=PROJECT,
        default=ChatTemplate(messages=[]),
        environment="production",
    )

    assert prompt.template.messages == CHAT
    assert prompt.format(subject="physics", text="entropy") == [
        {"role": "system", "content": "You summarize physics."},
        {"role": "user", "content": "entropy"},
    ]


def test_a_default_of_the_other_structure_raises(publish: Publish) -> None:
    publish("tutor", CHAT, "mustache")

    with pytest.raises(TypeError, match="chat prompt"):
        fetch("tutor", environment="production")


def test_a_jinja2_prompt_renders_the_same_fetched_or_embedded(
    publish: Publish,
) -> None:
    template = "{% for item in items %}- {{ item }}\n{% endfor %}"
    publish("summary", template, "jinja2")
    embedded = TextTemplate(template=template, type="jinja2")

    fetched = fetch(default=embedded, environment="production")

    assert fetched.template.type == "jinja2"
    assert fetched.format(items=["a", "b"]) == "- a\n- b\n"
    assert embedded.format(items=["a", "b"]) == "- a\n- b\n"


def test_the_hash_matches_the_embedded_default(publish: Publish) -> None:
    publish("summary", DEFAULT.template, "mustache")

    assert fetch(environment="production").template_hash == DEFAULT.template_hash


def test_the_label_defaults_to_the_deployment_environment(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.environments.add("ci")
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    prompt_library.add("summary", "Summarize {{text}} for CI.", environments=("ci",))

    assert fetch().version == "v2"


def test_a_pinned_version_among_several(prompt_library: PromptLibrary) -> None:
    prompt_library.add("summary", "one {{text}}")
    prompt_library.add("summary", "two {{text}}", environments=("production",))

    prompt = fetch(version="v1")

    assert (prompt.version, prompt.template.template) == ("v1", "one {{text}}")


def test_environment_and_version_together_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        fetch(environment="production", version="v1")


# Rendering. Mustache matches Opik's, so a version already in Opik renders as it
# did when Opik rendered it.


@pytest.mark.parametrize(
    ("template", "variables"),
    [
        ("Hi {{name}}.", {"name": "Ada"}),
        ("Hi {{ name }} and {{name}}.", {"name": "Ada"}),
        ("Hi {{name}}.", {" name ": "Ada"}),
        ('JSON: {"a": {{{value}}}}', {"value": "1"}),
        ("{{a}}{{b}}", {"a": "{{b}}", "b": "x"}),
        ("Nothing: [{{gone}}]", {"gone": None}),
        ("Number {{n}}", {"n": 3}),
        ("No placeholders {single} {{}}", {"": "empty"}),
    ],
)
def test_mustache_renders_as_opik_does(
    template: str, variables: dict[str, Any]
) -> None:
    ours = TextTemplate(template=template).format(**variables)

    assert ours == PromptTemplate(template).format(**variables)


def test_jinja2_renders_in_a_sandbox() -> None:
    template = TextTemplate(template="{{ x.__class__ }}", type="jinja2")

    assert template.format(x=1) == PromptTemplate(
        "{{ x.__class__ }}", type=PromptType.JINJA2
    ).format(x=1)


def test_a_chat_template_keeps_its_media_and_other_keys() -> None:
    image = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/{{name}}.png", "detail": "high"},
    }
    video = {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}
    template = ChatTemplate(
        messages=[
            {
                "role": "user",
                "name": "learner",
                "content": [{"type": "text", "text": "Hi {{name}}"}, image, video],
            }
        ]
    )

    [message] = template.format(name="cat")

    assert message == {
        "role": "user",
        "name": "learner",
        "content": [
            {"type": "text", "text": "Hi cat"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.png", "detail": "high"},
            },
            video,
        ],
    }


def test_a_media_part_whose_url_renders_empty_is_left_out() -> None:
    image = {"type": "image_url", "image_url": {"url": "{{image}}"}}
    template = ChatTemplate(
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}, image]}]
    )

    [message] = template.format(image="")

    assert message["content"] == [{"type": "text", "text": "Hi"}]


@pytest.mark.parametrize(
    "template",
    [TextTemplate(template="Hi {{name}}."), ChatTemplate(messages=CHAT)],
)
def test_mustache_variables_must_match_the_placeholders(
    template: TextTemplate | ChatTemplate,
) -> None:
    with pytest.raises(PromptVariablesError, match=r"unexpected \['nobody'\]"):
        template.format(nobody="here")


def test_a_media_url_placeholder_must_be_passed() -> None:
    template = ChatTemplate(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "{{url}}"}}],
            }
        ]
    )

    with pytest.raises(PromptVariablesError, match=r"missing \['url'\]"):
        template.format()


# Outages and refusals, against Opik.


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused"),
        ApiError(status_code=503, body="unavailable"),
    ],
)
def test_a_cold_fetch_with_opik_unreachable_uses_the_embedded_default(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    monkeypatch.setattr(OpikPromptRegistry, "_retrieve", failing(exc))

    prompt = fetch(environment="production")

    assert prompt.source == "embedded"
    assert prompt.template == DEFAULT
    assert prompt.version == "v1"
    assert prompt.degraded_reasons == ("embedded_prompt",)


def test_a_refusal_from_opik_is_not_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        OpikPromptRegistry, "_retrieve", failing(ApiError(status_code=401, body="no"))
    )

    with pytest.raises(ApiError):
        fetch(environment="production")


def keycloak_answering(status: int) -> Callable[..., None]:
    """A retrieval whose token request gets ``status`` from Keycloak."""
    tokens = KeycloakTokens(
        KeycloakClient(
            token_url="https://keycloak.example/token",
            client_id="ol-opik-client",
            client_secret="wrong",
        ),
        http=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(status, json={"error": "x"})
            )
        ),
    )

    def fetch_with_token(*_args: object, **_kwargs: object) -> None:
        tokens.token()

    return fetch_with_token


def test_a_keycloak_refusal_on_a_cold_fetch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OpikPromptRegistry, "_retrieve", keycloak_answering(401))

    with pytest.raises(httpx.HTTPStatusError, match="401"):
        fetch(environment="production")


def test_keycloak_down_on_a_cold_fetch_is_an_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OpikPromptRegistry, "_retrieve", keycloak_answering(503))

    assert fetch(environment="production").degraded_reasons == ("embedded_prompt",)


def test_a_bad_url_is_not_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        OpikPromptRegistry, "_retrieve", failing(httpx.UnsupportedProtocol("opik://"))
    )

    with pytest.raises(httpx.UnsupportedProtocol):
        fetch(environment="production")


def test_fetch_prompt_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        OpikPromptRegistry, "_retrieve", failing(httpx.ConnectError("refused"))
    )

    with pytest.raises(RegistryUnavailableError):
        prompts.fetch_prompt("summary", app=PROJECT, environment="production")


# The cache, the deadline, and the background refresh.


def test_a_worker_that_cannot_start_leaves_nothing_in_flight(
    monkeypatch: pytest.MonkeyPatch, prompt_library: PromptLibrary
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))

    def no_threads(**_kwargs: object) -> None:
        msg = "can't start new thread"
        raise RuntimeError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(prompts, "threading", SimpleNamespace(Thread=no_threads))
        with pytest.raises(RuntimeError):
            fetch(environment="production")

    assert prompts._inflight == {}  # noqa: SLF001
    assert fetch(environment="production").source == "registry"


def test_a_fetch_inside_a_mask_context_returns_the_masked_version(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    masked = prompt_library.add("summary", "Masked {{text}}.")
    [unmasked] = [
        v for v in prompt_library.prompts[PROJECT, "summary"] if v["id"] != masked["id"]
    ]

    with mask_context.prompt_mask_context({unmasked["prompt_id"]: masked["id"]}):
        inside = fetch(environment="production")
    outside = fetch(environment="production")

    assert inside.template.template == "Masked {{text}}."
    assert outside.template.template == "Summarize {{text}}."


def test_a_masked_chat_prompt_stays_chat(prompt_library: PromptLibrary) -> None:
    prompt_library.add("tutor", CHAT, environments=("production",))
    masked = prompt_library.add("tutor", [{"role": "system", "content": "Masked"}])
    [unmasked] = [
        v for v in prompt_library.prompts[PROJECT, "tutor"] if v["id"] != masked["id"]
    ]

    with mask_context.prompt_mask_context({unmasked["prompt_id"]: masked["id"]}):
        prompt = get_prompt(
            "tutor",
            app=PROJECT,
            default=ChatTemplate(messages=[]),
            environment="production",
        )

    assert prompt.template.messages == [{"role": "system", "content": "Masked"}]


def test_a_slow_cold_fetch_falls_back_and_then_fills_the_cache(
    monkeypatch: pytest.MonkeyPatch, prompt_library: PromptLibrary
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    prompt_library.delay_seconds = 0.5
    monkeypatch.setattr(prompts, "COLD_FETCH_DEADLINE_SECONDS", 0.05)

    first = fetch(environment="production")
    wait_for_fetches()
    second = fetch(environment="production")

    assert first.degraded_reasons == ("embedded_prompt",)
    assert (second.source, second.degraded_reasons) == ("registry", ())
    assert prompt_library.retrievals == 1


def test_a_cached_prompt_is_served_without_a_fetch(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    fetch(environment="production")
    prompt_library.reset()

    prompt = fetch(environment="production")

    assert (prompt.source, prompt.version) == ("registry", "v1")
    assert prompt_library.retrievals == 0


def make_stale() -> None:
    for entry in prompts._cache.values():  # noqa: SLF001
        entry.checked_at -= prompts.PROMPT_CACHE_TTL_SECONDS + 1


def test_a_stale_prompt_is_served_while_opik_is_down(
    monkeypatch: pytest.MonkeyPatch, prompt_library: PromptLibrary
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    fetch(environment="production")
    make_stale()
    attempts: list[object] = []

    def refused(*_args: object, **_kwargs: object) -> None:
        attempts.append(None)
        msg = "refused"
        raise httpx.ConnectError(msg)

    monkeypatch.setattr(OpikPromptRegistry, "_retrieve", refused)

    served = [fetch(environment="production") for _ in range(3)]

    for prompt in served:
        assert (prompt.source, prompt.version, prompt.degraded_reasons) == (
            "registry",
            "v1",
            (),
        )
    # Tried again after another TTL, not on every call.
    assert len(attempts) == 1


def test_a_stale_prompt_is_kept_when_its_label_is_gone(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    fetch(environment="production")
    for version in prompt_library.prompts[PROJECT, "summary"]:
        version["environments"] = []
    make_stale()

    assert fetch(environment="production").version == "v1"


def test_a_stale_prompt_is_fetched_again_on_the_next_call(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "one {{text}}", environments=("production",))
    fetch(environment="production")
    # Someone moves the label in the Opik UI.
    prompt_library.add("summary", "two {{text}}", environments=("production",))
    make_stale()

    assert fetch(environment="production").version == "v2"


def test_a_slow_refresh_serves_the_stale_prompt(
    monkeypatch: pytest.MonkeyPatch, prompt_library: PromptLibrary
) -> None:
    prompt_library.add("summary", "one {{text}}", environments=("production",))
    fetch(environment="production")
    prompt_library.add("summary", "two {{text}}", environments=("production",))
    make_stale()
    prompt_library.delay_seconds = 0.3
    monkeypatch.setattr(prompts, "COLD_FETCH_DEADLINE_SECONDS", 0.05)

    stale = fetch(environment="production")
    wait_for_fetches()
    fresh = fetch(environment="production")

    assert (stale.version, fresh.version) == ("v1", "v2")


def test_the_cache_keeps_the_most_recently_used(
    monkeypatch: pytest.MonkeyPatch, prompt_library: PromptLibrary
) -> None:
    monkeypatch.setattr(prompts, "PROMPT_CACHE_MAX_ENTRIES", 2)
    for text in ("one", "two", "three"):
        prompt_library.add("summary", f"{text} {{{{text}}}}")
    fetch(version="v1")
    fetch(version="v2")
    fetch(version="v1")
    fetch(version="v3")
    prompt_library.retrievals = 0

    fetch(version="v1")
    fetch(version="v3")
    fetch(version="v2")

    assert prompt_library.retrievals == 1


def test_with_tracing_off_the_embedded_default_is_used_unmarked(
    prompt_library: PromptLibrary,
) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False))
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))

    prompt = fetch(environment="production")

    assert (prompt.source, prompt.degraded_reasons) == ("embedded", ())
    assert prompt_library.retrievals == 0


def test_with_tracing_off_a_fetch_still_sends_the_keycloak_token(
    prompt_library: PromptLibrary,
) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False, keycloak=OPIK_KEYCLOAK))
    # The SDK reads its hooks when it builds its client.
    reset_global_client()
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))

    prompt = prompts.fetch_prompt("summary", app=PROJECT, environment="production")

    assert prompt is not None
    assert prompt.version == "v1"
    assert prompt_library.authorizations == [f"Bearer {TOKEN_REPLY['access_token']}"]


def test_a_fetch_inside_a_trace_links_the_version(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))

    @opik.track(project_name=PROJECT)
    def handler() -> None:
        fetch(environment="production")

    with opik.record_traces_locally() as storage:
        handler()
        [trace] = storage.trace_trees

    assert trace.metadata is not None
    [linked] = trace.metadata["opik_prompts"]
    assert linked["name"] == "summary"
    assert linked["version"]["version_number"] == "v1"


def test_the_hash_follows_the_text_not_where_it_came_from(
    monkeypatch: pytest.MonkeyPatch, prompt_library: PromptLibrary
) -> None:
    prompt_library.add("summary", DEFAULT.template, environments=("production",))
    fetched = fetch(environment="production")
    monkeypatch.setattr(
        OpikPromptRegistry, "_retrieve", failing(httpx.ConnectError("down"))
    )
    fallback = fetch("other", environment="production")
    edited = fetch(default=TextTemplate(template="Summarize {{text}}!"), version="v9")

    assert fetched.source != fallback.source
    assert fetched.template_hash == fallback.template_hash
    assert fetched.template_hash != edited.template_hash


# Writing versions and moving labels.


def test_create_prompt_adds_a_version_only_on_change(
    prompt_library: PromptLibrary,
) -> None:
    template = TextTemplate(template="Summarize {{text}}.")

    first = create_prompt("summary", template, app=PROJECT)
    again = create_prompt("summary", template, app=PROJECT)
    changed = create_prompt(
        "summary", TextTemplate(template="Summarize {{text}}!"), app=PROJECT
    )
    retyped = create_prompt(
        "summary",
        TextTemplate(template="Summarize {{text}}!", type="jinja2"),
        app=PROJECT,
    )

    assert [first.version, again.version, changed.version, retyped.version] == [
        "v1",
        "v1",
        "v2",
        "v3",
    ]
    assert prompt_library.versions("summary") == ["v1", "v2", "v3"]


def test_create_prompt_ignores_metadata_set_in_the_ui(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", metadata={"by": "ui"})

    created = create_prompt(
        "summary", TextTemplate(template="Summarize {{text}}."), app=PROJECT
    )

    assert created.version == "v1"
    assert prompt_library.versions("summary") == ["v1"]


def test_create_prompt_for_a_chat_prompt(prompt_library: PromptLibrary) -> None:
    template = ChatTemplate(messages=CHAT)

    create_prompt("tutor", template, app=PROJECT)
    created = create_prompt("tutor", template, app=PROJECT)

    assert created.template.messages == CHAT
    assert prompt_library.versions("tutor") == ["v1"]


def test_create_prompt_of_the_other_structure_raises(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("tutor", CHAT)

    with pytest.raises(TypeError):
        create_prompt("tutor", TextTemplate(template="Hi"), app=PROJECT)


def test_promote_keeps_the_versions_other_labels(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.environments.add("qa")
    prompt_library.add("summary", "one", environments=("production",))
    prompt_library.add("summary", "two", environments=("qa",))

    promote_prompt("summary", version="v2", environment="production", app=PROJECT)

    assert prompt_library.labels("summary") == {"v1": [], "v2": ["qa", "production"]}


def test_promote_reaches_the_next_fetch_in_this_process(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "one {{text}}", environments=("production",))
    prompt_library.add("summary", "two {{text}}")
    assert fetch(environment="production").version == "v1"

    promote_prompt("summary", version="v2", environment="production", app=PROJECT)

    assert fetch(environment="production").version == "v2"


def test_promote_registers_an_unregistered_environment(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "one")

    promote_prompt("summary", version="v1", environment="ci", app=PROJECT)

    assert "ci" in prompt_library.environments
    assert prompt_library.labels("summary") == {"v1": ["ci"]}


def test_promote_to_a_missing_version_raises(prompt_library: PromptLibrary) -> None:
    prompt_library.add("summary", "one")

    with pytest.raises(LookupError):
        promote_prompt("summary", version="v2", environment="production", app=PROJECT)


# The file registry and choosing a registry by name.


def test_the_file_registry_is_read_only(tmp_path: Path) -> None:
    registry = FilePromptRegistry(tmp_path)

    with pytest.raises(NotImplementedError):
        registry.move_label("summary", app=PROJECT, version="v1", label="production")
    with pytest.raises(NotImplementedError):
        registry.create_version("summary", DEFAULT, app=PROJECT)


def test_the_file_registry_serves_the_labels_it_is_given(tmp_path: Path) -> None:
    (tmp_path / "summary.prompt.yaml").write_text(
        render_export("summary", PROJECT, DEFAULT), encoding="utf-8"
    )
    registry = FilePromptRegistry(tmp_path, labels=("production", "development"))

    assert registry.label_version("summary", app=PROJECT, label="development") == "v1"
    assert registry.label_version("summary", app=PROJECT, label="qa") is None


def test_the_registry_is_chosen_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OL_LLM_PROMPT_REGISTRY", "file")
    monkeypatch.setenv("OL_LLM_PROMPT_REGISTRY_PATH", str(tmp_path))

    registry = prompt_registry()

    assert isinstance(registry, FilePromptRegistry)
    assert registry.directory == tmp_path


def test_the_file_registry_serves_the_configured_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "summary.prompt.yaml").write_text(
        render_export("summary", PROJECT, DEFAULT), encoding="utf-8"
    )
    monkeypatch.setenv("OL_LLM_PROMPT_REGISTRY", "file")
    monkeypatch.setenv("OL_LLM_PROMPT_REGISTRY_PATH", str(tmp_path))
    monkeypatch.setenv("OL_LLM_PROMPT_REGISTRY_LABELS", "production, ci")

    # The label defaults to Settings.environment, ci in these tests.
    prompt = fetch()

    assert (prompt.source, prompt.version) == ("registry", "v1")


def test_the_default_registry_is_opik() -> None:
    assert isinstance(prompt_registry(), OpikPromptRegistry)


def test_a_registered_registry_is_chosen_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = FilePromptRegistry(tmp_path)

    def factory(_env: object) -> PromptRegistry:
        return built

    register_prompt_registry("custom", factory)
    monkeypatch.setenv("OL_LLM_PROMPT_REGISTRY", "custom")

    assert prompt_registry() is built


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"OL_LLM_PROMPT_REGISTRY": "nowhere"}, "No prompt registry named 'nowhere'"),
        ({"OL_LLM_PROMPT_REGISTRY": "file"}, "OL_LLM_PROMPT_REGISTRY_PATH"),
    ],
)
def test_a_registry_that_cannot_be_built_raises(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], message: str
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        prompt_registry()
