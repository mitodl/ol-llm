import time
from types import SimpleNamespace

import httpx
import opik
import pytest
from conftest import PromptLibrary
from ol_llm import (
    ChatTemplate,
    TextTemplate,
    create_prompt,
    get_prompt,
    promote_prompt,
    prompts,
    tracing,
)
from ol_llm.tracing import (
    KeycloakClient,
    KeycloakTokens,
    TracingConfig,
    configure_tracing,
)
from opik import context_storage
from opik.api_objects.prompt import mask_context
from opik.exceptions import (
    EnvironmentNotFoundError,
    PromptPlaceholdersDontMatchFormatArguments,
    PromptTemplateStructureMismatch,
)
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
    return get_prompt(name, project_name=PROJECT, default=default, **kwargs)


def unreachable(exc: Exception) -> object:
    def fail(_lookup: object) -> None:
        raise exc

    return fail


def test_fetches_the_version_the_label_points_at(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    prompt_library.add("summary", "Summarize {{text}} briefly.")

    prompt = fetch(environment="production")

    assert prompt.source == "opik"
    assert prompt.version == "v1"
    assert prompt.degraded_reasons == ()
    assert prompt.format(text="this") == "Summarize this."


def test_the_label_defaults_to_the_deployment_environment(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.environments.add("ci")
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    prompt_library.add("summary", "Summarize {{text}} for CI.", environments=("ci",))

    assert fetch().version == "v2"


def test_a_pinned_version(prompt_library: PromptLibrary) -> None:
    prompt_library.add("summary", "one {{text}}")
    prompt_library.add("summary", "two {{text}}", environments=("production",))

    prompt = fetch(version="v1")

    assert (prompt.version, prompt.template.template) == ("v1", "one {{text}}")


def test_environment_and_version_together_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        fetch(environment="production", version="v1")


def test_a_missing_label_raises(prompt_library: PromptLibrary) -> None:
    prompt_library.add("summary", "Summarize {{text}}.")

    with pytest.raises(LookupError, match="environment production"):
        fetch(environment="production")


def test_a_chat_prompt_renders_every_message(prompt_library: PromptLibrary) -> None:
    prompt_library.add("tutor", CHAT, environments=("production",))

    prompt = get_prompt(
        "tutor",
        project_name=PROJECT,
        default=ChatTemplate(messages=[]),
        environment="production",
    )

    assert prompt.template.messages == CHAT
    assert prompt.format(subject="physics", text="entropy") == [
        {"role": "system", "content": "You summarize physics."},
        {"role": "user", "content": "entropy"},
    ]


def test_a_default_of_the_other_structure_raises(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("tutor", CHAT, environments=("production",))

    with pytest.raises(PromptTemplateStructureMismatch):
        fetch("tutor", environment="production")


def test_a_jinja2_prompt_renders_the_same_fetched_or_embedded(
    prompt_library: PromptLibrary,
) -> None:
    template = "{% for item in items %}- {{ item }}\n{% endfor %}"
    prompt_library.add("summary", template, type="jinja2", environments=("production",))
    embedded = TextTemplate(template=template, type="jinja2")

    fetched = fetch(default=embedded, environment="production")

    assert fetched.template.type == "jinja2"
    assert fetched.format(items=["a", "b"]) == "- a\n- b\n"
    assert embedded.format(items=["a", "b"]) == "- a\n- b\n"


@pytest.mark.parametrize(
    "template",
    [TextTemplate(template="Hi {{name}}."), ChatTemplate(messages=CHAT)],
)
def test_mustache_variables_must_match_the_placeholders(
    template: TextTemplate | ChatTemplate,
) -> None:
    with pytest.raises(PromptPlaceholdersDontMatchFormatArguments):
        template.format(nobody="here")


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
    monkeypatch.setattr(prompts, "_fetch", unreachable(exc))

    prompt = fetch(environment="production")

    assert prompt.source == "embedded"
    assert prompt.template == DEFAULT
    assert prompt.version == "v1"
    assert prompt.degraded_reasons == ("embedded_prompt",)


def test_a_refusal_from_opik_is_not_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prompts, "_fetch", unreachable(ApiError(status_code=401, body="no"))
    )

    with pytest.raises(ApiError):
        fetch(environment="production")


def keycloak_answering(status: int) -> object:
    """A fetch whose token request gets ``status`` from Keycloak."""
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

    def fetch_with_token(_lookup: object) -> None:
        tokens.token()

    return fetch_with_token


def test_a_keycloak_refusal_on_a_cold_fetch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts, "_fetch", keycloak_answering(401))

    with pytest.raises(httpx.HTTPStatusError, match="401"):
        fetch(environment="production")


def test_keycloak_down_on_a_cold_fetch_is_an_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts, "_fetch", keycloak_answering(503))

    assert fetch(environment="production").degraded_reasons == ("embedded_prompt",)


def test_a_bad_url_is_not_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prompts, "_fetch", unreachable(httpx.UnsupportedProtocol("opik://"))
    )

    with pytest.raises(httpx.UnsupportedProtocol):
        fetch(environment="production")


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
    assert fetch(environment="production").source == "opik"


def test_the_worker_sees_the_callers_masks_but_not_its_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    def record(_lookup: object) -> None:
        seen.extend(
            [mask_context.get_active_prompt_masks(), context_storage.get_trace_data()]
        )

    monkeypatch.setattr(prompts, "_fetch", record)

    @opik.track(project_name=PROJECT)
    def handler() -> None:
        with (
            mask_context.prompt_mask_context({"prompt-id": "mask-id"}),
            pytest.raises(LookupError),
        ):
            fetch(environment="production")

    with opik.record_traces_locally():
        handler()

    assert seen == [{"prompt-id": "mask-id"}, None]


def test_a_slow_cold_fetch_falls_back_and_then_fills_the_cache(
    monkeypatch: pytest.MonkeyPatch, prompt_library: PromptLibrary
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    prompt_library.delay_seconds = 0.5
    monkeypatch.setattr(prompts, "COLD_FETCH_DEADLINE_SECONDS", 0.05)

    first = fetch(environment="production")
    deadline = time.monotonic() + 5
    while prompts._inflight and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.05)
    second = fetch(environment="production")

    assert first.degraded_reasons == ("embedded_prompt",)
    assert (second.source, second.degraded_reasons) == ("opik", ())
    assert prompt_library.retrievals == 1


def test_a_cached_prompt_is_served_while_opik_is_down(
    monkeypatch: pytest.MonkeyPatch, prompt_library: PromptLibrary
) -> None:
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))
    fetch(environment="production")
    prompt_library.reset()
    monkeypatch.setattr(prompts, "COLD_FETCH_DEADLINE_SECONDS", 0.05)
    prompt_library.delay_seconds = 1

    prompt = fetch(environment="production")

    assert (prompt.source, prompt.version) == ("opik", "v1")
    assert prompt_library.retrievals == 0


def test_with_tracing_off_the_embedded_default_is_used_unmarked(
    prompt_library: PromptLibrary,
) -> None:
    tracing.reset()
    configure_tracing(TracingConfig(enabled=False))
    prompt_library.add("summary", "Summarize {{text}}.", environments=("production",))

    prompt = fetch(environment="production")

    assert (prompt.source, prompt.degraded_reasons) == ("embedded", ())
    assert prompt_library.retrievals == 0


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
    monkeypatch.setattr(prompts, "_fetch", unreachable(httpx.ConnectError("down")))
    prompt_library.reset()
    fallback = fetch("other", environment="production")
    edited = fetch(default=TextTemplate(template="Summarize {{text}}!"), version="v9")

    assert fetched.source != fallback.source
    assert fetched.template_hash == fallback.template_hash
    assert fetched.template_hash != edited.template_hash


def test_create_prompt_adds_a_version_only_on_change(
    prompt_library: PromptLibrary,
) -> None:
    template = TextTemplate(template="Summarize {{text}}.")

    first = create_prompt("summary", template, project_name=PROJECT)
    again = create_prompt("summary", template, project_name=PROJECT)
    changed = create_prompt(
        "summary", TextTemplate(template="Summarize {{text}}!"), project_name=PROJECT
    )
    retyped = create_prompt(
        "summary",
        TextTemplate(template="Summarize {{text}}!", type="jinja2"),
        project_name=PROJECT,
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
        "summary", TextTemplate(template="Summarize {{text}}."), project_name=PROJECT
    )

    assert created.version == "v1"
    assert prompt_library.versions("summary") == ["v1"]


def test_create_prompt_for_a_chat_prompt(prompt_library: PromptLibrary) -> None:
    template = ChatTemplate(messages=CHAT)

    create_prompt("tutor", template, project_name=PROJECT)
    created = create_prompt("tutor", template, project_name=PROJECT)

    assert created.template.messages == CHAT
    assert prompt_library.versions("tutor") == ["v1"]


def test_promote_keeps_the_versions_other_labels(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.environments.add("qa")
    prompt_library.add("summary", "one", environments=("production",))
    prompt_library.add("summary", "two", environments=("qa",))

    promote_prompt(
        "summary", version="v2", environment="production", project_name=PROJECT
    )

    assert prompt_library.labels("summary") == {"v1": [], "v2": ["qa", "production"]}


def test_promote_reaches_the_next_fetch_in_this_process(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "one {{text}}", environments=("production",))
    prompt_library.add("summary", "two {{text}}")
    assert fetch(environment="production").version == "v1"

    promote_prompt(
        "summary", version="v2", environment="production", project_name=PROJECT
    )

    assert fetch(environment="production").version == "v2"


def test_promote_to_an_unregistered_environment_raises(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "one")

    with pytest.raises(EnvironmentNotFoundError):
        promote_prompt("summary", version="v1", environment="ci", project_name=PROJECT)
