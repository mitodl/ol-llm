import json
from pathlib import Path

import pytest
from conftest import PromptLibrary
from ol_llm import ChatTemplate, TextTemplate, get_prompt, load_default
from ol_llm.pointers import (
    DRIFT_STATUS,
    POINTER_ERROR_STATUS,
    PointerError,
    PromptPointers,
    export,
    main,
    promote,
)

PROJECT = "mit-learn"
CHAT = [
    {"role": "system", "content": "You summarize {{subject}}."},
    {"role": "user", "content": "{{text}}"},
]


def pointers(**prompts: str) -> PromptPointers:
    return PromptPointers(project=PROJECT, prompts=prompts)


def write(tmp_path: Path, **prompts: str) -> Path:
    path = tmp_path / "prompts.toml"
    lines = [f'project = "{PROJECT}"', "", "[prompts]"]
    lines += [f'{name} = "{version}"' for name, version in prompts.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# The file


def test_the_documented_pointer_file_is_valid(tmp_path: Path) -> None:
    doc = (Path(__file__).parents[2] / "docs" / "prompt-promotion.md").read_text()
    path = tmp_path / "prompts.toml"
    path.write_text(doc.split("```toml\n", 1)[1].split("```", 1)[0])

    assert PromptPointers.load(path).prompts


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('project = "x"\n[prompts]\nsummary = "7"\n', "should match pattern"),
        ('project = "x"\n[prompts]\n', "at least 1 item"),
        ('project = "x"\n[prompts]\n"../escape" = "v1"\n', "should match pattern"),
        ('project = "x"\nextra = 1\n[prompts]\nsummary = "v1"\n', "extra"),
        ("not toml", "Expected"),
    ],
)
def test_a_malformed_pointer_file_is_refused(
    tmp_path: Path, text: str, message: str
) -> None:
    path = tmp_path / "prompts.toml"
    path.write_text(text)

    with pytest.raises(PointerError, match=message):
        PromptPointers.load(path)


# Promotion


def test_promote_moves_the_label_to_each_pointed_version(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.environments.add("qa")
    prompt_library.add("summary", "one", environments=("production",))
    prompt_library.add("summary", "two", environments=("qa",))
    prompt_library.add("tutor", CHAT, environments=("production",))

    states = promote(pointers(summary="v2", tutor="v1"))

    assert [(s.name, s.labeled, s.drifted) for s in states] == [
        ("summary", "v1", True),
        ("tutor", "v1", False),
    ]
    assert prompt_library.labels("summary") == {"v1": [], "v2": ["qa", "production"]}
    assert prompt_library.labels("tutor") == {"v1": ["production"]}


def test_promote_labels_a_prompt_nothing_carried_the_label_for(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "one")

    [state] = promote(pointers(summary="v1"))

    assert state.labeled is None
    assert prompt_library.labels("summary") == {"v1": ["production"]}


def test_a_pointer_to_a_missing_version_moves_no_label(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "one", environments=("production",))
    prompt_library.add("summary", "two")
    prompt_library.add("tutor", CHAT, environments=("production",))

    with pytest.raises(PointerError, match="no tutor v4, glossary v1"):
        promote(pointers(summary="v2", tutor="v4", glossary="v1"))

    assert prompt_library.labels("summary") == {"v1": ["production"], "v2": []}


def test_promoting_to_an_unregistered_environment_says_what_to_do(
    prompt_library: PromptLibrary,
) -> None:
    prompt_library.add("summary", "one")

    with pytest.raises(PointerError, match="'qa' is not a registered environment"):
        promote(pointers(summary="v1"), environment="qa")


# Export


@pytest.mark.usefixtures("prompt_library")
def test_an_export_reads_back_as_the_embedded_default(
    prompt_library: PromptLibrary, tmp_path: Path
) -> None:
    prompt_library.add("summary", "one {{text}}")
    prompt_library.add("summary", "two {{text}}", type="jinja2")
    prompt_library.add("tutor", CHAT)

    written = export(pointers(summary="v2", tutor="v1"), tmp_path / "prompts")

    assert [p.name for p in written] == ["summary.json", "tutor.json"]
    text = load_default(
        "summary", directory=tmp_path / "prompts", template=TextTemplate
    )
    assert text == TextTemplate(template="two {{text}}", type="jinja2", version="v2")
    chat = load_default("tutor", directory=tmp_path / "prompts", template=ChatTemplate)
    assert chat.messages == CHAT
    assert chat.version == "v1"
    exported = json.loads(written[0].read_text())
    assert exported["template_hash"] == text.template_hash
    assert exported["project"] == PROJECT


@pytest.mark.usefixtures("prompt_library")
def test_the_embedded_default_hashes_like_the_version_it_came_from(
    prompt_library: PromptLibrary, tmp_path: Path
) -> None:
    prompt_library.add("summary", "one {{text}}", environments=("production",))
    export(pointers(summary="v1"), tmp_path)
    default = load_default("summary", directory=tmp_path, template=TextTemplate)

    fetched = get_prompt(
        "summary", project_name=PROJECT, default=default, environment="production"
    )

    assert fetched.source == "opik"
    assert fetched.template_hash == default.template_hash


def test_loading_a_default_of_the_other_structure_raises(
    prompt_library: PromptLibrary, tmp_path: Path
) -> None:
    prompt_library.add("tutor", CHAT)
    export(pointers(tutor="v1"), tmp_path)

    with pytest.raises(ValueError, match="exported as a chat prompt"):
        load_default("tutor", directory=tmp_path, template=TextTemplate)


# The command


def test_check_exits_2_on_a_missing_version(
    prompt_library: PromptLibrary,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_library.add("summary", "one")

    assert main(["check", str(write(tmp_path, summary="v2"))]) == POINTER_ERROR_STATUS
    assert "has no summary v2" in capsys.readouterr().err


def test_check_with_an_environment_reports_drift(
    prompt_library: PromptLibrary,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_library.add("summary", "one", environments=("production",))
    prompt_library.add("summary", "two")
    path = write(tmp_path, summary="v2")

    assert main(["check", str(path)]) == 0
    assert main(["check", str(path), "--environment", "production"]) == DRIFT_STATUS
    assert "summary: production is on v1, the pointer says v2" in (
        capsys.readouterr().out
    )


def test_the_promote_command_says_what_moved(
    prompt_library: PromptLibrary,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_library.add("summary", "one", environments=("production",))
    prompt_library.add("summary", "two")
    path = write(tmp_path, summary="v2")

    assert main(["promote", str(path)]) == 0
    assert main(["promote", str(path)]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "summary: production moved to v2 (was v1)",
        "summary: production already on v2",
    ]


def test_the_command_refuses_an_unreadable_pointer_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.toml"

    assert main(["promote", str(missing)]) == POINTER_ERROR_STATUS
    assert "not a prompt pointer file" in capsys.readouterr().err
