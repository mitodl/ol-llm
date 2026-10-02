import importlib
import os
import sys

import pytest


def test_importing_ol_llm_disables_opik_sentry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPIK_SENTRY_ENABLE", raising=False)
    sys.modules.pop("ol_llm", None)

    importlib.import_module("ol_llm")

    assert os.environ["OPIK_SENTRY_ENABLE"] == "false"


def test_explicit_opik_sentry_setting_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPIK_SENTRY_ENABLE", "true")
    sys.modules.pop("ol_llm", None)

    importlib.import_module("ol_llm")

    assert os.environ["OPIK_SENTRY_ENABLE"] == "true"
