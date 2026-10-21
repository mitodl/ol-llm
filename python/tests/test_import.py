import importlib
import os
import subprocess
import sys

import pytest
from opik.config import OpikConfig

OPIK_DEFAULTS = ("OPIK_SENTRY_ENABLE", "OPIK_ANALYTICS_ENABLE")


@pytest.mark.parametrize("name", OPIK_DEFAULTS)
def test_importing_ol_llm_disables_opik_reporting(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.delenv(name, raising=False)
    sys.modules.pop("ol_llm", None)

    importlib.import_module("ol_llm")

    assert os.environ[name] == "false"


@pytest.mark.parametrize("name", OPIK_DEFAULTS)
def test_explicit_opik_reporting_setting_is_kept(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "true")
    sys.modules.pop("ol_llm", None)

    importlib.import_module("ol_llm")

    assert os.environ[name] == "true"


def test_opik_config_reads_analytics_off_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPIK_ANALYTICS_ENABLE", raising=False)
    sys.modules.pop("ol_llm", None)

    importlib.import_module("ol_llm")

    assert OpikConfig().analytics_enable is False


# opik never reports under pytest (opik/analytics/rules.py `_not_running_tests`), so
# whether it would report has to be asked in a process that has not imported pytest.
_REPORTING_ALLOWED = (
    "import ol_llm, opik; print(opik.analytics.reporting_allowed(), end='')"
)


@pytest.mark.parametrize(("setting", "expected"), [(None, "False"), ("true", "True")])
def test_opik_analytics_reporting_outside_pytest(
    setting: str | None, expected: str
) -> None:
    env = {k: v for k, v in os.environ.items() if k != "OPIK_ANALYTICS_ENABLE"}
    if setting is not None:
        env["OPIK_ANALYTICS_ENABLE"] = setting

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _REPORTING_ALLOWED],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == expected
