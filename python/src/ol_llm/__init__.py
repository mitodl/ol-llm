"""Shared library for LLM calls at MIT Open Learning.

The Opik SDK starts its own Sentry client on import, reporting to Comet's DSN, and
that replaces whatever Sentry client the host application already configured.
Setting OPIK_SENTRY_ENABLE before anything imports opik stops it. This module
does that first, so every consumer gets it by importing ol_llm before opik.
"""

import os
from importlib.metadata import version

os.environ.setdefault("OPIK_SENTRY_ENABLE", "false")

# The version lives only in python/pyproject.toml, which is what a release bumps.
__version__ = version("ol-llm")
