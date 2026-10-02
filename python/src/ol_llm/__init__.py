"""Shared library for LLM calls at MIT Open Learning.

The Opik SDK starts its own Sentry client on import, reporting to Comet's DSN, and
that replaces whatever Sentry client the host application already configured.
Setting OPIK_SENTRY_ENABLE before anything imports opik stops it. This module
does that first, so every consumer gets it by importing ol_llm before opik.
"""

import os

os.environ.setdefault("OPIK_SENTRY_ENABLE", "false")

__version__ = "0.0.1"
