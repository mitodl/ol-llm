"""Test seams for code that calls models through ol-llm.

``override_models`` makes ``build_model`` return PydanticAI's ``TestModel`` or
``FunctionModel`` (or any model) under the library's own policy wrapper, so a test
exercises the same timeouts, retries, error classes, and fallback as production.
For an agent built at import time, use PydanticAI's own ``agent.override(model=...)``
with ``policy_model`` to keep the wrapper.

``tracing_disabled`` turns Opik tracing off, so tests and local development need no
Opik instance.

Recorded fixtures: do not record real provider responses into the repository.
Tests use ``FunctionModel`` to script responses and failures; a recorded response
carries model output, and possibly learner content, that the data-handling policy
has not yet ruled on.
"""

from collections.abc import Generator
from contextlib import contextmanager

import opik
from pydantic_ai.models import Model

from ol_llm.models import PolicyModel, override_models
from ol_llm.policy import UseCase, resolve_policy

__all__ = ["override_models", "policy_model", "tracing_disabled"]


def policy_model(use_case: UseCase, model: Model) -> PolicyModel:
    """Wrap a test model in the use case's policy, as ``build_model`` would.

    :param use_case: the call site.
    :param model: the model standing in for the provider.
    :returns: the wrapped model.
    :rtype: PolicyModel
    """
    return PolicyModel(
        model,
        use_case=use_case,
        model_id=use_case.model,
        policy=resolve_policy(use_case),
    )


@contextmanager
def tracing_disabled() -> Generator[None]:
    """Turn Opik tracing off for the duration, restoring the previous state."""
    previous = opik.is_tracing_active()
    opik.set_tracing_active(False)
    try:
        yield
    finally:
        opik.set_tracing_active(previous)
