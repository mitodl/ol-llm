"""Use cases and the timeout and retry policy every model call runs under.

The values and their resolution order are the error contract (spec §3.2 rule 1) as
turned into types by the call contract (spec §6.2).
"""

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, TypeVar

Profile = Literal["request", "batch"]

_ENV_PREFIX = "OL_LLM"
_RESERVED_NAMES = frozenset({"request", "batch"})
_NON_ENV_CHARS = re.compile(r"[^A-Z0-9]")


@dataclass(frozen=True, kw_only=True)
class CallPolicy:
    """The resolved limits for one model call.

    :param timeout_seconds: bound on one attempt; on a stream, on the gap between
        chunks.
    :param max_retries: attempts after the first, on retryable failures only.
    :param max_backoff_seconds: on the request profile, the total time spent
        waiting between attempts; on the batch profile, the cap on each wait.
    """

    timeout_seconds: float
    max_retries: int
    max_backoff_seconds: float


PROFILE_DEFAULTS: Mapping[Profile, CallPolicy] = {
    "request": CallPolicy(timeout_seconds=60, max_retries=2, max_backoff_seconds=10),
    "batch": CallPolicy(timeout_seconds=300, max_retries=5, max_backoff_seconds=300),
}


@dataclass(frozen=True, kw_only=True)
class UseCase:
    """One LLM call site, declared once as a module-level constant.

    :param app: the application, e.g. ``learn-ai``. Becomes the Opik project.
    :param name: the use case, e.g. ``course-translation``. Names the environment
        overrides, so it may not be ``request`` or ``batch``.
    :param profile: ``request`` for a user waiting on the answer, ``batch`` for
        pipelines and background jobs.
    :param model: the model id in PydanticAI form, ``provider:model``.
    :param fallback_models: model ids tried in order once the primary has spent its
        retries on a rate limit or an outage. Empty means no fallback.
    :param timeout_seconds: overrides the profile default; needs ``policy_reason``.
    :param max_retries: overrides the profile default; needs ``policy_reason``.
    :param max_backoff_seconds: overrides the profile default; needs
        ``policy_reason``.
    :param policy_reason: why this use case departs from its profile's defaults.
    :param output_retries: validation retries on the same model before the call
        fails with ``LLMInvalidOutput``.
    :param request_limit: model requests allowed per run.
    :param cost_limit: dollar ceiling per run. Unset in phase one, where budgets
        are alerts only.
    """

    app: str
    name: str
    profile: Profile
    model: str
    fallback_models: Sequence[str] = ()
    timeout_seconds: float | None = None
    max_retries: int | None = None
    max_backoff_seconds: float | None = None
    policy_reason: str | None = None
    output_retries: int = 1
    request_limit: int | None = 50
    cost_limit: Decimal | None = None
    _env_name: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the declaration so a bad one fails at import, not on first call.

        :raises ValueError: on an empty or reserved name, an unknown profile, a
            policy override without a reason, or an out-of-range value.
        """
        if not self.app:
            msg = "UseCase.app must not be empty"
            raise ValueError(msg)
        if not self.name:
            msg = "UseCase.name must not be empty"
            raise ValueError(msg)
        if self.name in _RESERVED_NAMES:
            msg = f"UseCase.name may not be {self.name!r}, which is a profile name"
            raise ValueError(msg)
        if self.profile not in PROFILE_DEFAULTS:
            msg = f"UseCase.profile must be 'request' or 'batch', not {self.profile!r}"
            raise ValueError(msg)
        overrides = (self.timeout_seconds, self.max_retries, self.max_backoff_seconds)
        if any(value is not None for value in overrides) and not self.policy_reason:
            msg = (
                f"Use case {self.name!r} overrides its profile's timeout or retry "
                "defaults, so it must give a policy_reason"
            )
            raise ValueError(msg)
        if self.output_retries < 0:
            msg = "UseCase.output_retries must not be negative"
            raise ValueError(msg)
        object.__setattr__(self, "fallback_models", tuple(self.fallback_models))
        object.__setattr__(
            self, "_env_name", _NON_ENV_CHARS.sub("_", self.name.upper())
        )
        _check_policy_values(
            f"use case {self.name!r}",
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_backoff_seconds=self.max_backoff_seconds,
        )

    @property
    def model_ids(self) -> tuple[str, ...]:
        """The primary model followed by its fallbacks, in the order tried."""
        return (self.model, *self.fallback_models)

    @property
    def env_name(self) -> str:
        """The use case name as it appears in ``OL_LLM_<USE_CASE>_<SETTING>``."""
        return self._env_name


def resolve_policy(
    use_case: UseCase, environ: Mapping[str, str] | None = None
) -> CallPolicy:
    """Resolve a use case's timeout and retry values; later sources win.

    1. the profile default;
    2. a value set on the use case;
    3. ``OL_LLM_<PROFILE>_<SETTING>``;
    4. ``OL_LLM_<USE_CASE>_<SETTING>``.

    Call sites not built on PydanticAI apply the result to their own client.

    :param use_case: the call site.
    :param environ: the environment to read; ``os.environ`` when omitted.
    :returns: the effective policy.
    :rtype: CallPolicy
    :raises ValueError: if an environment value does not parse or is out of range.
    """
    env = os.environ if environ is None else environ
    default = PROFILE_DEFAULTS[use_case.profile]
    scopes = (use_case.profile.upper(), use_case.env_name)

    timeout = _env_override(env, scopes, "TIMEOUT_SECONDS", float)
    if timeout is None:
        timeout = (
            default.timeout_seconds
            if use_case.timeout_seconds is None
            else use_case.timeout_seconds
        )
    retries = _env_override(env, scopes, "MAX_RETRIES", int)
    if retries is None:
        retries = (
            default.max_retries
            if use_case.max_retries is None
            else use_case.max_retries
        )
    backoff = _env_override(env, scopes, "MAX_BACKOFF_SECONDS", float)
    if backoff is None:
        backoff = (
            default.max_backoff_seconds
            if use_case.max_backoff_seconds is None
            else use_case.max_backoff_seconds
        )
    _check_policy_values(
        f"the resolved policy for {use_case.name!r}",
        timeout_seconds=timeout,
        max_retries=retries,
        max_backoff_seconds=backoff,
    )
    return CallPolicy(
        timeout_seconds=timeout, max_retries=retries, max_backoff_seconds=backoff
    )


_Number = TypeVar("_Number", int, float)


def _env_override(
    env: Mapping[str, str],
    scopes: tuple[str, str],
    setting: str,
    parse: type[_Number],
) -> _Number | None:
    """Return the most specific environment value for a setting, if any is set."""
    value: _Number | None = None
    for scope in scopes:
        name = f"{_ENV_PREFIX}_{scope}_{setting}"
        if (raw := env.get(name)) is None:
            continue
        try:
            value = parse(raw)
        except ValueError as exc:
            msg = f"{name}={raw!r} is not a valid {parse.__name__}"
            raise ValueError(msg) from exc
    return value


def _check_policy_values(
    source: str,
    *,
    timeout_seconds: float | None,
    max_retries: int | None,
    max_backoff_seconds: float | None,
) -> None:
    if timeout_seconds is not None and not (
        math.isfinite(timeout_seconds) and timeout_seconds > 0
    ):
        msg = f"timeout_seconds for {source} must be a positive number"
        raise ValueError(msg)
    if max_retries is not None and max_retries < 0:
        msg = f"max_retries for {source} must not be negative"
        raise ValueError(msg)
    if max_backoff_seconds is not None and not (
        math.isfinite(max_backoff_seconds) and max_backoff_seconds >= 0
    ):
        msg = f"max_backoff_seconds for {source} must be a non-negative number"
        raise ValueError(msg)
