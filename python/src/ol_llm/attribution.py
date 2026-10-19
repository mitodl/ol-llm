"""The attribution schema (spec §4): the fields every LLM call carries.

``app`` and ``use_case`` come from the ``UseCase``, ``environment`` from ``Settings``,
and the per-call fields from ``Attribution``. Model and provider are emitted by
PydanticAI on its spans. Tags carry only the low-cardinality fields; identifiers go
in metadata and never in tags or metric labels.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ol_llm.policy import CallPolicy, UseCase

DegradedReason = Literal["fallback_model", "embedded_prompt", "partial_output"]

USE_CASE_HEADER = "X-OL-LLM-Use-Case"
"""Sent on every gateway request so the gateway can label its token metrics."""

MetadataValue = str | int | float | bool


@dataclass(frozen=True, kw_only=True)
class Attribution:
    """The per-call attribution fields.

    :param user_id: a pseudonymous, stable user id: the platform's global user id
        for a signed-in user, or ``anon-`` plus a keyed hash of the session for an
        anonymous one. Never an email, username, or raw session key.
    :param thread_id: the conversation, for chat.
    :param run_id: the batch run, e.g. a Dagster run id or Celery task id.
    """

    user_id: str | None = None
    thread_id: str | None = None
    run_id: str | None = None


NO_ATTRIBUTION = Attribution()


@dataclass(frozen=True, kw_only=True)
class TraceFields:
    """What the root trace of a run is opened with.

    :param project_name: the Opik project, which is the app.
    :param environment: Opik's native environment field.
    :param tags: ``key:value`` filter tags.
    :param metadata: everything else, as ``opik.metadata.*``.
    :param thread_id: Opik's native thread field, for conversations.
    """

    project_name: str
    environment: str
    tags: Sequence[str]
    metadata: Mapping[str, MetadataValue]
    thread_id: str | None


def trace_fields(  # noqa: PLR0913
    use_case: UseCase,
    *,
    environment: str,
    policy: CallPolicy,
    attribution: Attribution = NO_ATTRIBUTION,
    model_tags: Sequence[str] = (),
    prompt_name: str | None = None,
    prompt_version: str | None = None,
) -> TraceFields:
    """Build the root-trace fields for one run.

    The resolved policy goes in metadata so an environment override is visible on
    the trace.

    :param use_case: the call site.
    :param environment: the deployment environment.
    :param policy: the policy the run resolved to.
    :param attribution: the per-call fields.
    :param model_tags: tags for the models' catalog status, from
        ``ol_llm.models.catalog_tags``.
    :param prompt_name: the managed prompt the run uses, if any.
    :param prompt_version: that prompt's version, when known.
    :returns: the fields to open the root trace with.
    :rtype: TraceFields
    """
    metadata: dict[str, MetadataValue] = {
        "app": use_case.app,
        "use_case": use_case.name,
        "environment": environment,
        "policy_timeout_seconds": policy.timeout_seconds,
        "policy_max_retries": policy.max_retries,
        "policy_max_backoff_seconds": policy.max_backoff_seconds,
    }
    if attribution.user_id is not None:
        metadata["user_id"] = attribution.user_id
    if attribution.run_id is not None:
        metadata["run_id"] = attribution.run_id
    if prompt_name is not None:
        metadata["prompt_name"] = prompt_name
    if prompt_version is not None:
        metadata["prompt_version"] = prompt_version
    return TraceFields(
        project_name=use_case.app,
        environment=environment,
        tags=(
            f"app:{use_case.app}",
            f"use_case:{use_case.name}",
            f"env:{environment}",
            *model_tags,
        ),
        metadata=metadata,
        thread_id=attribution.thread_id,
    )


def degraded_metadata(
    reasons: Sequence[DegradedReason],
) -> Mapping[str, MetadataValue]:
    """Build the trace metadata that marks a degraded result.

    :param reasons: why the result is degraded; empty when it is not.
    :returns: ``degraded`` and ``degraded_reason`` when there is a reason, else
        nothing.
    :rtype: Mapping[str, MetadataValue]
    """
    if not reasons:
        return {}
    return {"degraded": True, "degraded_reason": ",".join(dict.fromkeys(reasons))}
