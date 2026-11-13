"""Opik's prompt library as a prompt registry.

Labels are Opik environments, an app's namespace is its Opik project, and versions
are Opik's ``v<N>``. Every request goes to Opik's REST API through the SDK's
client, with the Keycloak token when a client is configured; the SDK's own prompt
cache is not used, since ``ol_llm.prompts`` keeps its own. Inside an Opik prompt
mask context (Opik's local runner) a fetch returns the masked version, as the
SDK's ``get_prompt`` would.
"""

import contextlib
from collections.abc import Hashable
from typing import cast

import httpx
import opik
from opik import opik_context
from opik.api_objects.prompt import BasePrompt, PromptType, mask_context
from opik.exceptions import (
    EnvironmentAlreadyExists,
    EnvironmentNotFoundError,
    PromptNotFoundError,
    PromptTemplateStructureMismatch,
)
from opik.rest_api.core import ApiError
from opik.rest_api.types import PromptVersionDetail

from ol_llm import tracing
from ol_llm.prompt_registry.interface import (
    PromptRegistry,
    RegistryUnavailableError,
    RegistryVersion,
)
from ol_llm.templates import ChatTemplate, TemplateType, TextTemplate

_TRANSPORT_FAILURES = (
    TimeoutError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
)


class OpikPromptRegistry(PromptRegistry):
    """The prompt library of the Opik that tracing is configured for."""

    @property
    def enabled(self) -> bool:
        """Whether tracing is on: with it off there is no Opik to fetch from."""
        return tracing.enabled()

    def context_key(self) -> Hashable:
        """Return the active prompt masks, which change what a fetch returns.

        :returns: the masks.
        :rtype: Hashable
        """
        return frozenset((mask_context.get_active_prompt_masks() or {}).items())

    def fetch(
        self,
        name: str,
        *,
        app: str,
        version: str | None = None,
        label: str | None = None,
    ) -> RegistryVersion | None:
        """Fetch a version from Opik, by version or by environment.

        :param name: the prompt's name.
        :param app: the Opik project.
        :param version: the version, e.g. ``v7``.
        :param label: the environment, e.g. ``production``.
        :returns: the version, or ``None`` when Opik has no such prompt, version,
            or environment label.
        :rtype: RegistryVersion | None
        :raises ValueError: unless exactly one of ``version`` and ``label`` is
            given.
        :raises RegistryUnavailableError: on a transport failure, a timeout, or a
            5xx from Opik or from Keycloak's token endpoint.
        """
        if (version is None) == (label is None):
            msg = "Pass exactly one of version and label"
            raise ValueError(msg)
        try:
            detail = self._retrieve(name, app=app, version=version, label=label)
        except Exception as exc:
            if _unreachable(exc):
                msg = f"Opik did not answer for prompt {name!r}: {exc!r}"
                raise RegistryUnavailableError(msg) from exc
            raise
        if detail is None:
            return None
        prompt = _prompt_from_detail(name, detail, app)
        return RegistryVersion(template=_template(prompt), ref=prompt)

    def move_label(self, name: str, *, app: str, version: str, label: str) -> None:
        """Move an environment label to a version, registering the environment.

        Opik's ``set_prompt_environments`` replaces a version's whole set, so the
        version's other labels are passed with the new one.

        :param name: the prompt's name.
        :param app: the Opik project.
        :param version: the version to label.
        :param label: the environment.
        :raises LookupError: if there is no such prompt or version.
        """
        client = tracing.opik_client()
        detail = client.get_prompts_client().get_prompt(
            name=name, version=version, project_name=app
        )
        if detail is None:
            msg = f"Opik project {app!r} has no prompt {name!r} at {version}"
            raise LookupError(msg)
        labels = [*(detail.environments or []), label]
        try:
            try:
                client.set_prompt_environments(
                    name, labels, version=version, project_name=app
                )
            except EnvironmentNotFoundError:
                # Opik seeds only development, staging, and production.
                with contextlib.suppress(EnvironmentAlreadyExists):
                    client.create_environment(label)
                client.set_prompt_environments(
                    name, labels, version=version, project_name=app
                )
        except PromptNotFoundError as exc:
            raise LookupError(str(exc)) from exc

    def create_version(
        self,
        name: str,
        template: TextTemplate | ChatTemplate,
        *,
        app: str,
        change_description: str | None = None,
    ) -> RegistryVersion:
        """Create a version in Opik unless the latest one has the same template.

        Metadata is not compared, so a version created in the Opik UI with the
        same text is not duplicated.

        :param name: the prompt's name.
        :param template: the template.
        :param app: the Opik project.
        :param change_description: recorded on a new version.
        :returns: the latest version, new or existing.
        :rtype: RegistryVersion
        :raises TypeError: if the prompt exists with the other structure.
        """
        client = tracing.opik_client()
        structure = "chat" if isinstance(template, ChatTemplate) else "text"
        try:
            latest = client.get_prompts_client().get_prompt(
                name=name,
                project_name=app,
                raise_if_not_template_structure=structure,
            )
        except PromptTemplateStructureMismatch as exc:
            raise TypeError(str(exc)) from exc
        if latest is not None:
            existing = _prompt_from_detail(name, latest, app)
            if _template(existing).template_hash == template.template_hash:
                return RegistryVersion(template=_template(existing), ref=existing)
        created: BasePrompt
        if isinstance(template, TextTemplate):
            created = client.create_prompt(
                name=name,
                prompt=template.template,
                type=PromptType(template.type),
                change_description=change_description,
                project_name=app,
            )
        else:
            created = client.create_chat_prompt(
                name=name,
                messages=[dict(message) for message in template.messages],
                type=PromptType(template.type),
                change_description=change_description,
                project_name=app,
            )
        return RegistryVersion(template=_template(created), ref=created)

    def link(self, ref: object) -> None:
        """Link the version to the current Opik trace and span, as the SDK does.

        :param ref: the Opik prompt this registry returned.
        """
        prompt = cast("BasePrompt", ref)
        opik_context.attach_prompt_to_current_trace(prompt)
        opik_context.attach_prompt_to_current_span(prompt)

    def _retrieve(
        self, name: str, *, app: str, version: str | None, label: str | None
    ) -> PromptVersionDetail | None:
        client = tracing.opik_client()
        detail = client.get_prompts_client().get_prompt(
            name=name, version=version, environment=label, project_name=app
        )
        if detail is None or detail.prompt_id is None:
            return detail
        if (mask_id := mask_context.get_mask_for_prompt(detail.prompt_id)) is None:
            return detail
        masked = client.rest_client.prompts.get_prompt_version_by_id(version_id=mask_id)
        # Text or chat is the prompt's, which a version read by id may not carry.
        return masked.model_copy(
            update={"template_structure": detail.template_structure}
        )


def opik_prompt(ref: object) -> BasePrompt | None:
    """Return the Opik prompt a ``ManagedPrompt.registry_ref`` is, for evaluation.

    :param ref: the reference.
    :returns: the Opik prompt, or ``None`` for an embedded default or another
        registry's version.
    :rtype: BasePrompt | None
    """
    return ref if isinstance(ref, BasePrompt) else None


def opik_version_id(ref: object) -> str | None:
    """Return the Opik id of the version a ``ManagedPrompt.registry_ref`` is.

    Opik's experiments link prompt versions by this id.

    :param ref: the reference.
    :returns: the id, or ``None`` when ``ref`` is not an Opik version.
    :rtype: str | None
    """
    prompt = opik_prompt(ref)
    return None if prompt is None else prompt.__internal_api__version_id__


def _unreachable(exc: Exception) -> bool:
    """Whether ``exc`` means Opik could not be reached, not that it said no.

    Only transport failures, timeouts, and 5xx responses count. A 4xx from Opik,
    or from Keycloak on the token request (``invalid_client`` for a wrong secret),
    and a bad URL are configuration errors: falling back on them would hide the
    error in every process, and the degraded mark could not reach Opik either,
    since span export uses the same token.
    """
    if isinstance(exc, _TRANSPORT_FAILURES):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.is_server_error
    if isinstance(exc, ApiError):
        return (
            exc.status_code is not None
            and exc.status_code >= httpx.codes.INTERNAL_SERVER_ERROR
        )
    return False


def _prompt_from_detail(
    name: str, detail: PromptVersionDetail, project_name: str
) -> BasePrompt:
    cls = opik.ChatPrompt if detail.template_structure == "chat" else opik.Prompt
    return cls.from_fern_prompt_version(name, detail, project_name=project_name)


def _template(prompt: BasePrompt) -> TextTemplate | ChatTemplate:
    template_type: TemplateType = PromptType(prompt.type).value
    if isinstance(prompt, opik.ChatPrompt):
        return ChatTemplate(
            messages=prompt.template, type=template_type, version=prompt.version
        )
    return TextTemplate(
        template=cast("opik.Prompt", prompt).prompt,
        type=template_type,
        version=prompt.version,
    )
