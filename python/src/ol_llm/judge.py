"""The judge model evaluation metrics call, through this library.

A judge is catalog-checked, routed, retried and traced like any other call, under
the ``eval-judge`` use case (lifecycle spec §5.1 item 7). The judge metrics in
``ol_llm.evaluation`` take one, and so does ``judge_pairwise``.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pydantic
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from ol_llm.catalog import load_catalog
from ol_llm.policy import UseCase
from ol_llm.runs import LLMResult, complete, complete_sync

JUDGE_USE_CASE = "eval-judge"
"""The use case every judge call is attributed to (spec §4.2)."""

ChatMessage = Mapping[str, Any]
"""One turn of a judge conversation in OpenAI's chat shape: ``role`` and
``content``."""


def judge_model(app: str, model: str) -> "JudgeModel":
    """Return a judge that calls through ol-llm under the ``eval-judge`` use case.

    Pass it to a judge metric, e.g. ``hallucination(judge_model("learn-ai",
    "openai:gpt-4o-mini"))``. The judge should not be the model under test (spec
    §5.1 item 7).

    :param app: the app being evaluated; the judge's calls are traced and costed
        under it.
    :param model: the judge's model id, in ``provider:model`` form.
    :returns: the judge.
    :rtype: JudgeModel
    :raises ValueError: if the catalog does not list the model, or lists it as
        retired or as an embedding model. Unlike a call site, a judge may not use
        an uncataloged model: its spend is shared evaluation cost, and the catalog
        is where that is priced.
    """
    return JudgeModel(
        UseCase(app=app, name=JUDGE_USE_CASE, profile="batch", model=model)
    )


class JudgeModel:
    """A model that answers a judge's prompt through ``complete``.

    A judge metric sends a system and a user message and, for a structured
    verdict, a pydantic model as ``response_format``. The system messages become
    the instructions, the rest the conversation, and a structured verdict comes
    back as its JSON, which is what the metrics parse.

    Build it with ``judge_model``, which names the use case ``eval-judge``.

    :param use_case: the judge's use case.
    :raises ValueError: if the catalog does not list the use case's model, or
        lists it as retired or as an embedding model.
    """

    def __init__(self, use_case: UseCase) -> None:
        """Check the model against the catalog; it is built on the first call."""
        if load_catalog().buildable(use_case.model, kind="chat") is None:
            msg = (
                f"{use_case.model!r} is not in the model catalog; a judge must use "
                "a listed model"
            )
            raise ValueError(msg)
        self.use_case = use_case

    @property
    def model_name(self) -> str:
        """The judge's model id."""
        return self.use_case.model

    def generate_string(
        self,
        input: str,  # noqa: A002 - the name judge metrics pass it by
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Answer one user message.

        :param input: the message.
        :param response_format: the verdict's model, for a structured answer.
        :returns: the answer, as JSON for a structured one.
        :rtype: str
        :raises TypeError: if given provider options.
        :raises LLMError: on a failed call.
        """
        _no_options(kwargs)
        return _content(
            complete_sync(self.use_case, input, output_type=response_format or str)
        )

    async def agenerate_string(
        self,
        input: str,  # noqa: A002 - the name judge metrics pass it by
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Answer one user message; see ``generate_string``."""
        _no_options(kwargs)
        result = await complete(
            self.use_case, input, output_type=response_format or str
        )
        return _content(result)

    def generate_chat_completion(
        self,
        messages: Sequence[ChatMessage],
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, str]:
        """Answer a conversation.

        :param messages: the conversation, in OpenAI's chat shape.
        :param response_format: the verdict's model, for a structured answer.
        :returns: the assistant's turn.
        :rtype: dict[str, str]
        :raises ValueError: on a tool message, or a conversation that does not
            end with a user message.
        :raises TypeError: if given provider options.
        :raises LLMError: on a failed call.
        """
        _no_options(kwargs)
        instructions, history = _conversation(messages)
        result = complete_sync(
            self.use_case,
            history,
            output_type=response_format or str,
            instructions=instructions,
        )
        return {"role": "assistant", "content": _content(result)}

    async def agenerate_chat_completion(
        self,
        messages: Sequence[ChatMessage],
        response_format: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, str]:
        """Answer a conversation; see ``generate_chat_completion``."""
        _no_options(kwargs)
        instructions, history = _conversation(messages)
        result = await complete(
            self.use_case,
            history,
            output_type=response_format or str,
            instructions=instructions,
        )
        return {"role": "assistant", "content": _content(result)}

    def respond(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,  # noqa: ANN401
    ) -> LLMResult[str]:
        """Answer a conversation and return the whole result.

        :param messages: the conversation, in OpenAI's chat shape.
        :returns: the result.
        :rtype: LLMResult
        :raises TypeError: if given provider options.
        """
        _no_options(kwargs)
        instructions, history = _conversation(messages)
        return complete_sync(self.use_case, history, instructions=instructions)

    async def arespond(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,  # noqa: ANN401
    ) -> LLMResult[str]:
        """Answer a conversation and return the whole result; see ``respond``."""
        _no_options(kwargs)
        instructions, history = _conversation(messages)
        return await complete(self.use_case, history, instructions=instructions)


def _no_options(kwargs: Mapping[str, Any]) -> None:
    # Sampling options belong to the use case and the call policy, not to each
    # metric; judge metrics pass none to a model they did not build.
    if kwargs:
        msg = f"JudgeModel takes no provider options, got {sorted(kwargs)}"
        raise TypeError(msg)


def _conversation(
    messages: Sequence[ChatMessage],
) -> tuple[str | None, list[ModelMessage]]:
    """Split a chat conversation into instructions and PydanticAI messages."""
    instructions: list[str] = []
    history: list[ModelMessage] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            instructions.append(message["content"])
        elif role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(message["content"])]))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(message["content"])]))
        else:
            msg = f"A judge conversation has no {role!r} messages"
            raise ValueError(msg)
    if not history or not isinstance(history[-1], ModelRequest):
        msg = "A judge conversation must end with a user message"
        raise ValueError(msg)
    return "\n\n".join(instructions) or None, history


def _content(result: LLMResult[Any]) -> str:
    output = result.output
    if isinstance(output, pydantic.BaseModel):
        return output.model_dump_json()
    return cast("str", output)
