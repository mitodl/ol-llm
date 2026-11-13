"""Every OpenTelemetry attribute name ol-llm sets on a span (library spec §8.2).

The OTel GenAI conventions are Development status (open-telemetry/
semantic-conventions-genai has no release), so their names live here and nowhere
else. ``gen_ai.*`` is used where the conventions define a field; the §4
attribution fields and everything else ol-llm records go under ``ol.``. A trace
destination maps these to its backend's own names, e.g. Opik's ``opik.metadata.*``.
"""

OL_PREFIX = "ol."
"""Every ol-llm attribute that is not a GenAI convention starts with this."""

TAGS = "ol.tags"
"""The call's ``key:value`` filter tags, from ``TraceFields.tags``."""

INPUT = "ol.input"
"""The call's input as JSON, only when traces carry content."""

OUTPUT = "ol.output"
"""The call's output as JSON, only when traces carry content."""

DEGRADED = "ol.degraded"
DEGRADED_REASON = "ol.degraded_reason"
"""Set on a call whose result is degraded; see ``degraded_metadata``."""

INPUTS_COUNT = "ol.inputs_count"
"""How many texts an embedding request sent."""

EMBEDDING_INPUTS = "inputs"
"""The texts an embedding request sent, as JSON, only with content. Named as
PydanticAI's embedding span names them."""

PYDANTIC_AI_INPUTS_COUNT = "inputs_count"
PYDANTIC_AI_INPUT_TYPE = "input_type"
PYDANTIC_AI_EMBEDDING_SETTINGS = "embedding_settings"
"""How PydanticAI's own embedding span names the input count, the input type
(``query`` or ``document``), and the settings it sent."""

EXCEPTION_EVENT = "exception"
EXCEPTION_TYPE = "exception.type"
EXCEPTION_MESSAGE = "exception.message"
EXCEPTION_STACKTRACE = "exception.stacktrace"
EXCEPTION_ESCAPED = "exception.escaped"
"""OpenTelemetry's exception event and its attributes."""

CONVERSATION_ID = "gen_ai.conversation.id"
OPERATION_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"

MEMORY_OPERATION = "ol.memory.operation"
MEMORY_SCOPE = "ol.memory.scope"
MEMORY_BACKEND = "ol.memory.backend"
MEMORY_GENERAL_NAME = "ol.memory.general_name"
MEMORY_KEY = "ol.memory.key"
MEMORY_WRITER = "ol.memory.writer"
MEMORY_LIMIT = "ol.memory.limit"
MEMORY_RESULTS_COUNT = "ol.memory.results_count"
MEMORY_RESULT_KEYS = "ol.memory.result_keys"
MEMORY_REMOVED_COUNT = "ol.memory.removed_count"

MEMORY_TEXT = "memory.text"
MEMORY_METADATA = "memory.metadata"
MEMORY_QUERY = "memory.query"
MEMORY_RESULTS = "memory.results"
"""Content of a memory operation, only with content. Outside ``ol.`` so a backend
that files unknown attributes as the span's input, as Opik does, shows them
there."""


GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GEN_AI_CONTENT_PREFIXES = ("gen_ai.prompt", "gen_ai.completion")
"""PydanticAI's message content, and the older conventions' prompt and completion
attributes."""

USER_ID = "ol.user_id"


def attribution(key: str) -> str:
    """Return the attribute for one §4 field or other trace metadata key.

    :param key: the key, e.g. ``use_case`` or ``policy_timeout_seconds``.
    :returns: e.g. ``ol.use_case``.
    :rtype: str
    """
    return f"{OL_PREFIX}{key}"


CONTENT = frozenset(
    {
        INPUT,
        OUTPUT,
        EMBEDDING_INPUTS,
        MEMORY_TEXT,
        MEMORY_METADATA,
        MEMORY_QUERY,
        MEMORY_RESULTS,
        GEN_AI_INPUT_MESSAGES,
        GEN_AI_OUTPUT_MESSAGES,
        GEN_AI_SYSTEM_INSTRUCTIONS,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GEN_AI_TOOL_CALL_RESULT,
    }
)
"""Attributes that carry prompts, completions, tool arguments, or learner text."""
