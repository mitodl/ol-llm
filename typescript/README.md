# TypeScript client (reserved)

No production JavaScript or TypeScript code calls an LLM today. When one does, the client that
lives here implements the same contracts as the Python package (`../docs/`): the same
attribution fields, the same error classes, and the same prompt fetch semantics. Choosing its
client layer (OpenTelemetry JS or the Opik TypeScript SDK) is part of that work.
