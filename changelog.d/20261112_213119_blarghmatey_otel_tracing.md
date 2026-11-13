### Added

- Trace destinations, chosen by name like memory backends: `OL_LLM_TRACE_DESTINATIONS` (comma-separated) is `opik` by default, `otlp` sends the spans unmapped to the endpoint the standard `OTEL_EXPORTER_OTLP_*` variables name (e.g. a Tempo), and `memory` keeps them in the process for tests. `TracingConfig.destinations` sets them in code, `ol_llm.tracing.destinations()` returns them, and the `ol_llm.trace_destinations` entry-point group or `register_trace_destination` adds others.
- Every entry-point call opens an OpenTelemetry call span carrying the attribution fields as `ol.*` attributes (`ol.app`, `ol.use_case`, `ol.environment`, `ol.user_id`, `ol.run_id`, `ol.prompt_name`, `ol.degraded_reason`, ...), `ol.tags`, and the thread as `gen_ai.conversation.id`. The names are in `ol_llm.trace_attributes`.

### Changed

- In Opik, span attributes ol-llm records that Opik has no rule for now show under the span's metadata instead of its input: the memory spans' app, use case, environment, user, run and `memory.*` keys and counts, and the embedding spans' input count and vector dimensions (`opik.metadata.inputs_count`, `opik.metadata.embedding_dimensions`). Note text, queries and results still show as input. The `ol_llm.opik_project` span attribute is gone; each span names its project with Opik's own `opik.project_name`, so all of an app's spans go in one export request.
- The memory spans' attributes are renamed: `app`, `use_case`, `environment`, `user_id` and `run_id` are `ol.app` and so on, `thread_id` is `gen_ai.conversation.id`, and the `memory.*` keys and counts are `ol.memory.*`. Content attributes (`memory.text`, `memory.query`, `memory.metadata`, `memory.results`) keep their names. The embedding span's `inputs_count` is `ol.inputs_count`.
- The Keycloak auth classes moved to `ol_llm.keycloak`; `ol_llm.tracing` still exports them.

### Removed

- `ol_llm.tracing.PROJECT_ATTRIBUTE`, `ProjectExporter`, `ProjectStampProcessor`, `OpikActiveSampler` (now `ActiveSampler`), `OTLP_TRACES_PATH` and `PROJECT_HEADER`, and `ol_llm.traces.error_info`. `ChatTracer`, `GRAPH_METADATA_KEY` and `CUT_OFF_METADATA_KEY` moved from `ol_llm.langchain` to `ol_llm.trace_destinations.opik_langchain`.
