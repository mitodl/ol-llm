### Added

- Trace destinations, chosen by name like memory backends: `OL_LLM_TRACE_DESTINATIONS` (comma-separated) is `opik` by default, `otlp` sends the spans unmapped to the endpoint the standard `OTEL_EXPORTER_OTLP_*` variables name (e.g. a Tempo), and `memory` keeps them in the process for tests. `TracingConfig.destinations` sets them in code, `ol_llm.tracing.destinations()` returns them, and the `ol_llm.trace_destinations` entry-point group or `register_trace_destination` adds others.
- `otlp` receives the `ol.*` and `gen_ai.*` attributes (and no others but the content ones), and none that carry learner content or identify a user (call input and output, model messages, embedding inputs, note text and queries, `ol.user_id`, error messages) unless `OL_LLM_TRACE_OTLP_CONTENT=true`.
- Every entry-point call opens an OpenTelemetry call span carrying the attribution fields as `ol.*` attributes (`ol.app`, `ol.use_case`, `ol.environment`, `ol.user_id`, `ol.run_id`, `ol.prompt_name`, `ol.degraded_reason`, ...), `ol.tags`, and the thread as `gen_ai.conversation.id`. The names are in `ol_llm.trace_attributes`.

### Changed

- In Opik, span attributes ol-llm records that Opik has no rule for now show under the span's metadata instead of its input: the memory spans' app, use case, environment, user, run and `memory.*` keys and counts, and the embedding spans' input count, input type, settings and vector dimensions (`opik.metadata.inputs_count`, `opik.metadata.input_type`, `opik.metadata.embedding_settings`, `opik.metadata.embedding_dimensions`). Note text, queries and results still show as input. The `ol_llm.opik_project` span attribute is gone; each span names its project with Opik's own `opik.project_name`, so a span of a caller's trace no longer needs its own export request.
- The memory spans' attributes are renamed: `app`, `use_case`, `environment`, `user_id` and `run_id` are `ol.app` and so on, `thread_id` is `gen_ai.conversation.id`, and the `memory.*` keys and counts are `ol.memory.*`. Content attributes (`memory.text`, `memory.query`, `memory.metadata`, `memory.results`) keep their names. The embedding span's `inputs_count` is `ol.inputs_count`.
- `OPIK_TRACK_DISABLE` turns tracing off only when `opik` is a destination.
- Configuring Keycloak alongside `OPIK_API_KEY` is refused when the Opik destination is built or `ol_llm.tracing.opik_client` is first called, not by `TracingConfig.from_env`.
- The Keycloak auth classes moved to `ol_llm.keycloak`; `ol_llm.tracing` still exports them.

### Removed

- `ol_llm.tracing.PROJECT_ATTRIBUTE`, `ProjectExporter`, `ProjectStampProcessor`, `OpikActiveSampler` (now `ActiveSampler`), `ol_llm.traces.error_info`, and `TracedCall.mark_trace` (the trace destinations mark the call). `OTLP_TRACES_PATH`, `PROJECT_HEADER` and `WORKSPACE_HEADER` moved from `ol_llm.tracing` to `ol_llm.trace_destinations.opik_destination`. `ChatTracer`, `GRAPH_METADATA_KEY` and `CUT_OFF_METADATA_KEY` moved from `ol_llm.langchain` to `ol_llm.trace_destinations.opik_langchain`.
- Model spans no longer take their Opik trace from W3C baggage, or from `opik.trace_id` and `opik.span_id` attributes on a parent span that another tracer provider started, as the Opik SDK's `OpikSpanProcessor` did. They attach to the parent ol-llm span, else to the Opik SDK's current span (a caller's `@opik.track`, including one called with `opik_distributed_trace_headers`). No app used either path.
