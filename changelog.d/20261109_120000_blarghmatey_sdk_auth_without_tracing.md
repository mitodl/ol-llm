### Fixed

- With tracing off (`OPIK_TRACK_DISABLE=true` or `TracingConfig(enabled=False)`), Opik SDK calls made after tracing is configured (`fetch_prompt`, `create_prompt`, `promote_prompt`, datasets, experiments) now send the Keycloak token when the `OPIK_KEYCLOAK_*` client is configured. Before, ol-llm registered the SDK's auth hook only with tracing on, so these calls reached Opik without credentials and got a 401. Tracing off still opens no root trace and exports no spans, and `get_prompt` still returns the embedded default without asking Opik.
