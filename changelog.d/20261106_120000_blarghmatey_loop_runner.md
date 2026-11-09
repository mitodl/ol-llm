### Added

- `ol_llm.run_coroutine(coro)` runs a coroutine on one long-lived event loop per thread that ol-llm owns, so the provider clients built for that loop are reused by every later call in the thread. Sync code that awaits ol-llm should use it in place of `asyncio.run` or `async_to_sync`, which start a loop per call and leave a client behind per call. It refuses to run inside a running loop, and an interrupted coroutine is cancelled and cleaned up before the exception propagates.
- `ol_llm.close_thread_loop()` closes the calling thread's loop and the provider clients built for it. The main thread's loop closes at interpreter exit.

### Changed

- `run_sync`, `complete_sync`, and `embed_sync` run on that per-thread loop instead of PydanticAI's `run_sync`, so a Celery task or Dagster asset reuses one set of provider clients per worker thread.

### Fixed

- A process forked after ol-llm was used (a Celery prefork worker whose app configured tracing at startup) no longer shares the parent's sockets or waits on its dead threads. The child keeps every parent thread's event loop alive without closing it, drops the parent's provider clients without closing them, builds its own Bedrock client, gets new HTTP clients for the Keycloak token endpoint and the span exporter, and a new Opik SDK client, whose sender threads did not survive the fork. It also starts with fresh module locks (shared models, embedders, settings, tracing) and no in-flight prompt fetches, so a lock or fetch a parent thread held at the fork can't block it. Before, `ol_llm.tracing.flush()` in the child hung waiting on the parent's sender threads.
- Each Bedrock client is built from a boto3 session of its own instead of boto3's default session, which isn't thread-safe, so first calls from several models on different threads no longer race. Threads using one model share one client.
