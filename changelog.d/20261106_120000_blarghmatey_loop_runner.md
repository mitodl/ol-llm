### Added

- `ol_llm.run_coroutine(coro)` runs a coroutine on one long-lived event loop per thread that ol-llm owns, so the provider clients built for that loop are reused by every later call in the thread. Sync code that awaits ol-llm should use it in place of `asyncio.run` or `async_to_sync`, which start a loop per call and leave a client behind per call. It refuses to run inside a running loop, and an interrupted coroutine is cancelled and cleaned up before the exception propagates.
- `ol_llm.close_thread_loop()` closes the calling thread's loop and the provider clients built for it. The main thread's loop closes at interpreter exit.

### Changed

- `run_sync`, `complete_sync`, and `embed_sync` run on that per-thread loop instead of PydanticAI's `run_sync`, so a Celery task or Dagster asset reuses one set of provider clients per worker thread.

### Fixed

- A process forked after ol-llm was used (a Celery prefork worker whose app configured tracing at startup) no longer shares the parent's sockets or waits on its dead threads. The child sets the parent's event loop and provider clients aside without closing them, builds its own Bedrock client, gets new HTTP clients for the Keycloak token endpoint and the span exporter, and a new Opik SDK client, whose sender threads did not survive the fork. Before, `ol_llm.tracing.flush()` in the child hung waiting on the parent's sender threads.
