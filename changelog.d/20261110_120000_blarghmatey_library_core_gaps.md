### Fixed

- When every model in a fallback chain fails on a request that continues a suspended response (OpenAI background mode), the caller gets the primary's error. It got the error of the member holding the suspended job, which is tried first.
