### Changed

- A Bedrock `ModelErrorException` (HTTP 424 outside a stream) whose `originalStatusCode` is 400, 413, or 422 is now a rejection of that row's input in `process_batch`: it fails only that row, counts toward `max_consecutive_rejections`, and is not retried under the same key. Any other 424 still opens the circuit at once. `LLMRejected` has a new `original_status_code` (an `int`, or `None`) that holds the value for a 424, while `status_code` stays 424. The library logs the original status code and `resourceName` of each such 424 at warning level. (#89)
