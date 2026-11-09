### Fixed

- `ol_llm.evaluation.evaluate` puts `Settings.environment` on each item's trace and on the task's and judges' calls under it. `opik.evaluate` opens those traces with no environment and Opik gives nested spans their parent's, so experiment task and judge spend sat in Opik's default environment instead of the app's, and Opik logged a "nested span under the environment" warning for every call.
