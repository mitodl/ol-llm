### Security

- `check_chat_model` and the embedding catalog check keep at most 1,024 use cases (`CHECKED_USE_CASES_MAX`). They were unbounded, so an application that builds a use case from a request's model id kept every distinct id a caller sent for the life of the process. An unlisted or deprecated model now warns again once its use case falls out of that window.
- The catalog's unlisted-model and deprecation warnings write the model id with `repr`, so an id carrying a newline can't start a forged log line.
