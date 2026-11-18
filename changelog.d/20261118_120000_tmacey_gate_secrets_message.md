### Fixed

- The evaluation gate and prompt promotion workflows no longer blame a fork when the Opik secrets are missing on a run that is not a fork pull request (a same-repository pull request, a push, a schedule). They now say the calling workflow passed an empty `OPIK_KEYCLOAK_CLIENT_ID` or `OPIK_KEYCLOAK_CLIENT_SECRET` and point at the repository secrets its `secrets:` block maps to them, and keep the fork message for fork pull requests. Both still fail the job. Apps get this once they bump the commit their workflow pins. (#87)
