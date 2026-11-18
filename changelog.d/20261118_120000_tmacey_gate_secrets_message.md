### Fixed

- The evaluation gate and prompt promotion workflows no longer blame a fork when the Opik secrets are missing on a run that is not a fork pull request (a same-repository pull request, a push, a schedule). They now say the calling repository has not passed `OPIK_KEYCLOAK_CLIENT_ID` and `OPIK_KEYCLOAK_CLIENT_SECRET`, and keep the fork message for fork pull requests. Both still fail the job. Apps get this once they bump the commit their workflow pins.
