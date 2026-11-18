### Fixed

- The evaluation gate and prompt promotion workflows no longer blame a fork when `OPIK_KEYCLOAK_CLIENT_ID` is empty on a same-repository pull request. They now say the calling repository has not passed `OPIK_KEYCLOAK_CLIENT_ID` and `OPIK_KEYCLOAK_CLIENT_SECRET`, and keep the fork message for fork pull requests. Both still fail the job. Apps get this once they bump the commit their workflow pins.
