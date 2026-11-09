### Added

- The README documents how uv consumers override the version caps (`name<=X`) that opik 2.2.139 and later put on each of its dependencies, which an application inherits through ol-llm's exact opik pin. Without the overrides, uv fails to resolve an application that needs a newer version than a cap (e.g. `openai==3.46.0` next to opik 2.2.143's `openai<=3.42.1`), or installs the capped version. The opik release ol-llm pins now (2.2.94) has no caps, so nothing needs to change yet. pip has no equivalent. ol-llm now bumps opik once a month to keep the caps close to current.
