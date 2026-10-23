# Changelog fragments

Each pull request that changes what a consumer sees adds one fragment here instead of
editing `CHANGELOG.md`, so concurrent pull requests never conflict on it. From the
repository root:

```sh
uv run scriv create --edit
```

Uncomment the category that fits (Added, Changed, Deprecated, Removed, Fixed, Security),
write one bullet per change from a consumer's point of view, and end it with the pull
request number once you know it, e.g. `(#27)`. A change with no effect on consumers (CI,
tests, internal refactors) needs no fragment.

The release pull request runs `uv run scriv collect` after bumping the version in
`python/pyproject.toml`. That moves every fragment into `CHANGELOG.md` under the new
version and deletes them.
