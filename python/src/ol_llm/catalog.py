"""The model catalog: the models applications may call, and what each costs.

The catalog is ``catalog.toml``, released with the library (usage policies spec
§2.2). ``build_model`` accepts only its ids. An application holds its own model
lists (database rows, settings) to it in CI with ``check_model_ids`` or the
``ol-llm-catalog check`` command, so a release that retires a model fails the CI of
every app still naming it. ``price_rules`` exports the prices as the Prometheus
recording rules the spend dashboard and alerts join token counters against.

Needs no extra: call sites not built on PydanticAI check their ids here too.
"""

import argparse
import json
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import date
from decimal import Decimal
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, get_args

ModelKind = Literal["chat", "embedding"]
ModelStatus = Literal["allowed", "deprecated", "retired"]
Route = Literal["direct", "gateway"]

TOKENS_PER_MTOK = 1_000_000
CATALOG_RESOURCE = "catalog.toml"
PRICE_RULE_GROUP = "ol_llm_model_prices"

# PydanticAI providers that reach the same model through another API shape. The
# catalog lists the model once, under the base provider.
_PROVIDER_ALIASES: Mapping[str, str] = {
    "openai-chat": "openai",
    "openai-responses": "openai",
    "azure-responses": "azure",
}
_PRICE_FIELDS = ("input_usd_per_mtok", "output_usd_per_mtok")


@dataclass(frozen=True, kw_only=True)
class CatalogEntry:
    """One model in the catalog. The fields are described in ``catalog.toml``.

    :raises ValueError: on a malformed id, an unknown kind, status, or route, a
        deprecated entry with no ``retires_on``, ``route_falls_back`` on a direct
        route, only one of the two Opik names, or a negative price.
    """

    id: str
    kind: ModelKind
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    status: ModelStatus = "allowed"
    retires_on: date | None = None
    floating: bool = False
    route: Route = "direct"
    route_falls_back: bool = False
    opik_provider: str | None = None
    opik_model: str | None = None
    provider: str = field(init=False, repr=False, compare=False)
    model_name: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the entry so a bad catalog fails when it loads."""
        provider, sep, name = self.id.partition(":")
        if not (provider and sep and name):
            msg = f"Catalog id {self.id!r} is not in provider:model form"
            raise ValueError(msg)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model_name", name)
        for value, choices in (
            (self.kind, ModelKind),
            (self.status, ModelStatus),
            (self.route, Route),
        ):
            if value not in get_args(choices):
                msg = f"{self.id}: {value!r} is not one of {get_args(choices)}"
                raise ValueError(msg)
        if self.status == "deprecated" and self.retires_on is None:
            msg = f"{self.id}: a deprecated entry needs retires_on"
            raise ValueError(msg)
        if self.route_falls_back and self.route != "gateway":
            msg = f"{self.id}: only a gateway route can fall back on its own"
            raise ValueError(msg)
        if (self.opik_provider is None) != (self.opik_model is None):
            msg = f"{self.id}: set opik_provider and opik_model together, or neither"
            raise ValueError(msg)
        if self.input_usd_per_mtok < 0 or self.output_usd_per_mtok < 0:
            msg = f"{self.id}: prices must not be negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class Catalog:
    """The catalog's entries by id."""

    entries: Mapping[str, CatalogEntry]

    @classmethod
    def from_toml(cls, text: str) -> "Catalog":
        """Parse a catalog in the ``catalog.toml`` format.

        :param text: the TOML document.
        :returns: the catalog.
        :rtype: Catalog
        :raises ValueError: on an invalid entry, an unknown or missing field, a
            price that is not a quoted decimal, or a duplicate id.
        """
        entries: dict[str, CatalogEntry] = {}
        for raw in tomllib.loads(text)["model"]:
            entry = _parse_entry(raw)
            if entry.id in entries:
                msg = f"Catalog id {entry.id!r} is listed twice"
                raise ValueError(msg)
            entries[entry.id] = entry
        return cls(entries)

    def get(self, model_id: str) -> CatalogEntry | None:
        """Return the entry for a model id, or ``None`` when it is not listed.

        ``openai-chat:``, ``openai-responses:``, and ``azure-responses:`` ids find
        the entry of the base provider.

        :param model_id: a ``provider:model`` id.
        :returns: the entry, or ``None``.
        :rtype: CatalogEntry | None
        """
        provider, sep, name = model_id.partition(":")
        return self.entries.get(
            f"{_PROVIDER_ALIASES.get(provider, provider)}{sep}{name}"
        )

    def require(self, model_id: str, *, kind: ModelKind) -> CatalogEntry:
        """Return the entry for a model id that may be built.

        :param model_id: a ``provider:model`` id.
        :param kind: the kind of model the caller builds.
        :returns: the entry, which may be deprecated.
        :rtype: CatalogEntry
        :raises ValueError: if the id is not in the catalog, is retired, or is
            another kind of model.
        """
        if (entry := self.get(model_id)) is None:
            msg = (
                f"{model_id!r} is not in the ol-llm model catalog; add it to "
                "catalog.toml in mitodl/ol-llm, or use a listed model"
            )
            raise ValueError(msg)
        if entry.status == "retired":
            msg = f"{model_id!r} is retired; pick a replacement from the catalog"
            raise ValueError(msg)
        if entry.kind != kind:
            msg = f"{model_id!r} is listed with kind {entry.kind!r}, not {kind!r}"
            raise ValueError(msg)
        return entry


def _parse_entry(raw: Mapping[str, Any]) -> CatalogEntry:
    values = dict(raw)
    label = values.get("id", "<no id>")
    for name in _PRICE_FIELDS:
        price = values.get(name)
        if not isinstance(price, str):
            # A TOML float would be read as binary floating point.
            msg = f'{label}: {name} must be a quoted decimal, e.g. "0.15"'
            raise ValueError(msg)  # noqa: TRY004 - a malformed file, not a misused API
        values[name] = Decimal(price)
    known = {f.name for f in fields(CatalogEntry) if f.init}
    if unknown := set(values) - known:
        msg = f"{label}: unknown catalog fields {sorted(unknown)}"
        raise ValueError(msg)
    try:
        return CatalogEntry(**values)
    except TypeError as exc:
        msg = f"{label}: {exc}"
        raise ValueError(msg) from exc


@cache
def load_catalog() -> Catalog:
    """Return the catalog released with this version of the library.

    :returns: the catalog.
    :rtype: Catalog
    """
    text = files("ol_llm").joinpath(CATALOG_RESOURCE).read_text(encoding="utf-8")
    return Catalog.from_toml(text)


@dataclass(frozen=True)
class CheckResult:
    """What ``check_model_ids`` found.

    :param errors: ids that are not in the catalog or are retired.
    :param warnings: ids that are deprecated.
    """

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Whether every id may be used."""
        return not self.errors


def check_model_ids(
    model_ids: Iterable[str], catalog: Catalog | None = None
) -> CheckResult:
    """Check an application's own model list against the catalog.

    For an app test that reads its own list, e.g. learn-ai's ``LLMModel``
    fixtures::

        assert check_model_ids(ids).ok

    :param model_ids: the ids the application may use.
    :param catalog: the catalog to check against; the released one when omitted.
    :returns: errors for ids that may not be built and warnings for deprecated
        ones.
    :rtype: CheckResult
    """
    catalog = catalog or load_catalog()
    errors: list[str] = []
    warnings: list[str] = []
    for model_id in dict.fromkeys(model_ids):
        entry = catalog.get(model_id)
        if entry is None:
            errors.append(f"{model_id}: not in the model catalog")
        elif entry.status == "retired":
            errors.append(f"{model_id}: retired")
        elif entry.status == "deprecated":
            warnings.append(f"{model_id}: deprecated, retiring on {entry.retires_on}")
    return CheckResult(tuple(errors), tuple(warnings))


def price_rules(catalog: Catalog | None = None) -> dict[str, Any]:
    """Return the catalog prices as a Prometheus recording rule group.

    Each entry gets ``ol_llm_model_input_usd_per_token`` and
    ``ol_llm_model_output_usd_per_token`` series labeled with the ``provider`` and
    ``model`` halves of its id, the labels the library's token counters carry, so
    spend is ``tokens * on (provider, model) group_left price``.

    :param catalog: the catalog to export; the released one when omitted.
    :returns: a rule file with one group, ready for ``json.dumps``. JSON is YAML,
        so the output loads wherever a rule file does.
    :rtype: dict[str, Any]
    """
    catalog = catalog or load_catalog()
    rules = [
        {
            "record": f"ol_llm_model_{direction}_usd_per_token",
            "expr": f"vector({price / TOKENS_PER_MTOK:f})",
            "labels": {"provider": entry.provider, "model": entry.model_name},
        }
        for entry in sorted(catalog.entries.values(), key=lambda e: e.id)
        for direction, price in (
            ("input", entry.input_usd_per_mtok),
            ("output", entry.output_usd_per_mtok),
        )
    ]
    return {"groups": [{"name": PRICE_RULE_GROUP, "rules": rules}]}


def _read_ids(path: Path) -> Iterator[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if model_id := line.partition("#")[0].strip():
            yield model_id


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``ol-llm-catalog`` command.

    ``check`` exits 1 when any id is unlisted or retired, and prints deprecated
    ids as warnings. ``prices`` prints ``price_rules`` as JSON.

    :param argv: the arguments; ``sys.argv[1:]`` when omitted.
    :returns: the exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        prog="ol-llm-catalog", description="The ol-llm model catalog."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="check model ids against the catalog, for an app's CI"
    )
    check.add_argument("model_ids", nargs="*", metavar="MODEL_ID")
    check.add_argument(
        "--file",
        type=Path,
        action="append",
        default=[],
        help="a file of model ids, one per line; # starts a comment",
    )
    commands.add_parser("prices", help="print the prices as Prometheus recording rules")
    args = parser.parse_args(argv)
    if args.command == "prices":
        sys.stdout.write(json.dumps(price_rules(), indent=2) + "\n")
        return 0
    model_ids = [*args.model_ids, *(i for path in args.file for i in _read_ids(path))]
    if not model_ids:
        parser.error("no model ids given")
    result = check_model_ids(model_ids)
    for warning in result.warnings:
        sys.stderr.write(f"warning: {warning}\n")
    for error in result.errors:
        sys.stderr.write(f"error: {error}\n")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
