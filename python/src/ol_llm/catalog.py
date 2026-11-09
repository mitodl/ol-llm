"""The model catalog: what we know about the models applications call.

The catalog is ``catalog.toml``, released with the library (usage policies spec
§2.2). It records each model's status and price; it is not an allowlist.
``build_model`` builds an id the catalog does not list, with a warning and a trace
tag, so a new model needs no library release. It refuses a retired id. An
application holds its own model lists (database rows, settings) to the catalog in
CI with ``check_model_ids`` or the ``ol-llm-catalog check`` command, so a release
that retires a model fails the CI of every app still naming it. ``price_rules``
exports the prices as the Prometheus recording rules the spend dashboard and alerts
join token counters against; a model the catalog does not list has no price there.

Needs no extra: call sites not built on PydanticAI check their ids here too, and
their use cases with ``check_chat_model``.
"""

import argparse
import json
import logging
import sys
import threading
import tomllib
import weakref
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import cache, update_wrapper
from importlib.resources import files
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from ol_llm.loops import register_at_fork
from ol_llm.policy import UseCase

ModelKind = Literal["chat", "embedding"]
ModelStatus = Literal["allowed", "deprecated", "retired"]
Route = Literal["direct", "gateway"]

TOKENS_PER_MTOK = 1_000_000

# How many use cases whose models the catalog all lists a process keeps checks, models
# and embedders for (``UseCaseCache``). An application declares those, so there are
# only so many; this is a safety net, not a window callers can cycle through.
CHECKED_USE_CASES_MAX = 1024

# How many use cases naming a model the catalog does not list (or an embedding length
# it does not record) a process keeps checks, models and embedders for. An
# application can build a use case from a request's model id, so a caller can make
# as many of these as it sends ids. Kept apart and few, they can't evict the
# application's own cataloged use cases, which would then rebuild their clients
# under the lock every first call shares. An application's own use case on an
# unlisted model shares this tier, so caller churn can evict it too; listing its
# model in the catalog keeps it protected.
UNCATALOGED_USE_CASES_MAX = 64

CATALOG_RESOURCE = "catalog.toml"
PRICE_RULE_GROUP = "ol_llm_model_prices"

log = logging.getLogger(__name__)

# PydanticAI providers that reach the same model through another API shape. The
# catalog lists the model once, under the base provider.
_PROVIDER_ALIASES: Mapping[str, str] = {
    "openai-chat": "openai",
    "openai-responses": "openai",
    "azure-responses": "azure",
}


def require_model_id_form(model_id: str) -> str:
    """Return a model id that is in ``provider:model`` form.

    :param model_id: the id to check.
    :returns: the id, unchanged.
    :rtype: str
    :raises ValueError: if either half is missing.
    """
    provider, sep, name = model_id.partition(":")
    if not (provider and sep and name):
        msg = f"{model_id!r} is not in provider:model form"
        raise ValueError(msg)
    return model_id


class CatalogEntry(BaseModel):
    """One model in the catalog. The fields are described in ``catalog.toml``.

    Strict, so a quoted boolean or date is refused rather than read as truthy.

    :raises ValueError: on a malformed id, an unknown field, kind, status, or
        route, a price that is not a quoted decimal, a deprecated entry with no
        ``retires_on``, ``route_falls_back`` on a direct route, only one of the two
        Opik names, a negative price, an embedding entry without a positive
        ``dimensions``, or a chat entry with one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

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
    dimensions: int | None = None

    @field_validator("input_usd_per_mtok", "output_usd_per_mtok", mode="before")
    @classmethod
    def _exact_price(cls, value: object) -> Decimal:
        # A TOML float would be read as binary floating point.
        if isinstance(value, Decimal):
            return value
        if not isinstance(value, str):
            msg = 'must be a quoted decimal, e.g. "0.15"'
            raise ValueError(msg)  # noqa: TRY004 - pydantic reports ValueError
        return Decimal(value)

    @field_validator("id")
    @classmethod
    def _provider_model_form(cls, value: str) -> str:
        return require_model_id_form(value)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.status == "deprecated" and self.retires_on is None:
            msg = "a deprecated entry needs retires_on"
            raise ValueError(msg)
        if self.route_falls_back and self.route != "gateway":
            msg = "only a gateway route can fall back on its own"
            raise ValueError(msg)
        if (self.opik_provider is None) != (self.opik_model is None):
            msg = "set opik_provider and opik_model together, or neither"
            raise ValueError(msg)
        if self.input_usd_per_mtok < 0 or self.output_usd_per_mtok < 0:
            msg = "prices must not be negative"
            raise ValueError(msg)
        if self.kind == "embedding" and (self.dimensions or 0) <= 0:
            msg = "an embedding entry needs a positive dimensions"
            raise ValueError(msg)
        if self.kind == "chat" and self.dimensions is not None:
            msg = "only an embedding entry has dimensions"
            raise ValueError(msg)
        return self

    @property
    def provider(self) -> str:
        """The provider half of the id, e.g. ``bedrock``."""
        return self.id.partition(":")[0]

    @property
    def model_name(self) -> str:
        """The model half of the id, e.g. ``us.anthropic.claude-sonnet-5``."""
        return self.id.partition(":")[2]


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

    def buildable(self, model_id: str, *, kind: ModelKind) -> CatalogEntry | None:
        """Return the entry for a model id that may be built.

        :param model_id: a ``provider:model`` id.
        :param kind: the kind of model the caller builds.
        :returns: the entry, which may be deprecated, or ``None`` when the catalog
            does not list the id.
        :rtype: CatalogEntry | None
        :raises ValueError: if the id is not in ``provider:model`` form, is
            retired, or is another kind of model.
        """
        if (entry := self.get(require_model_id_form(model_id))) is None:
            return None
        if entry.status == "retired":
            msg = f"{model_id!r} is retired; pick a replacement from the catalog"
            raise ValueError(msg)
        if entry.kind != kind:
            msg = f"{model_id!r} is listed with kind {entry.kind!r}, not {kind!r}"
            raise ValueError(msg)
        return entry


def _parse_entry(raw: Mapping[str, Any]) -> CatalogEntry:
    try:
        return CatalogEntry.model_validate(raw)
    except ValidationError as exc:
        msg = f"Catalog entry {raw.get('id', '<no id>')!r}: {exc}"
        raise ValueError(msg) from exc


@cache
def load_catalog() -> Catalog:
    """Return the catalog released with this version of the library.

    :returns: the catalog.
    :rtype: Catalog
    """
    text = files("ol_llm").joinpath(CATALOG_RESOURCE).read_text(encoding="utf-8")
    return Catalog.from_toml(text)


def catalog_entries(use_case: UseCase, *, kind: ModelKind) -> list[CatalogEntry | None]:
    """Look up every model the use case names; retired models and other kinds raise.

    The catalog is not an allowlist: an id it does not list builds, with a warning
    and no entry, and a deprecated one builds with a warning.

    :param use_case: the call site.
    :param kind: the kind of model the caller builds.
    :returns: the entries, in the order of ``use_case.model_ids``, with ``None``
        for an id the catalog does not list.
    :rtype: list[CatalogEntry | None]
    :raises ValueError: on a model id not in ``provider:model`` form, that is
        retired, or that the catalog lists as another kind, or fallbacks behind a
        route that falls back on its own.
    """
    catalog = load_catalog()
    entries = [
        catalog.buildable(model_id, kind=kind) for model_id in use_case.model_ids
    ]
    primary = entries[0]
    if use_case.fallback_models and primary is not None and primary.route_falls_back:
        msg = (
            f"Use case {use_case.name!r} declares fallback_models, but the route to "
            f"{use_case.model!r} already falls back on its own (spec §6.4)"
        )
        raise ValueError(msg)
    for model_id, entry in zip(use_case.model_ids, entries, strict=True):
        if entry is None:
            log.warning(
                "Use case %s names %r, which the model catalog does not list, so "
                "the spend rules have no price for it",
                use_case.name,
                model_id,
            )
        elif entry.status == "deprecated":
            log.warning(
                "Use case %s names %r, which is deprecated and retires on %s",
                use_case.name,
                model_id,
                entry.retires_on,
            )
    return entries


def status_tags(entries: Iterable[CatalogEntry | None]) -> tuple[str, ...]:
    """Return the ``model_status`` trace tags for a chain's catalog entries.

    :param entries: one per model, ``None`` for a model the catalog does not list.
    :returns: the tags, in a stable order.
    :rtype: tuple[str, ...]
    """
    statuses = set()
    for entry in entries:
        if entry is None:
            statuses.add("uncataloged")
        elif entry.status == "deprecated":
            statuses.add("deprecated")
    return tuple(f"model_status:{status}" for status in sorted(statuses))


_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")

# Weak, so a cache a test replaces is not kept alive for the fork hook.
_caches: "weakref.WeakSet[UseCaseCache[Any, Any]]" = weakref.WeakSet()


def is_cataloged(use_case: UseCase) -> bool:
    """Return whether the catalog lists every model the use case names.

    :param use_case: the call site.
    :returns: ``True`` when it lists them all.
    :rtype: bool
    """
    catalog = load_catalog()
    return all(catalog.get(model_id) is not None for model_id in use_case.model_ids)


def is_cataloged_embedder(key: tuple[UseCase, int | None]) -> bool:
    """Return whether the catalog lists an embedder's model at the length asked for.

    An application can pass a request's vector length on, as it can a model id, so
    only no length or the one the catalog records counts.

    :param key: the use case and the vector length asked for, ``None`` for the
        model's own.
    :returns: ``True`` when the use case is cataloged at that length.
    :rtype: bool
    """
    use_case, dimensions = key
    entry = load_catalog().get(use_case.model)
    return (
        is_cataloged(use_case)
        and entry is not None
        and dimensions in (None, entry.dimensions)
    )


class UseCaseCache(Generic[_K, _V]):
    """Values kept per use case, in two tiers so a caller can't evict an app's own.

    A key ``protected`` accepts (e.g. ``is_cataloged``) is one the application
    declared, so its value goes in a tier of up to ``CHECKED_USE_CASES_MAX``. Any
    other key may come from a request, so its value goes in a tier of the
    ``UNCATALOGED_USE_CASES_MAX`` most recently used. Each tier drops its least
    recently used value once it is full.

    :param protected: whether a key belongs in the protected tier. Called only when
        a value is kept.
    """

    def __init__(self, protected: Callable[[_K], bool]) -> None:
        """Start empty."""
        self._protected = protected
        self._lock = threading.Lock()
        self._cataloged: OrderedDict[_K, _V] = OrderedDict()
        self._uncataloged: OrderedDict[_K, _V] = OrderedDict()
        _caches.add(self)

    def get(self, key: _K) -> _V | None:
        """Return the value kept for a key, or ``None``, and mark it recently used.

        :param key: what the value was kept for.
        :returns: the value, or ``None``.
        :rtype: _V | None
        """
        with self._lock:
            for tier in (self._cataloged, self._uncataloged):
                if key in tier:
                    tier.move_to_end(key)
                    return tier[key]
        return None

    def put(self, key: _K, value: _V) -> None:
        """Keep a value, in the tier its key belongs to.

        :param key: what the value is for.
        :param value: the value.
        """
        if self._protected(key):
            tier, most = self._cataloged, CHECKED_USE_CASES_MAX
        else:
            tier, most = self._uncataloged, UNCATALOGED_USE_CASES_MAX
        with self._lock:
            tier[key] = value
            tier.move_to_end(key)
            if len(tier) > most:
                tier.popitem(last=False)

    def get_or_build(self, key: _K, build: Callable[[], _V]) -> _V:
        """Return the value kept for a key, building and keeping it if there is none.

        A build that raises keeps nothing, so the next call builds again.

        :param key: what the value is for.
        :param build: builds the value.
        :returns: the value.
        :rtype: _V
        """
        value = self.get(key)
        if value is None:
            value = build()
            self.put(key, value)
        return value

    def sizes(self) -> tuple[int, int]:
        """Return how many values each tier keeps: cataloged, then uncataloged."""
        with self._lock:
            return len(self._cataloged), len(self._uncataloged)

    def clear(self) -> None:
        """Drop every value."""
        with self._lock:
            self._cataloged.clear()
            self._uncataloged.clear()


def _after_fork_in_child() -> None:
    # A parent thread may have held a cache's lock at the fork, and won't release it
    # here.
    for kept in list(_caches):
        kept._lock = threading.Lock()  # noqa: SLF001


register_at_fork(after_in_child=_after_fork_in_child)


class CachedCheck(Generic[_V]):
    """A use case check that runs once per use case, kept in a ``UseCaseCache``.

    A check that raises keeps nothing, so a rejected use case raises every time.

    :param check: the check.
    """

    def __init__(self, check: Callable[[UseCase], _V]) -> None:
        """Wrap the check."""
        self._check = check
        self.cache: UseCaseCache[UseCase, _V] = UseCaseCache(is_cataloged)
        update_wrapper(self, check)

    def __call__(self, use_case: UseCase) -> _V:
        """Return the check's result for the use case, running it on a miss."""
        return self.cache.get_or_build(use_case, lambda: self._check(use_case))

    def cache_clear(self) -> None:
        """Forget every result, so the next call for each use case checks again."""
        self.cache.clear()


@CachedCheck
def check_chat_model(use_case: UseCase) -> tuple[str, ...]:
    """Check a chat use case's models against the catalog, as ``build_model`` does.

    For call sites that build their own chat client, e.g. a LangChain chat model,
    at the point they build it. ``traced_chat`` runs it too and tags its trace with
    the result. A rejected use case raises every time; an unlisted or deprecated
    model warns once, until its use case falls out of ``UseCaseCache``.

    :param use_case: the call site.
    :returns: the ``model_status`` trace tags, from ``status_tags``.
    :rtype: tuple[str, ...]
    :raises ValueError: on a model id not in ``provider:model`` form, that is
        retired, or that the catalog lists as an embedding model, or fallbacks
        behind a route that falls back on its own.
    """
    return status_tags(catalog_entries(use_case, kind="chat"))


def embedding_entry(use_case: UseCase) -> CatalogEntry | None:
    """Check an embedding use case against the catalog and return its entry.

    Another model's vectors are not comparable with the primary's, so an embedding
    use case takes no fallback, its own or its gateway route's.

    :param use_case: the call site.
    :returns: the entry, or ``None`` when the catalog does not list the model.
    :rtype: CatalogEntry | None
    :raises ValueError: if the use case declares fallbacks, or its model is not in
        ``provider:model`` form, is retired, is listed as a chat model, or is on a
        gateway route that falls back.
    """
    if use_case.fallback_models:
        msg = (
            f"Use case {use_case.name!r} declares fallback_models; an embedding use "
            "case takes none, since another model's vectors are not comparable"
        )
        raise ValueError(msg)
    [entry] = catalog_entries(use_case, kind="embedding")
    if entry is not None and entry.route_falls_back:
        msg = (
            f"{use_case.model}'s gateway route falls back on its own; an embedding "
            "use case cannot take another model's vectors, which are not comparable"
        )
        raise ValueError(msg)
    return entry


@dataclass(frozen=True)
class CheckResult:
    """What ``check_model_ids`` found.

    :param errors: ids that are retired or not in ``provider:model`` form.
    :param warnings: ids that are deprecated or not in the catalog.
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
    :returns: errors for ids that may not be built, and warnings for deprecated
        ids and ids the catalog does not list.
    :rtype: CheckResult
    """
    catalog = catalog or load_catalog()
    errors: list[str] = []
    warnings: list[str] = []
    for model_id in dict.fromkeys(model_ids):
        try:
            require_model_id_form(model_id)
        except ValueError:
            errors.append(f"{model_id}: not in provider:model form")
            continue
        if (entry := catalog.get(model_id)) is None:
            warnings.append(f"{model_id}: not in the model catalog, so it has no price")
        elif entry.status == "retired":
            errors.append(f"{model_id}: retired")
        elif entry.status == "deprecated":
            warnings.append(f"{model_id}: deprecated, retiring on {entry.retires_on}")
    return CheckResult(tuple(errors), tuple(warnings))


def price_rules(catalog: Catalog | None = None) -> dict[str, Any]:
    """Return the catalog prices as a Prometheus recording rule group.

    Each entry gets ``ol_llm_model_input_usd_per_token`` and
    ``ol_llm_model_output_usd_per_token`` series labeled with the ``provider`` and
    ``model`` halves of its id, so spend is ``tokens * on (provider, model)
    group_left price``. Token counters joined against these must carry the same
    split of the catalog id: ``openai``, not ``openai-chat``, and the requested
    model name, not the dated one a response reports.

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

    ``check`` exits 1 when any id is retired or malformed, and prints deprecated
    and unlisted ids as warnings. ``prices`` prints ``price_rules`` as JSON.

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
