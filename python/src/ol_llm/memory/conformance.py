"""The checks every memory backend must pass, through the ``Memory`` interface.

A backend's tests supply a ``ConformanceHarness`` and run each of ``CHECKS``, e.g.
as parametrized tests. Each check raises ``AssertionError`` on a failure. The
library runs them against the in-memory backend in its unit tests, and against a
live Hindsight in an opt-in integration test
(``python/tests/test_memory_hindsight.py``).

The checks reach the service only through the interface, so they cover what a
call site can do. What the interface cannot express, such as a raw request that
writes under another app's document id, is the deployed service's own suite's
job (ol-infrastructure ``scripts/hindsight/test_isolation.py``).
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from ol_llm.memory.errors import MemoryFenced, MemoryForbidden, MemoryNotFound
from ol_llm.memory.interface import Note
from ol_llm.memory.service import LearnerMemory, Memory

USE_CASE = "memory-conformance"


@dataclass(frozen=True, kw_only=True)
class ConformanceLearner:
    """A learner with an empty memory.

    :param id: the learner's platform user id, which forget takes.
    :param session_tokens: by app, what that app's ``Memory.learner`` takes.
    """

    id: str
    session_tokens: Mapping[str, str]


class ConformanceHarness(Protocol):
    """What the checks need from the backend under test.

    :param apps: two app names. Each app's ``Memory`` shares one store.
    :param general_writer: the app whose ``Memory`` holds the general grant.
    :param search_attempts: how many times a search is tried before a check
        fails, for a backend that indexes after a write returns.
    :param search_wait_seconds: the wait between those tries.
    """

    apps: tuple[str, str]
    general_writer: str
    search_attempts: int
    search_wait_seconds: float

    def memory(self, app: str) -> Memory:
        """Return the app's client, with an admin client for forget."""
        ...

    async def new_learner(self) -> ConformanceLearner:
        """Return a learner nobody has written for."""
        ...

    def general_name(self) -> str:
        """Return the name of a general memory nobody has written."""
        ...


def _expect(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _raises(
    kind: type[Exception], call: Callable[[], Awaitable[object]], message: str
) -> None:
    try:
        await call()
    except kind:
        return
    raise AssertionError(message)


async def _learner(
    harness: ConformanceHarness, learner: ConformanceLearner, app: str
) -> LearnerMemory:
    return await harness.memory(app).learner(
        learner.session_tokens[app], use_case=USE_CASE
    )


def _keys(notes: list[Note]) -> set[tuple[str, str]]:
    return {(note.app, note.key) for note in notes}


async def _search_until(
    harness: ConformanceHarness, notes: LearnerMemory, query: str, key: str
) -> list[Note]:
    found: list[Note] = []
    for attempt in range(harness.search_attempts):
        found = await notes.search(query, limit=5)
        if any(note.key == key for note in found):
            return found
        if attempt < harness.search_attempts - 1:
            await asyncio.sleep(harness.search_wait_seconds)
    return found


async def check_put_get_round_trip(harness: ConformanceHarness) -> None:
    """Check that a note comes back from get and list as it was put."""
    learner = await harness.new_learner()
    app = harness.apps[0]
    notes = await _learner(harness, learner, app)
    text = "Studies 30 minutes per weekday."
    await notes.put("pace", text, metadata={"source": "chat"})
    note = await notes.get("pace")
    _expect((note.key, note.app, note.text) == ("pace", app, text), f"got {note}")
    _expect(note.metadata == {"source": "chat"}, f"metadata is {note.metadata}")
    _expect(_keys(await notes.list()) == {(app, "pace")}, "list does not match")


async def check_put_replaces_by_key(harness: ConformanceHarness) -> None:
    """Check that a second put of a key replaces the note."""
    learner = await harness.new_learner()
    notes = await _learner(harness, learner, harness.apps[0])
    await notes.put("pace", "Studies on weekends.")
    await notes.put("pace", "Studies before work.")
    note = await notes.get("pace")
    _expect(note.text == "Studies before work.", f"get returned {note.text!r}")
    listed = [n for n in await notes.list() if n.key == "pace"]
    _expect(len(listed) == 1, f"list holds {len(listed)} notes for one key")


async def check_get_missing_is_not_found(harness: ConformanceHarness) -> None:
    """Check that get of a key nobody wrote raises, and a new memory is empty."""
    learner = await harness.new_learner()
    notes = await _learner(harness, learner, harness.apps[0])
    await _raises(
        MemoryNotFound, lambda: notes.get("nothing-here"), "get found a missing key"
    )
    _expect(await notes.list() == [], "a new learner's memory is not empty")
    _expect(await notes.search("anything") == [], "a new learner's search found notes")


async def check_search_finds_the_note(harness: ConformanceHarness) -> None:
    """Check that search returns a relevant note, with a score."""
    learner = await harness.new_learner()
    notes = await _learner(harness, learner, harness.apps[0])
    await notes.put("pace", "The learner has 30 minutes per weekday.")
    await notes.put("goal", "The learner wants a data science certificate.")
    found = await _search_until(
        harness, notes, "how much time does the learner have each weekday?", "pace"
    )
    hits = [note for note in found if note.key == "pace"]
    _expect(hits, f"search did not find the note: {found}")
    _expect(hits[0].score is not None, "a search result has no score")


async def check_delete(harness: ConformanceHarness) -> None:
    """Check that delete removes the note, and deleting it again is no error."""
    learner = await harness.new_learner()
    app = harness.apps[0]
    notes = await _learner(harness, learner, app)
    await notes.put("pace", "Studies on weekends.")
    await notes.put("goal", "Wants a certificate.")
    await notes.delete("pace")
    await notes.delete("pace")
    await notes.delete("never-written")
    _expect(_keys(await notes.list()) == {(app, "goal")}, "delete removed others")


async def check_learner_isolation(harness: ConformanceHarness) -> None:
    """Check that one learner's notes are invisible to another's."""
    a, b = await harness.new_learner(), await harness.new_learner()
    app = harness.apps[0]
    a_notes = await _learner(harness, a, app)
    b_notes = await _learner(harness, b, app)
    await a_notes.put("pace", "Learner A studies 30 minutes per weekday.")
    await b_notes.put("goal", "Learner B wants a certificate.")
    _expect(_keys(await b_notes.list()) == {(app, "goal")}, "B lists A's notes")
    await _raises(MemoryNotFound, lambda: b_notes.get("pace"), "B read A's note")
    found = await b_notes.search("Learner A studies 30 minutes per weekday", limit=5)
    _expect(all(note.key != "pace" for note in found), "B's search found A's note")
    await b_notes.delete("pace")
    _expect(
        (await a_notes.get("pace")).text.startswith("Learner A"), "B deleted A's note"
    )


async def check_every_app_reads_every_apps_notes(harness: ConformanceHarness) -> None:
    """Check that each app lists, gets and searches the other app's notes."""
    learner = await harness.new_learner()
    first, second = harness.apps
    first_notes = await _learner(harness, learner, first)
    second_notes = await _learner(harness, learner, second)
    await first_notes.put("pace", "The learner has 30 minutes per weekday.")
    await second_notes.put("goal", "The learner wants a data science certificate.")
    expected = {(first, "pace"), (second, "goal")}
    for notes in (first_notes, second_notes):
        _expect(_keys(await notes.list()) == expected, f"{notes.app} lists less")
    other = await second_notes.get("pace", app=first)
    _expect(other.app == first, f"get returned {other}")
    found = await _search_until(
        harness, second_notes, "how much time each weekday?", "pace"
    )
    _expect(any(n.app == first for n in found), f"search missed {first}'s: {found}")


async def check_an_app_cannot_change_another_apps_note(
    harness: ConformanceHarness,
) -> None:
    """Check that an app's put and delete leave another app's note intact."""
    learner = await harness.new_learner()
    first, second = harness.apps
    first_notes = await _learner(harness, learner, first)
    second_notes = await _learner(harness, learner, second)
    await first_notes.put("pace", "Written by the first app.")
    await second_notes.put("pace", "Written by the second app.")
    await second_notes.delete("pace")
    note = await first_notes.get("pace")
    _expect(note.text == "Written by the first app.", f"it became {note.text!r}")
    _expect(
        _keys(await first_notes.list()) == {(first, "pace")},
        "the second app's put or delete reached the first app's note",
    )


async def check_general_memory(harness: ConformanceHarness) -> None:
    """Check general memory: written under the grant, read with a learner token."""
    name = harness.general_name()
    writer_app = harness.general_writer
    reader_app = next(app for app in harness.apps if app != writer_app)
    writer = harness.memory(writer_app).general(name, use_case=USE_CASE)
    text = "The course runs every spring term."
    await writer.put("schedule", text)
    learner = await harness.new_learner()
    reader = (await _learner(harness, learner, reader_app)).general(name)
    _expect(
        (writer_app, "schedule") in _keys(await reader.list()),
        "a learner token cannot list general memory",
    )
    general_note = await reader.get("schedule", app=writer_app)
    _expect(general_note.text == text, f"got {general_note}")
    await _raises(
        MemoryForbidden,
        lambda: reader.put("leak", "Something a learner said."),
        "a learner token wrote general memory",
    )
    await writer.delete("schedule")
    _expect(
        (writer_app, "schedule") not in _keys(await writer.list()),
        "the general writer could not delete its note",
    )


async def check_forget_removes_everything_and_fences(
    harness: ConformanceHarness,
) -> None:
    """Check that forget removes every app's notes and refuses later writes."""
    a, b = await harness.new_learner(), await harness.new_learner()
    first, second = harness.apps
    await (await _learner(harness, a, first)).put("pace", "Studies on weekends.")
    await (await _learner(harness, a, second)).put("goal", "Wants a certificate.")
    b_notes = await _learner(harness, b, first)
    await b_notes.put("pace", "Learner B studies before work.")
    admin = harness.memory(first).admin(use_case=USE_CASE)
    removed = await admin.forget(a.id)
    _expect(removed > 0, f"forget removed {removed} items")
    for app in harness.apps:
        notes = await _learner(harness, a, app)
        _expect(await notes.list() == [], f"{app} still lists notes after forget")
    late = await _learner(harness, a, first)
    await _raises(
        MemoryFenced,
        lambda: late.put("pace", "A late write."),
        "a write after forget was accepted",
    )
    _expect(_keys(await b_notes.list()) == {(first, "pace")}, "forget reached B")
    await admin.forget(a.id)


Check = Callable[[ConformanceHarness], Awaitable[None]]

CHECKS: tuple[Check, ...] = (
    check_put_get_round_trip,
    check_put_replaces_by_key,
    check_get_missing_is_not_found,
    check_search_finds_the_note,
    check_delete,
    check_learner_isolation,
    check_every_app_reads_every_apps_notes,
    check_an_app_cannot_change_another_apps_note,
    check_general_memory,
    check_forget_removes_everything_and_fences,
)
