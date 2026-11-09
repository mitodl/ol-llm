"""The event loops ol-llm runs on, and the values it keeps for each of them.

Provider clients bind to the event loop they first run on, so ol-llm keeps one per
loop (``LoopLocal``). Every synchronous entry point, and ``run_coroutine`` for an
app's own coroutines, runs on one long-lived loop per thread that ol-llm owns. A
thread's clients are then built once and reused by every call it makes, and they
are closed when ``close_thread_loop`` shuts that loop down (at interpreter exit for
the main thread). Trace context lives in contextvars, which a coroutine run here
inherits from the calling thread, so a call made inside the caller's own trace
still lands in it.

A loop ol-llm does not own (``asyncio.run`` per task, a framework's loop) is the
app's: values built for it are dropped, not closed, once it has closed, so an app
that starts a loop per call builds and abandons a client per call. Use
``run_coroutine`` instead.

In a forked child (a Celery prefork worker), the parent's loop and every value
built for it are set aside, never used or closed, since their sockets and
selector are shared with the parent and the threads that served them did not
survive the fork. The child builds its own on its next call.
"""

import asyncio
import atexit
import logging
import os
import threading
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import Any, Generic, TypeVar

_T = TypeVar("_T")
_V = TypeVar("_V")

log = logging.getLogger(__name__)


class LoopLocal(threading.local, Generic[_V]):
    """A value built once for each event loop that uses it, kept by the loop's thread.

    For what holds provider clients, which bind to the loop they first run on: a
    pooled ``httpx`` connection belongs to that loop's transport, and SDK clients
    keep asyncio locks and events (a provider's own enter lock among them). Shared
    across loops, they fail with "bound to a different event loop", or "Event loop
    is closed" once the first loop is gone. A loop runs on one thread, so keeping
    the values per thread loses nothing, and a thread's values are released with
    the thread.

    Every async client in ol-llm is held by one of these, so that the runner can
    close it and a forked child can drop it.

    :param build: builds the value; called again for each new loop.
    :param close: closes a value; awaited on its loop before ol-llm shuts that
        loop down. Without it, values are dropped with the loop.
    """

    def __init__(
        self,
        build: Callable[[], _V],
        close: Callable[[_V], Awaitable[None]] | None = None,
    ) -> None:
        """Start empty; ``threading.local`` runs this again in each new thread."""
        self._build = build
        self._close = close
        self._values: dict[asyncio.AbstractEventLoop, _V] = {}
        _instances.add(self)

    def get(self) -> _V:
        """Return the running loop's value, dropping the values of closed loops.

        :returns: the value.
        :rtype: _V
        :raises RuntimeError: if no event loop is running.
        """
        loop = asyncio.get_running_loop()
        for closed in [other for other in self._values if other.is_closed()]:
            del self._values[closed]
        if (value := self._values.get(loop)) is None:
            value = self._values[loop] = self._build()
        return value

    async def _close_running(self) -> None:
        """Close and forget the running loop's value, if this thread built one."""
        value = self._values.pop(asyncio.get_running_loop(), None)
        if value is not None and self._close is not None:
            await self._close(value)

    def _forget(self) -> None:
        self._values = {}


# Weak, so a LoopLocal owned by a discarded model is not kept alive by the runner.
_instances: "weakref.WeakSet[LoopLocal[Any]]" = weakref.WeakSet()


class _ThreadLoop(threading.local):
    loop: asyncio.AbstractEventLoop | None = None


_thread = _ThreadLoop()
# Every thread's runner loop, so a fork can keep the ones whose threads it drops.
_loops: "weakref.WeakSet[asyncio.AbstractEventLoop]" = weakref.WeakSet()


def _thread_loop() -> asyncio.AbstractEventLoop:
    loop = _thread.loop
    if loop is None or loop.is_closed():
        loop = _thread.loop = asyncio.new_event_loop()
        _loops.add(loop)
    return loop


def run_coroutine(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine on this thread's ol-llm event loop and return its result.

    The supported way for sync code (a Celery task, a Django view, a Dagster asset)
    to await ol-llm, in place of ``asyncio.run`` or ``async_to_sync``: the loop
    lives as long as the thread, so the clients built for it are reused by every
    later call instead of built and abandoned per call. If the caller is
    interrupted, the coroutine is cancelled and its cleanup run before the
    exception propagates.

    :param coro: the coroutine.
    :returns: what it returns.
    :rtype: _T
    :raises RuntimeError: if an event loop is already running in this thread;
        ``await`` the coroutine there instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coro.close()
        msg = (
            "run_coroutine cannot be called while an event loop is running in "
            "this thread; await the coroutine instead"
        )
        raise RuntimeError(msg)
    loop = _thread_loop()
    task = loop.create_task(coro)
    try:
        return loop.run_until_complete(task)
    except BaseException:
        if not task.done():
            task.cancel()
            with suppress(BaseException):
                loop.run_until_complete(task)
        raise


async def _close_values() -> None:
    for local in list(_instances):
        try:
            await local._close_running()  # noqa: SLF001
        except Exception:
            log.warning("Failed to close a loop-local client", exc_info=True)


def close_thread_loop() -> None:
    """Close this thread's ol-llm event loop and the clients built for it.

    For a worker thread that is about to finish, or a test. The next call in the
    thread starts a new loop. Registered for the main thread at interpreter exit.
    """
    loop = _thread.loop
    _thread.loop = None
    if loop is None or loop.is_closed():
        return
    try:
        loop.run_until_complete(_close_values())
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        loop.close()


def register_at_fork(**hooks: Callable[[], object]) -> None:
    """``os.register_at_fork`` where the platform can fork (not Windows).

    :param hooks: ``before``, ``after_in_parent`` and ``after_in_child``.
    """
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(**hooks)


# What a forked child inherited, kept so it is never garbage collected. Collecting
# a parent's loop closes it, and closing unregisters its self-pipe from the epoll
# instance the child shares with the parent, after which that loop is never woken
# by another thread again. That holds for every thread's loop: the child drops the
# other threads' thread-local state, so their loops are held from before the fork.
_inherited: list[object] = []
_held_across_fork: list[asyncio.AbstractEventLoop] = []


def _before_fork() -> None:
    _held_across_fork[:] = list(_loops)


def _after_fork_in_parent() -> None:
    _held_across_fork.clear()


def _after_fork_in_child() -> None:
    _inherited.extend(_held_across_fork)
    _held_across_fork.clear()
    _inherited.append(_thread.loop)
    _thread.loop = None
    for local in list(_instances):
        _inherited.append(local._values)  # noqa: SLF001
        local._forget()  # noqa: SLF001


atexit.register(close_thread_loop)
register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_in_parent,
    after_in_child=_after_fork_in_child,
)
