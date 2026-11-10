"""
Implementations for the history of a `Buffer`.

NOTE: There is no `DynamicHistory`:
      This doesn't work well, because the `Buffer` needs to be able to attach
      an event handler to the event when a history entry is loaded. This
      loading can be done asynchronously and making the history swappable would
      probably break this.
"""

from __future__ import annotations

import datetime
import os
import sys
import threading
from abc import ABCMeta, abstractmethod
from asyncio import get_running_loop
from collections.abc import AsyncGenerator, Generator, Iterable, Sequence
from contextlib import contextmanager
from typing import IO, Union

__all__ = [
    "History",
    "ThreadedHistory",
    "DummyHistory",
    "FileHistory",
    "InMemoryHistory",
]


class History(metaclass=ABCMeta):
    """
    Base ``History`` class.

    This also includes abstract methods for loading/storing history.
    """

    def __init__(self) -> None:
        # In memory storage for strings.
        self._loaded = False

        # History that's loaded already, in reverse order. Latest, most recent
        # item first.
        self._loaded_strings: list[str] = []

    #
    # Methods expected by `Buffer`.
    #

    async def load(self) -> AsyncGenerator[str, None]:
        """
        Load the history and yield all the entries in reverse order (latest,
        most recent history entry first).

        This method can be called multiple times from the `Buffer` to
        repopulate the history when prompting for a new input. So we are
        responsible here for both caching, and making sure that strings that
        were were appended to the history will be incorporated next time this
        method is called.
        """
        if not self._loaded:
            self._loaded_strings = list(self.load_history_strings())
            self._loaded = True

        for item in self._loaded_strings:
            yield item

    def get_strings(self) -> list[str]:
        """
        Get the strings from the history that are loaded so far.
        (In order. Oldest item first.)
        """
        return self._loaded_strings[::-1]

    def append_string(self, string: str) -> None:
        "Add string to the history."
        self._loaded_strings.insert(0, string)
        self.store_string(string)

    #
    # Implementation for specific backends.
    #

    @abstractmethod
    def load_history_strings(self) -> Iterable[str]:
        """
        This should be a generator that yields `str` instances.

        It should yield the most recent items first, because they are the most
        important. (The history can already be used, even when it's only
        partially loaded.)
        """
        while False:
            yield

    @abstractmethod
    def store_string(self, string: str) -> None:
        """
        Store the string in persistent storage.
        """


class ThreadedHistory(History):
    """
    Wrapper around `History` implementations that run the `load()` generator in
    a thread.

    Use this to increase the start-up time of prompt_toolkit applications.
    History entries are available as soon as they are loaded. We don't have to
    wait for everything to be loaded.
    """

    def __init__(self, history: History) -> None:
        super().__init__()

        self.history = history

        self._load_thread: threading.Thread | None = None

        # Lock for accessing/manipulating `_loaded_strings` and `_loaded`
        # together in a consistent state.
        self._lock = threading.Lock()

        # Events created by each `load()` call. Used to wait for new history
        # entries from the loader thread.
        self._string_load_events: list[threading.Event] = []

    async def load(self) -> AsyncGenerator[str, None]:
        """
        Like `History.load(), but call `self.load_history_strings()` in a
        background thread.
        """
        # Start the load thread, if this is called for the first time.
        if not self._load_thread:
            self._load_thread = threading.Thread(
                target=self._in_load_thread,
                daemon=True,
            )
            self._load_thread.start()

        # Consume the `_loaded_strings` list, using asyncio.
        loop = get_running_loop()

        # Create threading Event so that we can wait for new items.
        event = threading.Event()
        event.set()
        self._string_load_events.append(event)

        items_yielded = 0

        try:
            while True:
                # Wait for new items to be available.
                # (Use a timeout, because the executor thread is not a daemon
                # thread. The "slow-history.py" example would otherwise hang if
                # Control-C is pressed before the history is fully loaded,
                # because there's still this non-daemon executor thread waiting
                # for this event.)
                got_timeout = await loop.run_in_executor(
                    None, lambda: event.wait(timeout=0.5)
                )
                if not got_timeout:
                    continue

                # Read new items (in lock).
                def in_executor() -> tuple[list[str], bool]:
                    with self._lock:
                        new_items = self._loaded_strings[items_yielded:]
                        done = self._loaded
                        event.clear()
                    return new_items, done

                new_items, done = await loop.run_in_executor(None, in_executor)

                items_yielded += len(new_items)

                for item in new_items:
                    yield item

                if done:
                    break
        finally:
            self._string_load_events.remove(event)

    def _in_load_thread(self) -> None:
        try:
            # Start with an empty list. In case `append_string()` was called
            # before `load()` happened. Then `.store_string()` will have
            # written these entries back to disk and we will reload it.
            self._loaded_strings = []

            for item in self.history.load_history_strings():
                with self._lock:
                    self._loaded_strings.append(item)

                for event in self._string_load_events:
                    event.set()
        finally:
            with self._lock:
                self._loaded = True
            for event in self._string_load_events:
                event.set()

    def append_string(self, string: str) -> None:
        with self._lock:
            self._loaded_strings.insert(0, string)
        self.store_string(string)

    # All of the following are proxied to `self.history`.

    def load_history_strings(self) -> Iterable[str]:
        return self.history.load_history_strings()

    def store_string(self, string: str) -> None:
        self.history.store_string(string)

    def __repr__(self) -> str:
        return f"ThreadedHistory({self.history!r})"


class InMemoryHistory(History):
    """
    :class:`.History` class that keeps a list of all strings in memory.

    In order to prepopulate the history, it's possible to call either
    `append_string` for all items or pass a list of strings to `__init__` here.
    """

    def __init__(self, history_strings: Sequence[str] | None = None) -> None:
        super().__init__()
        # Emulating disk storage.
        if history_strings is None:
            self._storage = []
        else:
            self._storage = list(history_strings)

    def load_history_strings(self) -> Iterable[str]:
        yield from self._storage[::-1]

    def store_string(self, string: str) -> None:
        self._storage.append(string)


class DummyHistory(History):
    """
    :class:`.History` object that doesn't remember anything.
    """

    def load_history_strings(self) -> Iterable[str]:
        return []

    def store_string(self, string: str) -> None:
        pass

    def append_string(self, string: str) -> None:
        # Don't remember this.
        pass


_StrOrBytesPath = Union[str, bytes, "os.PathLike[str]", "os.PathLike[bytes]"]


if sys.platform == "win32":
    import msvcrt

    def _acquire_file_lock(fd: int) -> None:
        # `msvcrt.locking` locks a byte range starting at the current file
        # position. Lock the first byte of the file. This blocks (retries for
        # a couple of seconds) until the lock becomes available.
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

    def _release_file_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _acquire_file_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _release_file_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _locked_open(
    filename: _StrOrBytesPath, mode: str
) -> Generator[IO[bytes], None, None]:
    """
    Open a file while holding an exclusive, advisory file lock.

    A fresh file descriptor is used on every call, so independent sessions
    -- whether in separate processes or separate threads -- each acquire
    their own lock and are fully serialized.

    The lock is tied to the file descriptor and released automatically when
    the file is closed, including when the writer process crashes or is
    killed. Hence, a crashed writer can never leave a stale lock behind that
    blocks the other sessions.
    """
    with open(filename, mode) as f:
        _acquire_file_lock(f.fileno())
        try:
            yield f
        finally:
            _release_file_lock(f.fileno())


class FileHistory(History):
    """
    :class:`.History` class that stores all strings in a file.

    Several ``FileHistory`` instances -- also across processes, or even
    several :class:`~prompt_toolkit.shortcuts.PromptSession` objects sharing a
    single instance -- can point to the same file. Concurrent writes are
    serialized with a file lock and each (possibly multi-line) entry is
    written in a single operation, so entries are never interleaved or torn.
    The file is re-read automatically as soon as it changes on disk, which
    makes entries appended by other sessions appear in the history
    navigation, history search and auto suggestions of every session.

    The on-disk format is unchanged, so history files written by older
    prompt_toolkit versions keep working. A writer that exits in the middle
    of an entry can at worst leave an unterminated last line, which is
    silently ignored while all complete entries remain readable.
    """

    def __init__(self, filename: _StrOrBytesPath) -> None:
        self.filename = filename
        super().__init__()

        # Guards the in-memory cache (`_loaded_strings`) and the file
        # signature below against concurrent access from different sessions
        # running in different threads of the same process.
        self._lock = threading.RLock()

        # Signature of the history file the last time it was read. Used to
        # detect entries that were appended by another session/process.
        self._file_mtime_ns: int | None = None
        self._file_size: int | None = None
        self._file_inode: int | None = None

    async def load(self) -> AsyncGenerator[str, None]:
        """
        Load the history and yield all the entries in reverse order (latest,
        most recent history entry first).

        Unlike the base implementation, this re-reads the file whenever it
        changed on disk, so entries appended by other sessions/processes are
        incorporated on the next call.
        """
        with self._lock:
            self._reload_if_changed()
            # Snapshot the cache. `append_string()` could otherwise mutate
            # `_loaded_strings` while the items are being yielded.
            loaded_strings = list(self._loaded_strings)

        for item in loaded_strings:
            yield item

    def _reload_if_changed(self) -> None:
        """
        Reload the entries from disk when the file was created or modified
        since the last read (e.g. because another session appended an entry).
        """
        try:
            stat_result = os.stat(self.filename)
        except FileNotFoundError:
            # The file doesn't exist (yet). There is simply nothing to load.
            self._loaded = True
            return

        signature = (
            stat_result.st_mtime_ns,
            stat_result.st_size,
            stat_result.st_ino,
        )
        if not self._loaded or signature != (
            self._file_mtime_ns,
            self._file_size,
            self._file_inode,
        ):
            self._loaded_strings = list(self.load_history_strings())
            (
                self._file_mtime_ns,
                self._file_size,
                self._file_inode,
            ) = signature
            self._loaded = True

    def load_history_strings(self) -> Iterable[str]:
        # Reverse the order, because newest items have to go first.
        return reversed(self._read_history_strings())

    def _read_history_strings(self) -> list[str]:
        """
        Read and parse the history file. Returns the entries in order
        (oldest first).

        Lines starting with ``+`` are the lines of one history entry; any
        other line (like the ``# <timestamp>`` header) separates entries.
        Iterating over the file only yields newline-terminated lines, so an
        unterminated tail left by a writer that crashed mid-entry is ignored
        and every complete entry stays readable.
        """
        strings: list[str] = []
        lines: list[str] = []

        def add() -> None:
            if lines:
                # Join and drop trailing newline.
                string = "".join(lines)[:-1]

                strings.append(string)

        try:
            with _locked_open(self.filename, "rb") as f:
                for line_bytes in f:
                    line = line_bytes.decode("utf-8", errors="replace")

                    if line.startswith("+"):
                        lines.append(line[1:])
                    else:
                        add()
                        lines = []

                add()
        except FileNotFoundError:
            # The file doesn't exist (yet).
            return []

        return strings

    def store_string(self, string: str) -> None:
        # Build the whole (possibly multi-line) entry first, so it can be
        # written with a single write call while holding the exclusive lock.
        # Concurrent writers can therefore never interleave their entries.
        record = bytearray()
        record.extend(
            f"\n# {datetime.datetime.now()}\n".encode("utf-8", errors="replace")
        )
        for line in string.split("\n"):
            record.extend(f"+{line}\n".encode("utf-8", errors="replace"))

        with self._lock:
            with _locked_open(self.filename, "ab") as f:
                f.write(bytes(record))

            # The file changed on disk; force the next `load()` to re-read
            # it, even if the filesystem has a coarse mtime resolution.
            self._file_mtime_ns = None
            self._file_size = None
            self._file_inode = None
