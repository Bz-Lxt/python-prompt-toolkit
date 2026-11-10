from __future__ import annotations

import threading
from asyncio import run

from prompt_toolkit.history import FileHistory, InMemoryHistory, ThreadedHistory


def _call_history_load(history):
    """
    Helper: Call the history "load" method and return the result as a list of strings.
    """
    result = []

    async def call_load():
        async for item in history.load():
            result.append(item)

    run(call_load())
    return result


def test_in_memory_history():
    history = InMemoryHistory()
    history.append_string("hello")
    history.append_string("world")

    # Newest should yield first.
    assert _call_history_load(history) == ["world", "hello"]

    # Test another call.
    assert _call_history_load(history) == ["world", "hello"]

    history.append_string("test3")
    assert _call_history_load(history) == ["test3", "world", "hello"]

    # Passing history as a parameter.
    history2 = InMemoryHistory(["abc", "def"])
    assert _call_history_load(history2) == ["def", "abc"]


def test_file_history(tmpdir):
    histfile = tmpdir.join("history")

    history = FileHistory(histfile)

    history.append_string("hello")
    history.append_string("world")

    # Newest should yield first.
    assert _call_history_load(history) == ["world", "hello"]

    # Test another call.
    assert _call_history_load(history) == ["world", "hello"]

    history.append_string("test3")
    assert _call_history_load(history) == ["test3", "world", "hello"]

    # Create another history instance pointing to the same file.
    history2 = FileHistory(histfile)
    assert _call_history_load(history2) == ["test3", "world", "hello"]


def test_threaded_file_history(tmpdir):
    histfile = tmpdir.join("history")

    history = ThreadedHistory(FileHistory(histfile))

    history.append_string("hello")
    history.append_string("world")

    # Newest should yield first.
    assert _call_history_load(history) == ["world", "hello"]

    # Test another call.
    assert _call_history_load(history) == ["world", "hello"]

    history.append_string("test3")
    assert _call_history_load(history) == ["test3", "world", "hello"]

    # Create another history instance pointing to the same file.
    history2 = ThreadedHistory(FileHistory(histfile))
    assert _call_history_load(history2) == ["test3", "world", "hello"]


def test_threaded_in_memory_history():
    # Threaded in memory history is not useful. But testing it anyway, just to
    # see whether everything plays nicely together.
    history = ThreadedHistory(InMemoryHistory())
    history.append_string("hello")
    history.append_string("world")

    # Newest should yield first.
    assert _call_history_load(history) == ["world", "hello"]

    # Test another call.
    assert _call_history_load(history) == ["world", "hello"]

    history.append_string("test3")
    assert _call_history_load(history) == ["test3", "world", "hello"]

    # Passing history as a parameter.
    history2 = ThreadedHistory(InMemoryHistory(["abc", "def"]))
    assert _call_history_load(history2) == ["def", "abc"]


def test_file_history_multiline(tmpdir):
    histfile = tmpdir.join("history")
    history = FileHistory(histfile)

    history.append_string("line1\nline2\nline3")

    # The whole multi-line entry has to be preserved and read back intact.
    assert _call_history_load(history) == ["line1\nline2\nline3"]

    # A freshly created history instance (another session) reads the same.
    history2 = FileHistory(histfile)
    assert _call_history_load(history2) == ["line1\nline2\nline3"]


def test_file_history_reloads_entries_appended_by_other_sessions(tmpdir):
    histfile = tmpdir.join("history")

    history1 = FileHistory(histfile)
    history1.append_string("one")
    assert _call_history_load(history1) == ["one"]

    # Another session (different `FileHistory` instance on the same file)
    # appends entries while the first session is running.
    history2 = FileHistory(histfile)
    history2.append_string("two")
    history2.append_string("three\nlines")

    # The first session picks up the new entries on its next load. These are
    # what power navigation, history search and auto suggestions.
    assert _call_history_load(history1) == ["three\nlines", "two", "one"]

    # `get_strings()` (used by auto suggestions and yank-nth-arg) is refreshed
    # as well, in chronological order.
    assert history1.get_strings() == ["one", "two", "three\nlines"]


def test_file_history_does_not_reload_unchanged_file(tmpdir, monkeypatch):
    histfile = tmpdir.join("history")
    history = FileHistory(histfile)
    history.append_string("hello")

    assert _call_history_load(history) == ["hello"]

    # While the file doesn't change, the file must not be re-read.
    calls = 0
    original = history._read_history_strings

    def counting_read():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(history, "_read_history_strings", counting_read)
    assert _call_history_load(history) == ["hello"]
    assert calls == 0


def test_file_history_concurrent_appenders(tmpdir):
    histfile = tmpdir.join("history")

    thread_count = 20
    entries = [
        f"entry {i}\nsecond line {i}\nthird line {i}" for i in range(thread_count)
    ]
    errors = []

    def writer(entry):
        try:
            # Every thread acts as an independent session, but they all write
            # to the same history file.
            FileHistory(histfile).store_string(entry)
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(entry,)) for entry in entries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []

    # Every multi-line entry must be present exactly once and complete.
    loaded = _call_history_load(FileHistory(histfile))
    assert sorted(loaded) == sorted(entries)
    assert len(loaded) == thread_count


def test_file_history_concurrent_appenders_sharing_one_instance(tmpdir):
    histfile = tmpdir.join("history")

    # Several sessions share a single `FileHistory` instance.
    history = FileHistory(histfile)
    thread_count = 20
    entries = [f"shared {i}\nline {i}" for i in range(thread_count)]

    def writer(entry):
        history.store_string(entry)

    threads = [threading.Thread(target=writer, args=(entry,)) for entry in entries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    loaded = _call_history_load(FileHistory(histfile))
    assert sorted(loaded) == sorted(entries)
    assert len(loaded) == thread_count


def test_file_history_survives_writer_crashing_mid_entry(tmpdir):
    histfile = tmpdir.join("history")

    history = FileHistory(histfile)
    history.append_string("complete one")
    history.append_string("complete two\nwith two lines")

    # Simulate a writer that is killed after writing only part of its last
    # entry (no terminating newline on the final line).
    with open(histfile, "ab") as f:
        f.write(b"\n# 2020-01-01 00:00:00\n+crashed entry\n+partial line")

    # Loading must not fail, and all complete entries remain readable.
    loaded = _call_history_load(FileHistory(histfile))
    assert "complete one" in loaded
    assert "complete two\nwith two lines" in loaded
    for entry in loaded:
        assert "partial line" not in entry

    # The history remains usable: new entries can still be stored and read.
    recovered = FileHistory(histfile)
    recovered.append_string("after crash")
    assert _call_history_load(recovered)[0] == "after crash"


def test_file_history_reads_legacy_file_format(tmpdir):
    histfile = tmpdir.join("history")

    # Write a file exactly like older prompt_toolkit versions did.
    with open(histfile, "wb") as f:
        f.write(b"\n# 2020-01-01 00:00:00\n+hello\n")
        f.write(b"\n# 2020-01-02 00:00:00\n+multi\n+line\n")

    history = FileHistory(histfile)
    assert _call_history_load(history) == ["multi\nline", "hello"]

    # Appending keeps working on top of the legacy file.
    history.append_string("new entry")
    assert _call_history_load(history) == ["new entry", "multi\nline", "hello"]


def test_file_history_missing_file(tmpdir):
    histfile = tmpdir.join("does-not-exist")
    history = FileHistory(histfile)

    # Loading before the file was created must not raise.
    assert _call_history_load(history) == []

    history.append_string("first")
    assert _call_history_load(history) == ["first"]
