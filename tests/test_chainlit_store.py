"""Persistence pieces the web UI needs.

Skipped when the `gui` extra is not installed — the CLI must stay usable without
Chainlit, so these cannot be hard requirements of the test suite.
"""

import sqlite3

import pytest

pytest.importorskip("chainlit", reason="install the gui extra to test the web UI store")

import chainlit_store  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    from ragbot.config import settings
    monkeypatch.setattr(settings, "chainlit_db", str(tmp_path / "chainlit.sqlite"))
    return tmp_path


def test_schema_creates_every_table_the_data_layer_writes_to(store):
    chainlit_store.ensure_schema()

    with sqlite3.connect(chainlit_store.db_path()) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

    assert {"users", "threads", "steps", "elements", "feedbacks"} <= tables


def test_schema_is_idempotent(store):
    chainlit_store.ensure_schema()
    chainlit_store.ensure_schema()   # must not raise on an existing database


def test_threads_table_has_the_columns_chainlit_inserts(store):
    """The data layer builds INSERTs from dict keys, so a missing column only
    shows up as a runtime SQL error mid-conversation."""
    chainlit_store.ensure_schema()

    with sqlite3.connect(chainlit_store.db_path()) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(threads)")}

    assert {"id", "createdAt", "name", "userId", "userIdentifier", "tags",
            "metadata"} <= columns


async def test_upload_then_read_round_trips(tmp_path):
    client = chainlit_store.LocalStorageClient(tmp_path)

    result = await client.upload_file("thread/el/[1]", "цитата", mime="text/plain")

    assert result["url"] == f"{chainlit_store.ELEMENTS_ROUTE}/thread/el/[1]"
    assert (tmp_path / "thread/el/[1]").read_text("utf-8") == "цитата"


async def test_upload_accepts_bytes(tmp_path):
    client = chainlit_store.LocalStorageClient(tmp_path)
    await client.upload_file("a/b", b"\x00\x01binary")
    assert (tmp_path / "a/b").read_bytes() == b"\x00\x01binary"


async def test_object_key_cannot_escape_the_storage_directory(tmp_path):
    """Keys come from Chainlit, but a traversal would write anywhere on disk."""
    client = chainlit_store.LocalStorageClient(tmp_path / "elements")

    with pytest.raises(ValueError, match="escapes"):
        await client.upload_file("../../etc/passwd", "pwned")


async def test_delete_removes_the_file_and_tolerates_a_missing_one(tmp_path):
    client = chainlit_store.LocalStorageClient(tmp_path)
    await client.upload_file("x", "data")

    assert await client.delete_file("x") is True
    assert not (tmp_path / "x").exists()
    assert await client.delete_file("x") is True        # already gone
    assert await client.delete_file("../escape") is False
