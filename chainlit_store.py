"""Persistence for the Chainlit UI — what makes the conversation sidebar work.

Chainlit keeps its own record of threads, steps and elements, separate from the
LangGraph checkpointer. That duplication is not avoidable without implementing
`BaseDataLayer` by hand, but it is harmless as long as the *ids match*: `app.py`
uses Chainlit's thread id as the graph's `thread_id`, so both stores describe the
same conversation and neither can silently drift.

Two things Chainlit does not do for you:

- `SQLAlchemyDataLayer` never creates its schema, so `ensure_schema()` does;
- the thread endpoints read `current_user.identifier`, and `get_current_user()`
  returns None when no auth callback is registered — so without a login the
  sidebar raises rather than degrading. `app.py` registers password auth.

Columns mirror Chainlit's `ThreadDict` / `StepDict` / `ElementDict`: the data
layer builds its INSERTs from whatever keys a dict happens to carry, so a
missing column surfaces as a runtime SQL error rather than a startup one.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, Union

from chainlit.data.storage_clients.base import BaseStorageClient

from ragbot.config import settings

ELEMENTS_ROUTE = "/element-files"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id"          TEXT PRIMARY KEY,
    "identifier"  TEXT NOT NULL UNIQUE,
    "metadata"    TEXT NOT NULL,
    "createdAt"   TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id"             TEXT PRIMARY KEY,
    "createdAt"      TEXT,
    "name"           TEXT,
    "userId"         TEXT,
    "userIdentifier" TEXT,
    "tags"           TEXT,
    "metadata"       TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id"            TEXT PRIMARY KEY,
    "name"          TEXT NOT NULL,
    "type"          TEXT NOT NULL,
    "threadId"      TEXT NOT NULL,
    "parentId"      TEXT,
    "streaming"     INTEGER NOT NULL,
    "waitForAnswer" INTEGER,
    "isError"       INTEGER,
    "metadata"      TEXT,
    "tags"          TEXT,
    "input"         TEXT,
    "output"        TEXT,
    "createdAt"     TEXT,
    "command"       TEXT,
    "start"         TEXT,
    "end"           TEXT,
    "generation"    TEXT,
    "showInput"     TEXT,
    "defaultOpen"   INTEGER,
    "autoCollapse"  INTEGER,
    "modes"         TEXT,
    "language"      TEXT,
    "indent"        INTEGER,
    "icon"          TEXT
);

CREATE TABLE IF NOT EXISTS elements (
    "id"           TEXT PRIMARY KEY,
    "threadId"     TEXT,
    "type"         TEXT,
    "url"          TEXT,
    "chainlitKey"  TEXT,
    "name"         TEXT NOT NULL,
    "display"      TEXT,
    "objectKey"    TEXT,
    "size"         TEXT,
    "page"         INTEGER,
    "language"     TEXT,
    "forId"        TEXT,
    "mime"         TEXT,
    "props"        TEXT,
    "autoPlay"     INTEGER,
    "playerConfig" TEXT,
    "path"         TEXT
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id"       TEXT PRIMARY KEY,
    "forId"    TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value"    INTEGER NOT NULL,
    "comment"  TEXT
);
"""


def db_path() -> Path:
    path = Path(settings.chainlit_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_schema() -> None:
    """Create the tables if they are missing. Safe to call on every startup."""
    with sqlite3.connect(db_path()) as conn:
        conn.executescript(SCHEMA)


def elements_dir() -> Path:
    path = db_path().parent / "elements"
    path.mkdir(parents=True, exist_ok=True)
    return path


class LocalStorageClient(BaseStorageClient):
    """Stores element payloads on disk instead of S3/GCS/Azure.

    Chainlit ships cloud clients only, and without one it logs
    "No blob_storage_client is configured" and drops every element. Live chats
    still show their sources — those are served from the running session — but
    reopening a conversation from the sidebar would leave the `[1]` links dead.

    `get_read_url` returns a path served by the static mount that `app.py` adds;
    the browser fetches it like any other URL.
    """

    def __init__(self, root: Path):
        self.root = root

    def _path(self, object_key: str) -> Path:
        # object keys come from Chainlit, but a traversal here would write
        # anywhere on disk, so resolve and confine it
        path = (self.root / object_key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"object key escapes the storage directory: {object_key!r}")
        return path

    async def upload_file(
        self,
        object_key: str,
        data: Union[bytes, str],
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> Dict[str, Any]:
        path = self._path(object_key)
        if path.exists() and not overwrite:
            return {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
        return {"object_key": object_key, "url": await self.get_read_url(object_key)}

    async def delete_file(self, object_key: str) -> bool:
        try:
            self._path(object_key).unlink(missing_ok=True)
            return True
        except (OSError, ValueError):
            return False

    async def get_read_url(self, object_key: str) -> str:
        return f"{ELEMENTS_ROUTE}/{object_key}"

    async def close(self) -> None:
        pass


def data_layer():
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    ensure_schema()
    return SQLAlchemyDataLayer(
        conninfo=f"sqlite+aiosqlite:///{db_path()}",
        storage_provider=LocalStorageClient(elements_dir()),
    )
