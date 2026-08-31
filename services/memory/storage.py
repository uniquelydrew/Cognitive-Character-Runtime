"""SQLite connection and transaction primitives for the memory service."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection(path: Path, *, before_commit: Any | None = None, on_abort: Any | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        if before_commit is not None:
            before_commit()
        conn.commit()
    except Exception:
        conn.rollback()
        if on_abort is not None:
            on_abort()
        raise
    finally:
        conn.close()
