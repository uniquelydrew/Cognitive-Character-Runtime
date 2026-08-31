"""Versioned SQLite migrations for runtime data."""

from __future__ import annotations

import sqlite3


MIGRATIONS: list[tuple[int, str]] = [
    (1, """
        CREATE TABLE IF NOT EXISTS executive_claim_audit (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            character_id TEXT NOT NULL,
            claim_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('verified', 'rejected')),
            created_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
            FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_claim_audit_event ON executive_claim_audit(event_id, created_at);
    """),
    (2, """
        CREATE TRIGGER IF NOT EXISTS events_session_character_matches
        BEFORE INSERT ON events WHEN NEW.session_id IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM sessions WHERE id=NEW.session_id AND character_id=NEW.character_id)
        BEGIN SELECT RAISE(ABORT, 'event session must belong to its character'); END;
        CREATE TRIGGER IF NOT EXISTS event_links_character_matches
        BEFORE INSERT ON event_links
        WHEN NOT EXISTS (SELECT 1 FROM events WHERE id=NEW.from_event_id AND character_id=NEW.character_id)
          OR NOT EXISTS (SELECT 1 FROM events WHERE id=NEW.to_event_id AND character_id=NEW.character_id)
        BEGIN SELECT RAISE(ABORT, 'event links must remain within one character'); END;
        CREATE TRIGGER IF NOT EXISTS memories_supersession_character_matches
        BEFORE UPDATE OF superseded_by ON memories WHEN NEW.superseded_by IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM memories WHERE id=NEW.superseded_by AND character_id=NEW.character_id)
        BEGIN SELECT RAISE(ABORT, 'memory supersession must remain within one character'); END;
    """),
]


def apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    current = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    for version, sql in MIGRATIONS:
        if version not in current:
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'))", (version,))
