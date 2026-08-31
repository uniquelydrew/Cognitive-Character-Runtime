"""Event and memory persistence operations for the memory service.

These functions deliberately receive a caller-owned SQLite transaction.  A
turn can therefore create its reply event, derived memories, and mutations in
one atomic commit without an HTTP route owning database details.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from fastapi import HTTPException

from services.common import EventRecord, MemoryRecord
from services.memory.storage import now_iso


def add_event(event: EventRecord, conn: sqlite3.Connection) -> EventRecord:
    """Validate and insert an event in an existing transaction."""

    event_id = event.id or f"evt_{uuid.uuid4().hex}"
    character = conn.execute("SELECT 1 FROM characters WHERE id=?", (event.character_id,)).fetchone()
    if not character:
        raise HTTPException(404, "Character not found")
    if event.session_id:
        session = conn.execute(
            "SELECT character_id, status FROM sessions WHERE id=?", (event.session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        if session["character_id"] != event.character_id:
            raise HTTPException(409, "Event character does not match session character")
        if session["status"] != "open" and event.event_type != "reflection":
            raise HTTPException(409, "Interaction is already closed")
    try:
        metadata_json = json.dumps(event.metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "Event metadata must be JSON-safe.") from exc
    conn.execute(
        """
        INSERT INTO events(id, character_id, session_id, event_type, actor, content, topic, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (event_id, event.character_id, event.session_id, event.event_type, event.actor,
         event.content, event.topic, metadata_json, now_iso()),
    )
    return event.model_copy(update={"id": event_id})


def interaction_history(
    character_id: str, topic: str, limit: int, conn: sqlite3.Connection
) -> dict[str, Any]:
    """Return answered interaction history without counting failed turns."""

    rows = conn.execute(
        """
        SELECT * FROM events
        WHERE character_id=? AND topic=? AND event_type IN ('user_message', 'character_message')
        ORDER BY created_at DESC, rowid DESC LIMIT ?
        """, (character_id, topic, limit),
    ).fetchall()
    events = [{**dict(row), "metadata": json.loads(row["metadata_json"])} for row in reversed(rows)]
    answered_user_ids = {
        str(event["metadata"].get("responds_to")) for event in events
        if event["event_type"] == "character_message" and event["metadata"].get("responds_to")
    }
    completed_events = [
        event for event in events
        if event["event_type"] == "character_message" or event["id"] in answered_user_ids
    ]
    prior_answer = next(
        (event["content"] for event in reversed(completed_events)
         if event["event_type"] == "character_message"), None,
    )
    return {
        "topic": topic,
        "times_asked": sum(event["event_type"] == "user_message" for event in completed_events),
        "prior_answer": prior_answer,
        "events": [{key: event[key] for key in (
            "id", "event_type", "actor", "content", "topic", "metadata", "created_at"
        )} for event in completed_events],
    }


def add_memory(memory: MemoryRecord, conn: sqlite3.Connection) -> MemoryRecord:
    """Validate and insert a memory in an existing transaction."""

    memory_id = memory.id or f"mem_{uuid.uuid4().hex}"
    character = conn.execute("SELECT 1 FROM characters WHERE id=?", (memory.character_id,)).fetchone()
    if not character:
        raise HTTPException(404, "Character not found")
    source_ids = [source_id.strip() for source_id in memory.source_event_ids]
    if len(source_ids) != len(set(source_ids)) or any(not source_id for source_id in source_ids):
        raise HTTPException(422, "Memory provenance must contain distinct, non-empty event IDs.")
    if source_ids:
        placeholders = ", ".join("?" for _ in source_ids)
        rows = conn.execute(
            f"SELECT id FROM events WHERE character_id=? AND id IN ({placeholders})",
            [memory.character_id, *source_ids],
        ).fetchall()
        if {str(row["id"]) for row in rows} != set(source_ids):
            raise HTTPException(422, "Memory provenance must reference recorded events for this character.")
    try:
        metadata_json = json.dumps(memory.metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "Memory metadata must be JSON-safe.") from exc
    timestamp = now_iso()
    conn.execute(
        """
        INSERT INTO memories(
            id, character_id, kind, topic, content, epistemic_type, confidence, salience,
            source_event_ids_json, status, superseded_by, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, memory.character_id, memory.kind, memory.topic, memory.content,
         memory.epistemic_type.value, memory.confidence, memory.salience, json.dumps(source_ids),
         memory.status, memory.superseded_by, metadata_json, timestamp, timestamp),
    )
    return memory.model_copy(update={"id": memory_id})


def get_memories(
    character_id: str, topic: str | None, limit: int, conn: sqlite3.Connection
) -> list[dict[str, Any]]:
    clauses = ["character_id=?", "status='active'"]
    args: list[Any] = [character_id]
    if topic:
        clauses.append("(topic=? OR topic IS NULL)")
        args.append(topic)
    rows = conn.execute(
        f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY salience DESC, created_at DESC LIMIT ?",
        [*args, limit],
    ).fetchall()
    return [{
        "id": row["id"], "character_id": row["character_id"], "kind": row["kind"],
        "topic": row["topic"], "content": row["content"], "epistemic_type": row["epistemic_type"],
        "confidence": row["confidence"], "salience": row["salience"],
        "source_event_ids": json.loads(row["source_event_ids_json"]), "status": row["status"],
        "superseded_by": row["superseded_by"], "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    } for row in rows]
