from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from services.common import (
    CharacterDocument,
    EpistemicType,
    EventRecord,
    MemoryRecord,
    MutationOperation,
    MutationProposal,
    ValidatedMutation,
)

DB_PATH = Path(os.getenv("MEMORY_DATABASE", "/data/cognition.db"))
CHARACTER_DIR = Path(os.getenv("CHARACTER_DIR", "/characters"))
PROFILE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    load_character_files()
    yield


app = FastAPI(title="Character Memory Service", version="0.1.0", lifespan=lifespan)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                document_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                closed_at TEXT,
                FOREIGN KEY(character_id) REFERENCES characters(id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                session_id TEXT,
                event_type TEXT NOT NULL,
                actor TEXT,
                content TEXT NOT NULL,
                topic TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(character_id) REFERENCES characters(id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_character_topic ON events(character_id, topic, created_at);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, created_at);

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                topic TEXT,
                content TEXT NOT NULL,
                epistemic_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                salience REAL NOT NULL,
                source_event_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'active',
                superseded_by TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(character_id) REFERENCES characters(id)
            );
            CREATE INDEX IF NOT EXISTS idx_memories_lookup ON memories(character_id, topic, status, salience);

            CREATE TABLE IF NOT EXISTS mutable_state (
                character_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(character_id, key),
                FOREIGN KEY(character_id) REFERENCES characters(id)
            );

            CREATE TABLE IF NOT EXISTS beliefs (
                character_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                epistemic_type TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '[]',
                revision INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(character_id, key),
                FOREIGN KEY(character_id) REFERENCES characters(id)
            );

            CREATE TABLE IF NOT EXISTS belief_history (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                key TEXT NOT NULL,
                old_value_json TEXT,
                new_value_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '[]',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS event_links (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                from_event_id TEXT NOT NULL,
                to_event_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                UNIQUE(character_id, from_event_id, to_event_id, relationship)
            );

            CREATE TABLE IF NOT EXISTS mutation_audit (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def load_character_files() -> None:
    if not CHARACTER_DIR.exists():
        return
    for path in sorted(CHARACTER_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "biography_file" in raw:
            bio_path = path.parent / raw.pop("biography_file")
            if bio_path.exists():
                raw["biography"] = bio_path.read_text(encoding="utf-8").strip()
        char = CharacterDocument.model_validate(raw)
        upsert_character(char, initialize=True)


def profile_path(character_id: str) -> Path:
    """Return the canonical YAML path, rejecting traversal and unstable IDs."""

    if not PROFILE_ID_RE.fullmatch(character_id):
        raise HTTPException(
            422,
            "Character IDs must start with a lowercase letter and use only lowercase letters, numbers, _ or -.",
        )
    return CHARACTER_DIR / f"{character_id}.yaml"


def write_character_source(char: CharacterDocument) -> Path:
    """Atomically persist the validated profile that bootstraps the runtime."""

    path = profile_path(char.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    source = yaml.safe_dump(
        char.model_dump(mode="json", exclude_none=True),
        allow_unicode=True,
        sort_keys=False,
    )
    try:
        temporary.write_text(source, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise HTTPException(503, "The canonical profile source could not be saved.") from exc
    return path


def profile_summary(char: CharacterDocument) -> dict[str, Any]:
    identity = char.identity
    return {
        "id": char.id,
        "name": str(identity.get("name", char.id)),
        "occupation": str(identity.get("occupation", "")),
        "faction": str(identity.get("faction", "")),
        "source_file": f"{char.id}.yaml",
    }


def read_character_source(character_id: str) -> CharacterDocument:
    """Read the editable canonical document rather than a runtime projection."""

    path = profile_path(character_id)
    if not path.exists():
        raise HTTPException(404, "Canonical profile source was not found.")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "biography_file" in raw:
            bio_path = path.parent / str(raw.pop("biography_file"))
            if bio_path.exists():
                raw["biography"] = bio_path.read_text(encoding="utf-8").strip()
        return CharacterDocument.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise HTTPException(422, "Canonical profile source is not a valid character document.") from exc


def upsert_character(char: CharacterDocument, initialize: bool = False) -> None:
    ts = now_iso()
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM characters WHERE id=?", (char.id,)).fetchone()
        conn.execute(
            """
            INSERT INTO characters(id, document_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET document_json=excluded.document_json, updated_at=excluded.updated_at
            """,
            (char.id, char.model_dump_json(), ts, ts),
        )
        if initialize and not exists:
            for key, value in char.mutable_state.items():
                conn.execute(
                    "INSERT OR REPLACE INTO mutable_state VALUES (?, ?, ?, ?)",
                    (char.id, key, json.dumps(value), ts),
                )
            for key, value in char.beliefs.items():
                conn.execute(
                    "INSERT OR REPLACE INTO beliefs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (char.id, key, json.dumps(value), 1.0, EpistemicType.BELIEF.value, "[]", 1, ts),
                )
            for goal in char.initial_goals:
                conn.execute(
                    "INSERT INTO goals VALUES (?, ?, ?, 'active', '{}', ?, ?)",
                    (f"goal_{uuid.uuid4().hex}", char.id, goal, ts, ts),
                )



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "database": str(DB_PATH)}


@app.get("/characters", response_model=list[CharacterDocument])
def list_characters() -> list[CharacterDocument]:
    with db() as conn:
        rows = conn.execute("SELECT document_json FROM characters ORDER BY id").fetchall()
    return [CharacterDocument.model_validate_json(r["document_json"]) for r in rows]


@app.get("/profiles")
def list_profiles() -> list[dict[str, Any]]:
    """List the editable source-backed profiles for Profile Studio."""

    return [profile_summary(character) for character in list_characters()]


@app.get("/profiles/{character_id}")
def get_profile(character_id: str) -> dict[str, Any]:
    """Return source document alongside its distinct live runtime state."""

    source = read_character_source(character_id)
    runtime = get_character(character_id)
    return {
        "source": source.model_dump(mode="json"),
        "source_file": f"{source.id}.yaml",
        "runtime": {
            "mutable_state": runtime["mutable_state"],
            "beliefs": runtime["beliefs"],
            "goals": runtime["goals"],
        },
    }


@app.post("/profiles", status_code=201)
def create_profile(profile: CharacterDocument) -> dict[str, Any]:
    """Create a validated profile and initialize a matching runtime record."""

    path = profile_path(profile.id)
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM characters WHERE id=?", (profile.id,)).fetchone()
    if exists or path.exists():
        raise HTTPException(409, "A profile with this ID already exists.")
    write_character_source(profile)
    upsert_character(profile, initialize=True)
    return get_profile(profile.id)


@app.put("/profiles/{character_id}")
def update_profile(character_id: str, profile: CharacterDocument) -> dict[str, Any]:
    """Update the canonical YAML and immediately refresh its runtime primer."""

    profile_path(character_id)
    if profile.id != character_id:
        raise HTTPException(422, "A profile ID cannot be renamed. Create a new profile instead.")
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM characters WHERE id=?", (character_id,)).fetchone()
    if not exists:
        raise HTTPException(404, "Character not found")
    write_character_source(profile)
    # This updates immutable/design-time profile data only. Conversation-derived
    # mutable state, beliefs, and goals remain separate runtime records.
    upsert_character(profile)
    return get_profile(profile.id)


@app.get("/characters/{character_id}")
def get_character(character_id: str) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT document_json FROM characters WHERE id=?", (character_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Character not found")
        char = CharacterDocument.model_validate_json(row["document_json"])
        mutable = {
            r["key"]: json.loads(r["value_json"])
            for r in conn.execute("SELECT key, value_json FROM mutable_state WHERE character_id=?", (character_id,))
        }
        beliefs = {
            r["key"]: {
                "value": json.loads(r["value_json"]),
                "confidence": r["confidence"],
                "epistemic_type": r["epistemic_type"],
                "evidence": json.loads(r["evidence_json"]),
                "revision": r["revision"],
            }
            for r in conn.execute("SELECT * FROM beliefs WHERE character_id=?", (character_id,))
        }
        goals = [dict(r) for r in conn.execute("SELECT * FROM goals WHERE character_id=? ORDER BY created_at", (character_id,))]
    return {"character": char.model_dump(), "mutable_state": mutable, "beliefs": beliefs, "goals": goals}


class SessionCreate(BaseModel):
    character_id: str


@app.post("/sessions")
def create_session(req: SessionCreate) -> dict[str, Any]:
    sid = f"sess_{uuid.uuid4().hex}"
    ts = now_iso()
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM characters WHERE id=?", (req.character_id,)).fetchone()
        if not exists:
            raise HTTPException(404, "Character not found")
        conn.execute(
            "INSERT INTO sessions(id, character_id, status, created_at) VALUES (?, ?, 'open', ?)",
            (sid, req.character_id, ts),
        )
    return {"id": sid, "character_id": req.character_id, "status": "open", "created_at": ts}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Session not found")
    return dict(row)


@app.get("/sessions/{session_id}/events")
def session_events(session_id: str) -> list[dict[str, Any]]:
    with db() as conn:
        session = conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        rows = conn.execute("SELECT * FROM events WHERE session_id=? ORDER BY created_at, rowid", (session_id,)).fetchall()
    return [
        {
            **{k: r[k] for k in r.keys() if k != "metadata_json"},
            "metadata": json.loads(r["metadata_json"]),
        }
        for r in rows
    ]


@app.post("/sessions/{session_id}/close")
def close_session(session_id: str) -> dict[str, str]:
    ts = now_iso()
    with db() as conn:
        row = conn.execute("SELECT status, closed_at FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Session not found")
        if row["status"] == "closed":
            return {"id": session_id, "status": "closed", "closed_at": row["closed_at"]}
        conn.execute("UPDATE sessions SET status='closed', closed_at=? WHERE id=?", (ts, session_id))
    return {"id": session_id, "status": "closed", "closed_at": ts}


@app.post("/events", response_model=EventRecord)
def add_event(event: EventRecord) -> EventRecord:
    eid = event.id or f"evt_{uuid.uuid4().hex}"
    ts = now_iso()
    with db() as conn:
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
            if session["status"] != "open":
                raise HTTPException(409, "Interaction is already closed")
        conn.execute(
            """
            INSERT INTO events(id, character_id, session_id, event_type, actor, content, topic, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eid,
                event.character_id,
                event.session_id,
                event.event_type,
                event.actor,
                event.content,
                event.topic,
                json.dumps(event.metadata),
                ts,
            ),
        )
    return event.model_copy(update={"id": eid})


@app.get("/interaction-history/{character_id}")
def interaction_history(
    character_id: str,
    topic: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM events
            WHERE character_id=? AND topic=? AND event_type IN ('user_message', 'character_message')
            ORDER BY created_at DESC, rowid DESC LIMIT ?
            """,
            (character_id, topic, limit),
        ).fetchall()
    raw_events = [dict(r) for r in reversed(rows)]
    events = [{**row, "metadata": json.loads(row["metadata_json"])} for row in raw_events]
    answered_user_ids = {
        str(event["metadata"].get("responds_to"))
        for event in events
        if event["event_type"] == "character_message" and event["metadata"].get("responds_to")
    }
    # An upstream model failure can leave a user event without a character
    # response. It remains available in the raw session audit trail, but must not
    # count as an answered question or make a retry look adversarial.
    completed_events = [
        event
        for event in events
        if event["event_type"] == "character_message" or event["id"] in answered_user_ids
    ]
    user_questions = [event for event in completed_events if event["event_type"] == "user_message"]
    prior_answer = None
    for event in reversed(completed_events):
        if event["event_type"] == "character_message":
            prior_answer = event["content"]
            break
    return {
        "topic": topic,
        "times_asked": len(user_questions),
        "prior_answer": prior_answer,
        "events": [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "actor": r["actor"],
                "content": r["content"],
                "topic": r["topic"],
                "metadata": r["metadata"],
                "created_at": r["created_at"],
            }
            for r in completed_events
        ],
    }


@app.post("/memories", response_model=MemoryRecord)
def add_memory(memory: MemoryRecord) -> MemoryRecord:
    mid = memory.id or f"mem_{uuid.uuid4().hex}"
    ts = now_iso()
    with db() as conn:
        character = conn.execute("SELECT 1 FROM characters WHERE id=?", (memory.character_id,)).fetchone()
        if not character:
            raise HTTPException(404, "Character not found")
        conn.execute(
            """
            INSERT INTO memories(
                id, character_id, kind, topic, content, epistemic_type, confidence, salience,
                source_event_ids_json, status, superseded_by, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mid, memory.character_id, memory.kind, memory.topic, memory.content,
                memory.epistemic_type.value, memory.confidence, memory.salience,
                json.dumps(memory.source_event_ids), memory.status, memory.superseded_by,
                json.dumps(memory.metadata), ts, ts,
            ),
        )
    return memory.model_copy(update={"id": mid})


@app.get("/memories/{character_id}")
def get_memories(
    character_id: str,
    topic: str | None = None,
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    clauses = ["character_id=?", "status='active'"]
    args: list[Any] = [character_id]
    if topic:
        clauses.append("(topic=? OR topic IS NULL)")
        args.append(topic)
    args.append(limit)
    with db() as conn:
        rows = conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY salience DESC, created_at DESC LIMIT ?",
            args,
        ).fetchall()
    return [
        {
            "id": r["id"], "character_id": r["character_id"], "kind": r["kind"], "topic": r["topic"],
            "content": r["content"], "epistemic_type": r["epistemic_type"], "confidence": r["confidence"],
            "salience": r["salience"], "source_event_ids": json.loads(r["source_event_ids_json"]),
            "status": r["status"], "superseded_by": r["superseded_by"], "metadata": json.loads(r["metadata_json"]),
            "created_at": r["created_at"], "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def validate_proposal(proposal: MutationProposal) -> ValidatedMutation:
    if proposal.operation == MutationOperation.UPDATE_CORE:
        return ValidatedMutation(proposal=proposal, status="rejected", reason="Core biography/identity is immutable at runtime.")
    if proposal.operation in {MutationOperation.SET_BELIEF, MutationOperation.SET_MUTABLE_STATE, MutationOperation.UPDATE_GOAL}:
        if not proposal.evidence:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="State revisions require provenance/evidence.")
        return ValidatedMutation(proposal=proposal, status="versioned", reason="Mutable state may change, but revision history is preserved.")
    if proposal.operation == MutationOperation.ADD_MEMORY:
        if proposal.epistemic_type in {EpistemicType.INFERENCE, EpistemicType.SUSPICION, EpistemicType.BELIEF} and not proposal.evidence:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Derived memories require source evidence.")
        return ValidatedMutation(proposal=proposal, status="allowed", reason="Append-only derived memory is permitted.")
    return ValidatedMutation(proposal=proposal, status="allowed", reason="Operation is permitted by runtime policy.")


class MutationBatch(BaseModel):
    proposals: list[MutationProposal] = Field(default_factory=list)


@app.post("/mutations/{character_id}")
def apply_mutations(character_id: str, batch: MutationBatch) -> list[ValidatedMutation]:
    results: list[ValidatedMutation] = []
    ts = now_iso()
    with db() as conn:
        character = conn.execute("SELECT 1 FROM characters WHERE id=?", (character_id,)).fetchone()
        if not character:
            raise HTTPException(404, "Character not found")
        for proposal in batch.proposals:
            checked = validate_proposal(proposal)
            results.append(checked)
            conn.execute(
                "INSERT INTO mutation_audit VALUES (?, ?, ?, ?, ?, ?)",
                (f"mut_{uuid.uuid4().hex}", character_id, proposal.model_dump_json(), checked.status, checked.reason, ts),
            )
            if checked.status == "rejected":
                continue

            if proposal.operation == MutationOperation.SET_MUTABLE_STATE:
                conn.execute(
                    """
                    INSERT INTO mutable_state(character_id, key, value_json, updated_at) VALUES (?, ?, ?, ?)
                    ON CONFLICT(character_id, key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                    """,
                    (character_id, proposal.target, json.dumps(proposal.value), ts),
                )
            elif proposal.operation == MutationOperation.SET_BELIEF:
                old = conn.execute(
                    "SELECT value_json, revision FROM beliefs WHERE character_id=? AND key=?",
                    (character_id, proposal.target),
                ).fetchone()
                revision = int(old["revision"]) + 1 if old else 1
                old_json = old["value_json"] if old else None
                conn.execute(
                    """
                    INSERT INTO beliefs(character_id, key, value_json, confidence, epistemic_type, evidence_json, revision, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(character_id, key) DO UPDATE SET
                      value_json=excluded.value_json, confidence=excluded.confidence,
                      epistemic_type=excluded.epistemic_type, evidence_json=excluded.evidence_json,
                      revision=excluded.revision, updated_at=excluded.updated_at
                    """,
                    (
                        character_id, proposal.target, json.dumps(proposal.value), proposal.confidence,
                        proposal.epistemic_type.value, json.dumps(proposal.evidence), revision, ts,
                    ),
                )
                conn.execute(
                    "INSERT INTO belief_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"bh_{uuid.uuid4().hex}", character_id, proposal.target, old_json,
                        json.dumps(proposal.value), proposal.confidence, json.dumps(proposal.evidence), proposal.reason, ts,
                    ),
                )
            elif proposal.operation == MutationOperation.ADD_MEMORY:
                conn.execute(
                    """
                    INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, '{}', ?, ?)
                    """,
                    (
                        f"mem_{uuid.uuid4().hex}", character_id, proposal.target, proposal.target,
                        str(proposal.value), proposal.epistemic_type.value, proposal.confidence, 0.6,
                        json.dumps(proposal.evidence), ts, ts,
                    ),
                )
            elif proposal.operation == MutationOperation.ADD_GOAL:
                conn.execute(
                    "INSERT INTO goals VALUES (?, ?, ?, 'active', '{}', ?, ?)",
                    (f"goal_{uuid.uuid4().hex}", character_id, str(proposal.value), ts, ts),
                )
            elif proposal.operation == MutationOperation.UPDATE_GOAL:
                conn.execute(
                    "UPDATE goals SET status=?, updated_at=? WHERE id=? AND character_id=?",
                    (str(proposal.value), ts, proposal.target, character_id),
                )
            elif proposal.operation == MutationOperation.LINK_EVENTS:
                data = proposal.value if isinstance(proposal.value, dict) else {}
                from_id, to_id = data.get("from"), data.get("to")
                relationship = data.get("relationship", "related_to")
                if from_id and to_id:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO event_links VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"lnk_{uuid.uuid4().hex}", character_id, from_id, to_id, relationship,
                            json.dumps(proposal.evidence), ts,
                        ),
                    )
            elif proposal.operation == MutationOperation.SUPERSEDE_MEMORY:
                conn.execute(
                    "UPDATE memories SET status='superseded', superseded_by=?, updated_at=? WHERE id=? AND character_id=?",
                    (str(proposal.value), ts, proposal.target, character_id),
                )
    return results


@app.get("/debug/{character_id}")
def debug_state(character_id: str) -> dict[str, Any]:
    with db() as conn:
        events = [dict(r) for r in conn.execute("SELECT * FROM events WHERE character_id=? ORDER BY created_at, rowid", (character_id,))]
        memories = [dict(r) for r in conn.execute("SELECT * FROM memories WHERE character_id=? ORDER BY created_at, rowid", (character_id,))]
        mutations = [dict(r) for r in conn.execute("SELECT * FROM mutation_audit WHERE character_id=? ORDER BY created_at, rowid", (character_id,))]
        links = [dict(r) for r in conn.execute("SELECT * FROM event_links WHERE character_id=? ORDER BY created_at, rowid", (character_id,))]
    return {"events": events, "memories": memories, "mutations": mutations, "links": links}
