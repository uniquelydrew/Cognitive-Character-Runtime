from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from yaml.tokens import AliasToken
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from services.common import (
    CharacterDocument,
    EpistemicType,
    EventRecord,
    GeneralKnowledgeRecord,
    KnowledgeCatalog,
    KnowledgeClassification,
    MemoryRecord,
    MutationOperation,
    MutationProposal,
    ValidatedMutation,
)
from services.memory.migrations import apply_migrations

DB_PATH = Path(os.getenv("MEMORY_DATABASE", "/data/cognition.db"))
CHARACTER_DIR = Path(os.getenv("CHARACTER_DIR", "/characters"))
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", "/knowledge"))
PROFILE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
SNAPSHOT_FORMAT = "cognitive-character-runtime/character-snapshot/v1"
MAX_SESSION_EVENTS_RETURNED = int(os.getenv("MAX_SESSION_EVENTS_RETURNED", "200"))
MANAGED_KNOWLEDGE_FILENAME = "catalog.yaml"
KNOWLEDGE_SAMPLE_FILENAME = "catalog.example.yaml"

if MAX_SESSION_EVENTS_RETURNED < 20:
    raise RuntimeError("MAX_SESSION_EVENTS_RETURNED must be at least 20")

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    load_character_files()
    load_knowledge_files()
    yield


app = FastAPI(title="Character Memory Service", version="0.1.0", lifespan=lifespan)


class ProfileImportRequest(BaseModel):
    """YAML submitted by Profile Studio for a source or full snapshot import."""

    yaml: str = Field(min_length=1, max_length=5_000_000)


class ProfileDiffRequest(BaseModel):
    """An earlier exported snapshot to compare with the current profile state."""

    yaml: str = Field(min_length=1, max_length=5_000_000)


class KnowledgeCatalogRequest(BaseModel):
    """A complete general-knowledge catalog submitted by Knowledge Studio."""

    yaml: str = Field(min_length=1, max_length=5_000_000)


class ReflectionRetrySchedule(BaseModel):
    """A safe, bounded explanation for a reflection queued after close."""

    error: str = Field(min_length=1, max_length=1_000)


class ReflectionRetryClaim(BaseModel):
    lease_seconds: int = Field(default=300, ge=30, le=3_600)


class SnapshotModel(BaseModel):
    """Strict, JSON-safe import contract for data that bypasses normal APIs."""

    model_config = {"extra": "forbid"}


def _validate_timestamp(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("must be an ISO-8601 timestamp") from exc
    return value


def _validate_optional_timestamp(value: str | None) -> str | None:
    return _validate_timestamp(value) if value is not None else None


class SnapshotCharacterMeta(SnapshotModel):
    created_at: str
    updated_at: str

    _created_at = field_validator("created_at")(_validate_timestamp)
    _updated_at = field_validator("updated_at")(_validate_timestamp)


class SnapshotBelief(SnapshotModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    epistemic_type: EpistemicType
    evidence: list[str] = Field(default_factory=list, max_length=100)
    revision: int = Field(ge=1)


class SnapshotGoal(SnapshotModel):
    id: str = Field(min_length=1, max_length=120)
    character_id: str
    goal: str = Field(min_length=1, max_length=4_000)
    status: str = Field(min_length=1, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    _created_at = field_validator("created_at")(_validate_timestamp)
    _updated_at = field_validator("updated_at")(_validate_timestamp)


class SnapshotSession(SnapshotModel):
    id: str = Field(min_length=1, max_length=120)
    character_id: str
    status: Literal["open", "closed"]
    created_at: str
    closed_at: str | None = None

    _created_at = field_validator("created_at")(_validate_timestamp)
    _closed_at = field_validator("closed_at")(_validate_optional_timestamp)


class SnapshotEvent(SnapshotModel):
    id: str = Field(min_length=1, max_length=120)
    character_id: str
    session_id: str | None = Field(default=None, max_length=120)
    event_type: str = Field(min_length=1, max_length=120)
    actor: str | None = Field(default=None, max_length=120)
    content: str = Field(min_length=1, max_length=8_000)
    topic: str | None = Field(default=None, max_length=160)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    _created_at = field_validator("created_at")(_validate_timestamp)


class SnapshotMemory(SnapshotModel):
    id: str = Field(min_length=1, max_length=120)
    character_id: str
    kind: str = Field(min_length=1, max_length=120)
    topic: str | None = Field(default=None, max_length=160)
    content: str = Field(min_length=1, max_length=8_000)
    epistemic_type: EpistemicType
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    salience: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source_event_ids: list[str] = Field(default_factory=list, max_length=200)
    status: Literal["active", "superseded"]
    superseded_by: str | None = Field(default=None, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    _created_at = field_validator("created_at")(_validate_timestamp)
    _updated_at = field_validator("updated_at")(_validate_timestamp)


class SnapshotEventLink(SnapshotModel):
    id: str = Field(min_length=1, max_length=120)
    character_id: str
    from_event_id: str = Field(min_length=1, max_length=120)
    to_event_id: str = Field(min_length=1, max_length=120)
    relationship: str = Field(min_length=1, max_length=120)
    evidence: list[str] = Field(default_factory=list, max_length=100)
    created_at: str

    _created_at = field_validator("created_at")(_validate_timestamp)


class SnapshotMutationAudit(SnapshotModel):
    id: str = Field(min_length=1, max_length=120)
    character_id: str
    proposal: MutationProposal
    status: Literal["allowed", "versioned", "rejected"]
    reason: str = Field(max_length=4_000)
    created_at: str

    _created_at = field_validator("created_at")(_validate_timestamp)


class SnapshotBeliefHistory(SnapshotModel):
    id: str = Field(min_length=1, max_length=120)
    character_id: str
    key: str = Field(min_length=1, max_length=160)
    old_value: Any = None
    new_value: Any
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence: list[str] = Field(default_factory=list, max_length=100)
    reason: str = Field(max_length=4_000)
    created_at: str

    _created_at = field_validator("created_at")(_validate_timestamp)


class SnapshotState(SnapshotModel):
    mutable_state: dict[str, Any]
    beliefs: dict[str, SnapshotBelief]
    goals: list[SnapshotGoal] = Field(default_factory=list, max_length=50_000)


class SnapshotRuntime(SnapshotModel):
    character: SnapshotCharacterMeta
    state: SnapshotState
    memories: list[SnapshotMemory] = Field(default_factory=list, max_length=50_000)
    sessions: list[SnapshotSession] = Field(default_factory=list, max_length=50_000)
    events: list[SnapshotEvent] = Field(default_factory=list, max_length=50_000)
    event_links: list[SnapshotEventLink] = Field(default_factory=list, max_length=50_000)
    mutation_audit: list[SnapshotMutationAudit] = Field(default_factory=list, max_length=50_000)
    belief_history: list[SnapshotBeliefHistory] = Field(default_factory=list, max_length=50_000)

    @model_validator(mode="after")
    def validate_references_and_json(self) -> "SnapshotRuntime":
        collections = {
            "goals": self.state.goals,
            "memories": self.memories,
            "sessions": self.sessions,
            "events": self.events,
            "event_links": self.event_links,
            "mutation_audit": self.mutation_audit,
            "belief_history": self.belief_history,
        }
        for label, records in collections.items():
            ids = [record.id for record in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} contains duplicate IDs")
        event_ids = {record.id for record in self.events}
        session_ids = {record.id for record in self.sessions}
        memory_ids = {record.id for record in self.memories}
        if any(record.session_id and record.session_id not in session_ids for record in self.events):
            raise ValueError("events reference sessions not included in the snapshot")
        if any(
            event_id not in event_ids
            for record in self.memories
            for event_id in record.source_event_ids
        ):
            raise ValueError("memories reference events not included in the snapshot")
        if any(record.superseded_by and record.superseded_by not in memory_ids for record in self.memories):
            raise ValueError("memories reference a replacement not included in the snapshot")
        if any(
            record.from_event_id not in event_ids or record.to_event_id not in event_ids
            for record in self.event_links
        ):
            raise ValueError("event links reference events not included in the snapshot")
        try:
            json.dumps(self.model_dump(mode="json"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("snapshot contains a value that cannot be stored as JSON") from exc
        return self


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def db(*, before_commit: Any | None = None, on_abort: Any | None = None):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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

            CREATE TABLE IF NOT EXISTS reflection_jobs (
                session_id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                next_attempt_at TEXT NOT NULL,
                lease_expires_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(character_id) REFERENCES characters(id)
            );
            CREATE INDEX IF NOT EXISTS idx_reflection_jobs_due
                ON reflection_jobs(status, next_attempt_at, lease_expires_at);

            CREATE TABLE IF NOT EXISTS knowledge_classifications (
                id TEXT PRIMARY KEY,
                document_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_records (
                id TEXT PRIMARY KEY,
                document_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_record_labels (
                record_id TEXT NOT NULL,
                label_id TEXT NOT NULL,
                PRIMARY KEY(record_id, label_id),
                FOREIGN KEY(record_id) REFERENCES knowledge_records(id)
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_record_labels_label ON knowledge_record_labels(label_id, record_id);
            """
        )
        apply_migrations(conn)


def load_character_files() -> None:
    if not CHARACTER_DIR.exists():
        return
    for path in sorted(CHARACTER_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "biography_file" in raw:
            bio_path = _safe_biography_path(path, raw.pop("biography_file"))
            if bio_path.exists():
                raw["biography"] = bio_path.read_text(encoding="utf-8").strip()
        char = CharacterDocument.model_validate(raw)
        upsert_character(char, initialize=True)


def _validate_knowledge_taxonomy(
    classifications: list[KnowledgeClassification], records: list[GeneralKnowledgeRecord]
) -> None:
    nodes = {node.id: node for node in classifications}
    if len(nodes) != len(classifications):
        raise ValueError("Knowledge classifications contain duplicate IDs.")
    record_ids = [record.id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Knowledge records contain duplicate IDs.")
    for node in classifications:
        unknown = set(node.parents) - set(nodes)
        if unknown:
            raise ValueError(f"Knowledge classification {node.id} references unknown parents: {sorted(unknown)}")
    for record in records:
        referenced = set(record.labels) | set(record.access.require_all) | set(record.access.require_any)
        unknown = referenced - set(nodes)
        if unknown:
            raise ValueError(f"Knowledge record {record.id} references unknown labels: {sorted(unknown)}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise ValueError(f"Knowledge taxonomy contains a cycle at {node_id}.")
        visiting.add(node_id)
        for parent in nodes[node_id].parents:
            visit(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)


def _knowledge_source_paths() -> list[Path]:
    """Select the editable managed catalog, or fall back to bundled source files."""

    if not KNOWLEDGE_DIR.exists():
        return []
    managed = KNOWLEDGE_DIR / MANAGED_KNOWLEDGE_FILENAME
    if managed.exists():
        return [managed]
    return [
        path
        for path in sorted(KNOWLEDGE_DIR.glob("*.yaml"))
        if not path.name.endswith(".example.yaml")
    ]


def _catalog_from_source_paths(paths: list[Path]) -> KnowledgeCatalog:
    classifications: list[KnowledgeClassification] = []
    records: list[GeneralKnowledgeRecord] = []
    for path in paths:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            catalog = KnowledgeCatalog.model_validate(raw)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise RuntimeError(f"Knowledge source {path.name} is invalid: {exc}") from exc
        classifications.extend(catalog.classifications)
        records.extend(catalog.records)
    _validate_knowledge_taxonomy(classifications, records)
    return KnowledgeCatalog(classifications=classifications, records=records)


def _replace_knowledge_catalog(catalog: KnowledgeCatalog, conn: sqlite3.Connection) -> None:
    """Replace the indexed corpus inside an existing transaction."""

    conn.execute("DELETE FROM knowledge_record_labels")
    conn.execute("DELETE FROM knowledge_records")
    conn.execute("DELETE FROM knowledge_classifications")
    for node in catalog.classifications:
        conn.execute(
            "INSERT INTO knowledge_classifications(id, document_json) VALUES (?, ?)",
            (node.id, node.model_dump_json()),
        )
    for record in catalog.records:
        conn.execute(
            "INSERT INTO knowledge_records(id, document_json) VALUES (?, ?)",
            (record.id, record.model_dump_json()),
        )
        conn.executemany(
            "INSERT INTO knowledge_record_labels(record_id, label_id) VALUES (?, ?)",
            [(record.id, label) for label in record.labels],
        )


def load_knowledge_files() -> None:
    """Load general knowledge from its canonical managed or starter source files."""

    catalog = _catalog_from_source_paths(_knowledge_source_paths())
    with db() as conn:
        _replace_knowledge_catalog(catalog, conn)


def profile_path(character_id: str) -> Path:
    """Return the canonical YAML path, rejecting traversal and unstable IDs."""

    if not PROFILE_ID_RE.fullmatch(character_id):
        raise HTTPException(
            422,
            "Character IDs must start with a lowercase letter and use only lowercase letters, numbers, _ or -.",
        )
    return CHARACTER_DIR / f"{character_id}.yaml"


def _safe_biography_path(profile: Path, raw_path: Any) -> Path:
    """Allow legacy sidecar biographies without allowing profile-file traversal."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HTTPException(422, "biography_file must be a non-empty relative path.")
    root = profile.parent.resolve()
    candidate = (root / raw_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(422, "biography_file must stay inside the character source directory.")
    return candidate


def _character_source_text(char: CharacterDocument) -> str:
    return yaml.safe_dump(
        char.model_dump(mode="json", exclude_none=True),
        allow_unicode=True,
        sort_keys=False,
    )


def _stage_character_source(char: CharacterDocument) -> tuple[Path, Path]:
    """Write a validated source into a sibling staging file for atomic replacement."""

    path = profile_path(char.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".yaml.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(_character_source_text(char), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(503, "The canonical profile source could not be saved.") from exc
    return path, temporary


def _restore_source_bytes(path: Path, original: bytes | None) -> None:
    """Compensate a failed cross-store snapshot transaction without silent drift."""

    try:
        if original is None:
            if path.exists():
                path.unlink()
            return
        temporary = path.with_suffix(f".yaml.{uuid.uuid4().hex}.rollback")
        temporary.write_bytes(original)
        temporary.replace(path)
    except OSError as exc:
        raise RuntimeError("Could not restore the canonical source after a failed snapshot import.") from exc


def write_character_source(char: CharacterDocument) -> Path:
    """Atomically persist the validated profile that bootstraps the runtime."""

    path, temporary = _stage_character_source(char)
    try:
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
            bio_path = _safe_biography_path(path, raw.pop("biography_file"))
            if bio_path.exists():
                raw["biography"] = bio_path.read_text(encoding="utf-8").strip()
        return CharacterDocument.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise HTTPException(422, "Canonical profile source is not a valid character document.") from exc


def upsert_character(
    char: CharacterDocument,
    initialize: bool = False,
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Upsert a runtime primer, optionally inside a caller-owned transaction."""

    if conn is None:
        with db() as owned_connection:
            upsert_character(char, initialize=initialize, conn=owned_connection)
        return
    ts = now_iso()
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


def persist_profile_and_runtime(profile: CharacterDocument, *, initialize: bool) -> None:
    """Commit profile YAML and its DB primer together, with source rollback on failure."""

    path, temporary = _stage_character_source(profile)
    original = path.read_bytes() if path.exists() else None
    source_replaced = False

    def commit_source() -> None:
        nonlocal source_replaced
        try:
            temporary.replace(path)
            source_replaced = True
        except OSError as exc:
            raise HTTPException(503, "The canonical profile source could not be saved.") from exc

    def abort_source() -> None:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        if source_replaced:
            _restore_source_bytes(path, original)

    try:
        with db(before_commit=commit_source, on_abort=abort_source) as conn:
            upsert_character(profile, initialize=initialize, conn=conn)
    except Exception:
        # If validation or a DB operation failed before the commit hook, the
        # staging file has no authority and must not accumulate on disk.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise



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
    persist_profile_and_runtime(profile, initialize=True)
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
    # This updates immutable/design-time profile data only. Conversation-derived
    # mutable state, beliefs, and goals remain separate runtime records.
    persist_profile_and_runtime(profile, initialize=False)
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


def _snapshot_runtime(character_id: str) -> dict[str, Any]:
    """Return every durable runtime record in a readable, diff-friendly shape."""

    state = get_character(character_id)
    with db() as conn:
        character = conn.execute(
            "SELECT created_at, updated_at FROM characters WHERE id=?", (character_id,)
        ).fetchone()
        if not character:
            raise HTTPException(404, "Character not found")
        sessions = [dict(row) for row in conn.execute(
            "SELECT * FROM sessions WHERE character_id=? ORDER BY created_at, rowid", (character_id,)
        )]
        event_rows = conn.execute(
            "SELECT * FROM events WHERE character_id=? ORDER BY created_at, rowid", (character_id,)
        ).fetchall()
        memory_rows = conn.execute(
            "SELECT * FROM memories WHERE character_id=? ORDER BY created_at, rowid", (character_id,)
        ).fetchall()
        goal_rows = conn.execute(
            "SELECT * FROM goals WHERE character_id=? ORDER BY created_at, rowid", (character_id,)
        ).fetchall()
        link_rows = conn.execute(
            "SELECT * FROM event_links WHERE character_id=? ORDER BY created_at, rowid", (character_id,)
        ).fetchall()
        mutation_rows = conn.execute(
            "SELECT * FROM mutation_audit WHERE character_id=? ORDER BY created_at, rowid", (character_id,)
        ).fetchall()
        belief_history_rows = conn.execute(
            "SELECT * FROM belief_history WHERE character_id=? ORDER BY created_at, rowid", (character_id,)
        ).fetchall()

    return {
        "character": dict(character),
        "state": {
            "mutable_state": state["mutable_state"],
            "beliefs": state["beliefs"],
            "goals": [
                {
                    **{key: row[key] for key in row.keys() if key != "metadata_json"},
                    "metadata": json.loads(row["metadata_json"]),
                }
                for row in goal_rows
            ],
        },
        "memories": [
            {
                **{
                    key: row[key]
                    for key in row.keys()
                    if key not in {"source_event_ids_json", "metadata_json"}
                },
                "source_event_ids": json.loads(row["source_event_ids_json"]),
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in memory_rows
        ],
        "sessions": sessions,
        "events": [
            {
                **{key: row[key] for key in row.keys() if key != "metadata_json"},
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in event_rows
        ],
        "event_links": [
            {
                **{key: row[key] for key in row.keys() if key != "evidence_json"},
                "evidence": json.loads(row["evidence_json"]),
            }
            for row in link_rows
        ],
        "mutation_audit": [
            {
                **{key: row[key] for key in row.keys() if key != "proposal_json"},
                "proposal": json.loads(row["proposal_json"]),
            }
            for row in mutation_rows
        ],
        "belief_history": [
            {
                **{
                    key: row[key]
                    for key in row.keys()
                    if key not in {"old_value_json", "new_value_json", "evidence_json"}
                },
                "old_value": json.loads(row["old_value_json"]) if row["old_value_json"] is not None else None,
                "new_value": json.loads(row["new_value_json"]),
                "evidence": json.loads(row["evidence_json"]),
            }
            for row in belief_history_rows
        ],
    }


def character_snapshot(character_id: str) -> dict[str, Any]:
    """Build a portable YAML payload containing primer plus learned runtime state."""

    source = read_character_source(character_id)
    return {
        "format": SNAPSHOT_FORMAT,
        "source": source.model_dump(mode="json", exclude_none=True),
        "runtime": _snapshot_runtime(character_id),
    }


def _knowledge_catalog_text(catalog: KnowledgeCatalog) -> str:
    return yaml.safe_dump(
        catalog.model_dump(mode="json", exclude_none=True),
        allow_unicode=True,
        sort_keys=False,
    )


def _knowledge_catalog_summary(catalog: KnowledgeCatalog) -> dict[str, int]:
    return {
        "classification_count": len(catalog.classifications),
        "record_count": len(catalog.records),
        "assertion_count": sum(len(record.assertions) for record in catalog.records),
    }


def current_knowledge_catalog() -> KnowledgeCatalog:
    """Return the complete static corpus, never character-specific retrieval output."""

    with db() as conn:
        classifications = [
            KnowledgeClassification.model_validate_json(row["document_json"])
            for row in conn.execute(
                "SELECT document_json FROM knowledge_classifications ORDER BY id"
            )
        ]
        records = [
            GeneralKnowledgeRecord.model_validate_json(row["document_json"])
            for row in conn.execute("SELECT document_json FROM knowledge_records ORDER BY id")
        ]
    catalog = KnowledgeCatalog(classifications=classifications, records=records)
    _validate_knowledge_taxonomy(catalog.classifications, catalog.records)
    return catalog


def _catalog_from_uploaded_yaml(source_text: str) -> KnowledgeCatalog:
    raw = _safe_load_uploaded_yaml(source_text)
    try:
        catalog = KnowledgeCatalog.model_validate(raw)
        _validate_knowledge_taxonomy(catalog.classifications, catalog.records)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, f"Knowledge catalog is invalid: {exc}") from exc
    return catalog


def _write_managed_knowledge_catalog(catalog: KnowledgeCatalog) -> Path:
    """Atomically activate one editable catalog and replace its runtime index."""

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    path = KNOWLEDGE_DIR / MANAGED_KNOWLEDGE_FILENAME
    original = path.read_bytes() if path.exists() else None
    temporary = path.with_suffix(f".yaml.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(_knowledge_catalog_text(catalog), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(503, "The managed knowledge source could not be saved.") from exc
    source_replaced = False

    def commit_source() -> None:
        nonlocal source_replaced
        try:
            temporary.replace(path)
            source_replaced = True
        except OSError as exc:
            raise HTTPException(503, "The managed knowledge source could not be saved.") from exc

    def abort_source() -> None:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        if source_replaced:
            _restore_source_bytes(path, original)

    try:
        with db(before_commit=commit_source, on_abort=abort_source) as conn:
            _replace_knowledge_catalog(catalog, conn)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def _knowledge_sample_text() -> str:
    sample_path = KNOWLEDGE_DIR / KNOWLEDGE_SAMPLE_FILENAME
    if sample_path.exists():
        try:
            return sample_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(503, "The knowledge sample could not be read.") from exc
    # Keep the endpoint useful for API-only deployments even when the optional
    # example file was not mounted alongside the active source.
    return _knowledge_catalog_text(KnowledgeCatalog(
        classifications=[
            KnowledgeClassification(id="access.public", aliases=["public", "common knowledge"]),
            KnowledgeClassification(id="place.example_town", aliases=["example town"]),
            KnowledgeClassification(id="community.example_staff", aliases=["example staff"]),
            KnowledgeClassification(id="domain.example_subject", aliases=["example subject"]),
        ],
        records=[
            GeneralKnowledgeRecord(
                id="example_town.public_fact",
                labels=["place.example_town", "domain.example_subject"],
                access={"require_all": ["access.public", "place.example_town"]},
                assertions=["Replace this assertion with a setting fact supported by an in-world source."],
                source="Example source or authority",
            )
        ],
    ))


def _knowledge_nodes() -> dict[str, KnowledgeClassification]:
    with db() as conn:
        rows = conn.execute("SELECT document_json FROM knowledge_classifications ORDER BY id").fetchall()
    return {
        node.id: node
        for row in rows
        for node in [KnowledgeClassification.model_validate_json(row["document_json"])]
    }


def _label_slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _character_knowledge_labels(character: CharacterDocument, nodes: dict[str, KnowledgeClassification]) -> set[str]:
    """Derive only declared/background labels, then close them through parents."""

    identity = character.identity
    direct = {"access.public", *character.knowledge_labels}
    birthplace = _label_slug(identity.get("birthplace"))
    faction = _label_slug(identity.get("faction"))
    occupation = _label_slug(identity.get("occupation"))
    if birthplace:
        direct.add(f"place.{birthplace}")
    if faction:
        direct.add(f"organization.{faction}")
    if occupation:
        direct.update({f"role.{occupation}", f"occupation.{occupation}"})
    allowed = {label for label in direct if label in nodes}
    pending = list(allowed)
    while pending:
        label = pending.pop()
        for parent in nodes[label].parents:
            if parent not in allowed:
                allowed.add(parent)
                pending.append(parent)
    return allowed


def _query_knowledge_labels(query: str, nodes: dict[str, KnowledgeClassification]) -> set[str]:
    """Resolve only taxonomy aliases; this is not a free-text knowledge search."""

    lower = query.lower()
    matched: set[str] = set()
    for node in nodes.values():
        aliases = {*node.aliases, node.id.replace(".", " ").replace("_", " ")}
        for alias in aliases:
            normalized = alias.strip().lower()
            if normalized and re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", lower):
                matched.add(node.id)
                break
    return matched


def _knowledge_descendants(labels: set[str], nodes: dict[str, KnowledgeClassification]) -> set[str]:
    expanded = set(labels)
    changed = True
    while changed:
        changed = False
        for node in nodes.values():
            if node.id not in expanded and any(parent in expanded for parent in node.parents):
                expanded.add(node.id)
                changed = True
    return expanded


def _knowledge_access_reason(record: GeneralKnowledgeRecord, character_labels: set[str]) -> str | None:
    all_required = set(record.access.require_all)
    any_required = set(record.access.require_any)
    if not all_required.issubset(character_labels):
        return None
    if any_required and not (any_required & character_labels):
        return None
    satisfied = sorted(all_required | (any_required & character_labels))
    return "public" if not satisfied else "derived:" + ",".join(satisfied)


@app.get("/knowledge/classifications")
def knowledge_classifications() -> list[dict[str, Any]]:
    """Expose the controlled taxonomy, not raw event/memory content."""

    return [node.model_dump(mode="json") for node in _knowledge_nodes().values()]


@app.get("/knowledge/catalog")
def knowledge_catalog() -> dict[str, Any]:
    """Return the complete editable catalog and its authoritative source mode."""

    catalog = current_knowledge_catalog()
    source_paths = _knowledge_source_paths()
    managed = len(source_paths) == 1 and source_paths[0].name == MANAGED_KNOWLEDGE_FILENAME
    return {
        "catalog": catalog.model_dump(mode="json"),
        "summary": _knowledge_catalog_summary(catalog),
        "source_files": [path.name for path in source_paths],
        "managed": managed,
    }


@app.get("/knowledge/export")
def export_knowledge_catalog() -> Response:
    catalog = current_knowledge_catalog()
    return Response(
        content=_knowledge_catalog_text(catalog),
        media_type="application/yaml",
        headers={"Content-Disposition": 'attachment; filename="knowledge-catalog.yaml"'},
    )


@app.get("/knowledge/schema-sample")
def export_knowledge_schema_sample() -> Response:
    return Response(
        content=_knowledge_sample_text(),
        media_type="application/yaml",
        headers={"Content-Disposition": 'attachment; filename="catalog.example.yaml"'},
    )


@app.post("/knowledge/validate")
def validate_knowledge_catalog(request: KnowledgeCatalogRequest) -> dict[str, Any]:
    catalog = _catalog_from_uploaded_yaml(request.yaml)
    return {"valid": True, "summary": _knowledge_catalog_summary(catalog)}


@app.put("/knowledge/catalog")
@app.post("/knowledge/import")
def import_knowledge_catalog(request: KnowledgeCatalogRequest) -> dict[str, Any]:
    """Validate and atomically activate a complete managed knowledge catalog."""

    catalog = _catalog_from_uploaded_yaml(request.yaml)
    path = _write_managed_knowledge_catalog(catalog)
    return {
        "catalog": catalog.model_dump(mode="json"),
        "summary": _knowledge_catalog_summary(catalog),
        "source_file": path.name,
        "managed": True,
    }


@app.get("/knowledge/for-character/{character_id}")
def character_knowledge(
    character_id: str,
    query: str = Query(..., min_length=1, max_length=4_000),
    limit: int = Query(12, ge=1, le=24),
) -> dict[str, Any]:
    """Return the bounded, label-authorized general knowledge for one character."""

    with db() as conn:
        row = conn.execute("SELECT document_json FROM characters WHERE id=?", (character_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Character not found")
        character = CharacterDocument.model_validate_json(row["document_json"])
    nodes = _knowledge_nodes()
    character_labels = _character_knowledge_labels(character, nodes)
    query_labels = _query_knowledge_labels(query, nodes)
    indexed_labels = _knowledge_descendants(query_labels, nodes)
    if not indexed_labels:
        return {"query_labels": [], "character_labels": sorted(character_labels), "items": []}
    placeholders = ",".join("?" for _ in indexed_labels)
    with db() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT r.document_json FROM knowledge_records r "
            f"JOIN knowledge_record_labels l ON l.record_id=r.id WHERE l.label_id IN ({placeholders})",
            sorted(indexed_labels),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        record = GeneralKnowledgeRecord.model_validate_json(row["document_json"])
        access_reason = _knowledge_access_reason(record, character_labels)
        if access_reason is None:
            continue
        matched_labels = sorted(set(record.labels) & indexed_labels)
        score = sum(2 if label in query_labels else 1 for label in matched_labels)
        items.append(
            {
                "id": record.id,
                "assertions": record.assertions,
                "labels": record.labels,
                "matched_labels": matched_labels,
                "access_reason": access_reason,
                "epistemic_type": record.epistemic_type,
                "confidence": record.confidence,
                "source": record.source,
                "_score": score,
            }
        )
    items.sort(key=lambda item: (-int(item["_score"]), item["id"]))
    for item in items:
        item.pop("_score", None)
    return {
        "query_labels": sorted(query_labels),
        "character_labels": sorted(character_labels),
        "items": items[:limit],
    }


def _safe_load_uploaded_yaml(source_text: str) -> dict[str, Any]:
    """Reject parser-amplification inputs before loading a user-supplied YAML file."""

    if len(source_text.encode("utf-8")) > 5_000_000:
        raise HTTPException(413, "Uploaded YAML exceeds the 5 MB import limit.")
    try:
        alias_count = 0
        token_count = 0
        for token in yaml.scan(source_text):
            token_count += 1
            alias_count += isinstance(token, AliasToken)
            if alias_count > 64 or token_count > 250_000:
                raise HTTPException(422, "Uploaded YAML is too structurally complex to import safely.")
        loaded = yaml.safe_load(source_text)
    except HTTPException:
        raise
    except yaml.YAMLError as exc:
        raise HTTPException(422, "Uploaded YAML could not be parsed.") from exc
    if not isinstance(loaded, dict):
        raise HTTPException(422, "Uploaded YAML must contain a mapping at its top level.")
    return loaded


def _snapshot_import_payload(source_text: str) -> tuple[CharacterDocument, dict[str, Any] | None]:
    """Accept either a source YAML primer or a complete exported snapshot."""

    loaded = _safe_load_uploaded_yaml(source_text)

    if loaded.get("format") != SNAPSHOT_FORMAT:
        try:
            return CharacterDocument.model_validate(loaded), None
        except ValueError as exc:
            raise HTTPException(422, "Uploaded YAML is not a valid character source document.") from exc

    try:
        source = CharacterDocument.model_validate(loaded.get("source"))
    except ValueError as exc:
        raise HTTPException(422, "Snapshot source is not a valid character document.") from exc
    try:
        runtime_model = SnapshotRuntime.model_validate(loaded.get("runtime"))
    except (TypeError, ValueError, ValidationError) as exc:
        raise HTTPException(422, f"Snapshot runtime is invalid: {exc}") from exc
    for label, records in {
        "goals": runtime_model.state.goals,
        "memories": runtime_model.memories,
        "sessions": runtime_model.sessions,
        "events": runtime_model.events,
        "event_links": runtime_model.event_links,
        "mutation_audit": runtime_model.mutation_audit,
        "belief_history": runtime_model.belief_history,
    }.items():
        if any(record.character_id != source.id for record in records):
            raise HTTPException(422, f"Snapshot {label} contains records for another character.")
    return source, runtime_model.model_dump(mode="json")


def _restore_snapshot(source: CharacterDocument, runtime: dict[str, Any]) -> None:
    """Atomically replace a character's runtime with an exported snapshot."""

    character_id = source.id
    profile_path(character_id)
    try:
        normalized_runtime = SnapshotRuntime.model_validate(runtime).model_dump(mode="json")
    except (TypeError, ValueError, ValidationError) as exc:
        raise HTTPException(422, f"Snapshot runtime is invalid: {exc}") from exc
    state = normalized_runtime["state"]
    mutable_state = state["mutable_state"]
    beliefs = state["beliefs"]
    character_meta = normalized_runtime["character"]
    goals = state["goals"]
    memories = normalized_runtime["memories"]
    sessions = normalized_runtime["sessions"]
    events = normalized_runtime["events"]
    links = normalized_runtime["event_links"]
    mutations = normalized_runtime["mutation_audit"]
    belief_history = normalized_runtime["belief_history"]

    # Stage source first, then replace it immediately before the database commit.
    # If either side fails, the database rolls back and the prior YAML is restored.
    path, staged_source = _stage_character_source(source)
    original_source = path.read_bytes() if path.exists() else None
    source_replaced = False

    def replace_source_before_commit() -> None:
        nonlocal source_replaced
        staged_source.replace(path)
        source_replaced = True

    def compensate_source() -> None:
        if source_replaced:
            _restore_source_bytes(path, original_source)
        elif staged_source.exists():
            staged_source.unlink()

    ts = now_iso()
    with db(before_commit=replace_source_before_commit, on_abort=compensate_source) as conn:
        for table in (
            "event_links",
            "mutation_audit",
            "belief_history",
            "memories",
            "events",
            "sessions",
            "goals",
            "beliefs",
            "mutable_state",
        ):
            conn.execute(f"DELETE FROM {table} WHERE character_id=?", (character_id,))
        conn.execute("DELETE FROM characters WHERE id=?", (character_id,))
        conn.execute(
            "INSERT INTO characters(id, document_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (
                character_id,
                source.model_dump_json(),
                str(character_meta.get("created_at") or ts),
                str(character_meta.get("updated_at") or ts),
            ),
        )
        for key, value in mutable_state.items():
            conn.execute(
                "INSERT INTO mutable_state(character_id, key, value_json, updated_at) VALUES (?, ?, ?, ?)",
                (character_id, str(key), json.dumps(value), ts),
            )
        for key, belief in beliefs.items():
            data = belief
            conn.execute(
                """
                INSERT INTO beliefs(character_id, key, value_json, confidence, epistemic_type, evidence_json, revision, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character_id,
                    str(key),
                    json.dumps(data.get("value")),
                    float(data.get("confidence", 1.0)),
                    str(data.get("epistemic_type", EpistemicType.BELIEF.value)),
                    json.dumps(data.get("evidence", [])),
                    int(data.get("revision", 1)),
                    str(data.get("updated_at") or ts),
                ),
            )
        for row in goals:
            conn.execute(
                "INSERT INTO goals VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(row.get("id") or f"goal_{uuid.uuid4().hex}"), character_id, str(row.get("goal", "")),
                    str(row.get("status", "active")), json.dumps(row.get("metadata", {})),
                    str(row.get("created_at") or ts), str(row.get("updated_at") or ts),
                ),
            )
        for row in sessions:
            conn.execute(
                "INSERT INTO sessions(id, character_id, status, created_at, closed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    str(row.get("id") or f"sess_{uuid.uuid4().hex}"), character_id, str(row.get("status", "open")),
                    str(row.get("created_at") or ts), row.get("closed_at"),
                ),
            )
        for row in events:
            conn.execute(
                """
                INSERT INTO events(id, character_id, session_id, event_type, actor, content, topic, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("id") or f"evt_{uuid.uuid4().hex}"), character_id, row.get("session_id"),
                    str(row.get("event_type", "imported_event")), row.get("actor"), str(row.get("content", "")),
                    row.get("topic"), json.dumps(row.get("metadata", {})), str(row.get("created_at") or ts),
                ),
            )
        for row in memories:
            conn.execute(
                """
                INSERT INTO memories(
                    id, character_id, kind, topic, content, epistemic_type, confidence, salience,
                    source_event_ids_json, status, superseded_by, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("id") or f"mem_{uuid.uuid4().hex}"), character_id, str(row.get("kind", "self_history")),
                    row.get("topic"), str(row.get("content", "")), str(row.get("epistemic_type", "observation")),
                    float(row.get("confidence", 1.0)), float(row.get("salience", 0.5)),
                    json.dumps(row.get("source_event_ids", [])), str(row.get("status", "active")),
                    row.get("superseded_by"), json.dumps(row.get("metadata", {})),
                    str(row.get("created_at") or ts), str(row.get("updated_at") or ts),
                ),
            )
        for row in links:
            conn.execute(
                "INSERT INTO event_links VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(row.get("id") or f"lnk_{uuid.uuid4().hex}"), character_id,
                    str(row.get("from_event_id", "")), str(row.get("to_event_id", "")),
                    str(row.get("relationship", "related_to")), json.dumps(row.get("evidence", [])),
                    str(row.get("created_at") or ts),
                ),
            )
        for row in mutations:
            conn.execute(
                "INSERT INTO mutation_audit VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(row.get("id") or f"mut_{uuid.uuid4().hex}"), character_id,
                    json.dumps(row.get("proposal", {})), str(row.get("status", "allowed")),
                    str(row.get("reason", "Imported snapshot record.")), str(row.get("created_at") or ts),
                ),
            )
        for row in belief_history:
            conn.execute(
                "INSERT INTO belief_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(row.get("id") or f"bh_{uuid.uuid4().hex}"), character_id, str(row.get("key", "")),
                    json.dumps(row["old_value"]) if row.get("old_value") is not None else None,
                    json.dumps(row.get("new_value")), float(row.get("confidence", 1.0)),
                    json.dumps(row.get("evidence", [])), str(row.get("reason", "Imported snapshot record.")),
                    str(row.get("created_at") or ts),
                ),
            )


def _diff_display_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 400:
        return value[:397] + "..."
    if isinstance(value, list) and len(value) > 12:
        return {"items": len(value), "preview": [_diff_display_value(item) for item in value[:12]]}
    if isinstance(value, dict) and len(value) > 20:
        keys = sorted(value)[:20]
        return {key: _diff_display_value(value[key]) for key in keys} | {"_truncated_keys": len(value) - len(keys)}
    return value


def _snapshot_diff(before: Any, after: Any, path: str = "", changes: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Produce a bounded, ID-aware structural diff suitable for profile review."""

    changes = changes if changes is not None else []
    if len(changes) >= 250:
        return changes
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}" if path else key
            if key not in before:
                changes.append({"kind": "added", "path": child_path, "after": _diff_display_value(after[key])})
            elif key not in after:
                changes.append({"kind": "removed", "path": child_path, "before": _diff_display_value(before[key])})
            else:
                _snapshot_diff(before[key], after[key], child_path, changes)
            if len(changes) >= 250:
                break
        return changes
    if isinstance(before, list) and isinstance(after, list):
        if all(isinstance(item, dict) and "id" in item for item in before + after):
            before_by_id = {str(item["id"]): item for item in before}
            after_by_id = {str(item["id"]): item for item in after}
            return _snapshot_diff(before_by_id, after_by_id, path, changes)
        if before != after:
            changes.append({
                "kind": "changed",
                "path": path,
                "before": _diff_display_value(before),
                "after": _diff_display_value(after),
            })
        return changes
    if before != after:
        changes.append({
            "kind": "changed",
            "path": path,
            "before": _diff_display_value(before),
            "after": _diff_display_value(after),
        })
    return changes


@app.get("/profiles/{character_id}/export")
def export_profile(character_id: str) -> Response:
    """Download source plus durable runtime conclusions as a portable YAML file."""

    profile_path(character_id)
    payload = character_snapshot(character_id)
    source = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    return Response(
        content=source,
        media_type="application/yaml; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{character_id}.snapshot.yaml"'},
    )


@app.post("/profiles/{character_id}/diff")
def diff_profile(character_id: str, request: ProfileDiffRequest) -> dict[str, Any]:
    """Compare an exported snapshot with the currently stored source and runtime."""

    profile_path(character_id)
    source, runtime = _snapshot_import_payload(request.yaml)
    if runtime is None:
        raise HTTPException(422, "Comparison requires a full exported character snapshot.")
    if source.id != character_id:
        raise HTTPException(422, "The comparison snapshot belongs to another character.")
    previous = {"format": SNAPSHOT_FORMAT, "source": source.model_dump(mode="json", exclude_none=True), "runtime": runtime}
    current = character_snapshot(character_id)
    changes = _snapshot_diff(previous, current)
    return {
        "character_id": character_id,
        "changed": bool(changes),
        "truncated": len(changes) >= 250,
        "changes": changes,
        "summary": {"change_count": len(changes)},
    }


@app.post("/profiles/import")
def import_profile(request: ProfileImportRequest) -> dict[str, Any]:
    """Import a primer or replace a profile with a full exported snapshot."""

    source, runtime = _snapshot_import_payload(request.yaml)
    path = profile_path(source.id)
    if runtime is not None:
        _restore_snapshot(source, runtime)
        return get_profile(source.id)

    with db() as conn:
        exists = conn.execute("SELECT 1 FROM characters WHERE id=?", (source.id,)).fetchone()
    if exists or path.exists():
        return update_profile(source.id, source)
    return create_profile(source)


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
def session_events(
    session_id: str,
    limit: int = Query(MAX_SESSION_EVENTS_RETURNED, ge=1, le=MAX_SESSION_EVENTS_RETURNED),
) -> list[dict[str, Any]]:
    with db() as conn:
        session = conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        rows = list(reversed(conn.execute(
            "SELECT * FROM events WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()))
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


def _reflection_job_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "character_id": row["character_id"],
        "status": row["status"],
        "attempts": int(row["attempts"]),
        "last_error": row["last_error"],
        "next_attempt_at": row["next_attempt_at"],
        "lease_expires_at": row["lease_expires_at"],
        "completed_at": row["completed_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.post("/reflection-jobs/{session_id}/schedule")
def schedule_reflection_retry(session_id: str, request: ReflectionRetrySchedule) -> dict[str, Any]:
    """Persist a close-time reflection failure for asynchronous retry."""

    ts = now_iso()
    with db() as conn:
        session = conn.execute(
            "SELECT character_id FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        existing = conn.execute(
            "SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)
        ).fetchone()
        if existing and existing["status"] == "completed":
            return {**_reflection_job_payload(existing), "already_completed": True}
        if existing:
            conn.execute(
                """
                UPDATE reflection_jobs
                SET status='pending', last_error=?, next_attempt_at=?, lease_expires_at=NULL, updated_at=?
                WHERE session_id=?
                """,
                (request.error, ts, ts, session_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO reflection_jobs(
                    session_id, character_id, status, attempts, last_error, next_attempt_at,
                    lease_expires_at, completed_at, created_at, updated_at
                ) VALUES (?, ?, 'pending', 0, ?, ?, NULL, NULL, ?, ?)
                """,
                (session_id, session["character_id"], request.error, ts, ts, ts),
            )
        row = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
    return _reflection_job_payload(row)


@app.post("/reflection-jobs/claim")
def claim_reflection_retry(request: ReflectionRetryClaim) -> dict[str, Any]:
    """Atomically lease one due reflection job to a single orchestrator."""

    ts = now_iso()
    lease_until = (datetime.now(timezone.utc) + timedelta(seconds=request.lease_seconds)).isoformat()
    with db() as conn:
        row = conn.execute(
            """
            SELECT * FROM reflection_jobs
            WHERE (status='pending' AND next_attempt_at <= ?)
               OR (status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
            ORDER BY next_attempt_at, created_at
            LIMIT 1
            """,
            (ts, ts),
        ).fetchone()
        if not row:
            return {"job": None}
        conn.execute(
            """
            UPDATE reflection_jobs
            SET status='running', attempts=attempts + 1, lease_expires_at=?, updated_at=?
            WHERE session_id=?
            """,
            (lease_until, ts, row["session_id"]),
        )
        claimed = conn.execute(
            "SELECT * FROM reflection_jobs WHERE session_id=?", (row["session_id"],)
        ).fetchone()
    return {"job": _reflection_job_payload(claimed)}


@app.post("/reflection-jobs/{session_id}/complete")
def complete_reflection_retry(session_id: str) -> dict[str, Any]:
    """Mark a queued reflection complete without deleting its operational audit."""

    ts = now_iso()
    with db() as conn:
        row = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return {"found": False, "session_id": session_id}
        conn.execute(
            """
            UPDATE reflection_jobs
            SET status='completed', last_error=NULL, next_attempt_at=?, lease_expires_at=NULL,
                completed_at=?, updated_at=?
            WHERE session_id=?
            """,
            (ts, ts, ts, session_id),
        )
        completed = conn.execute(
            "SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)
        ).fetchone()
    return {"found": True, **_reflection_job_payload(completed)}


@app.post("/reflection-jobs/{session_id}/reschedule")
def reschedule_reflection_retry(session_id: str, request: ReflectionRetrySchedule) -> dict[str, Any]:
    """Release a failed lease with bounded exponential backoff."""

    ts = now_iso()
    with db() as conn:
        row = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Reflection retry job not found")
        attempts = max(int(row["attempts"]), 1)
        delay_seconds = min(30 * (2 ** min(attempts - 1, 7)), 3_600)
        next_attempt = (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()
        conn.execute(
            """
            UPDATE reflection_jobs
            SET status='pending', last_error=?, next_attempt_at=?, lease_expires_at=NULL, updated_at=?
            WHERE session_id=?
            """,
            (request.error, next_attempt, ts, session_id),
        )
        scheduled = conn.execute(
            "SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)
        ).fetchone()
    return {"backoff_seconds": delay_seconds, **_reflection_job_payload(scheduled)}


@app.get("/reflection-jobs")
def reflection_jobs(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    """Expose retry state for the authenticated runtime-status dashboard."""

    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM reflection_jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    jobs = [_reflection_job_payload(row) for row in rows]
    return {
        "jobs": jobs,
        "pending_count": sum(job["status"] in {"pending", "running"} for job in jobs),
    }


def _add_event(event: EventRecord, conn: sqlite3.Connection) -> EventRecord:
    """Validate and insert an event using an existing transaction."""

    eid = event.id or f"evt_{uuid.uuid4().hex}"
    ts = now_iso()
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
        (
            eid,
            event.character_id,
            event.session_id,
            event.event_type,
            event.actor,
            event.content,
            event.topic,
            metadata_json,
            ts,
        ),
    )
    return event.model_copy(update={"id": eid})


@app.post("/events", response_model=EventRecord)
def add_event(event: EventRecord) -> EventRecord:
    with db() as conn:
        return _add_event(event, conn)


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


def _add_memory(memory: MemoryRecord, conn: sqlite3.Connection) -> MemoryRecord:
    """Validate and insert a memory using an existing transaction."""

    mid = memory.id or f"mem_{uuid.uuid4().hex}"
    ts = now_iso()
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
            json.dumps(source_ids), memory.status, memory.superseded_by,
            metadata_json, ts, ts,
        ),
    )
    return memory.model_copy(update={"id": mid})


@app.post("/memories", response_model=MemoryRecord)
def add_memory(memory: MemoryRecord) -> MemoryRecord:
    with db() as conn:
        return _add_memory(memory, conn)


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


def validate_proposal(
    proposal: MutationProposal,
    *,
    allowed_mutable_keys: set[str] | None = None,
    evidence_event_ids: set[str] | None = None,
    goal_ids: set[str] | None = None,
    memory_ids: set[str] | None = None,
) -> ValidatedMutation:
    """Apply a narrow state-transition policy to untrusted executive proposals."""

    try:
        json.dumps(proposal.value, allow_nan=False)
    except (TypeError, ValueError):
        return ValidatedMutation(proposal=proposal, status="rejected", reason="Mutation value is not JSON-safe.")
    evidence = {event_id for event_id in proposal.evidence if event_id}
    if proposal.evidence and len(evidence) != len(proposal.evidence):
        return ValidatedMutation(proposal=proposal, status="rejected", reason="Mutation evidence contains blank or duplicate event IDs.")
    if evidence_event_ids is not None and not evidence.issubset(evidence_event_ids):
        return ValidatedMutation(proposal=proposal, status="rejected", reason="Mutation evidence must reference this character's recorded events.")
    if proposal.operation == MutationOperation.UPDATE_CORE:
        return ValidatedMutation(proposal=proposal, status="rejected", reason="Core biography/identity is immutable at runtime.")
    if proposal.operation == MutationOperation.SET_MUTABLE_STATE:
        if not evidence:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="State revisions require provenance/evidence.")
        if allowed_mutable_keys is not None and proposal.target not in allowed_mutable_keys:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Mutable state target is not declared by this character.")
        return ValidatedMutation(proposal=proposal, status="versioned", reason="Declared mutable state may change with provenance.")
    if proposal.operation == MutationOperation.SET_BELIEF:
        if not evidence:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Belief revisions require provenance/evidence.")
        if proposal.epistemic_type in {EpistemicType.FACT, EpistemicType.OBSERVATION, EpistemicType.SELF_STATEMENT}:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="A model may not promote conversation evidence to a fact or observation.")
        return ValidatedMutation(proposal=proposal, status="versioned", reason="Mutable state may change, but revision history is preserved.")
    if proposal.operation == MutationOperation.ADD_MEMORY:
        if not evidence:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Derived memories require source evidence.")
        return ValidatedMutation(proposal=proposal, status="allowed", reason="Append-only derived memory is permitted.")
    if proposal.operation == MutationOperation.ADD_GOAL:
        if not evidence:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="New goals require source evidence.")
        return ValidatedMutation(proposal=proposal, status="allowed", reason="A sourced goal may be added.")
    if proposal.operation == MutationOperation.UPDATE_GOAL:
        if not evidence:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Goal revisions require source evidence.")
        if goal_ids is not None and proposal.target not in goal_ids:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Goal revision targets an unknown goal.")
        return ValidatedMutation(proposal=proposal, status="versioned", reason="A sourced goal revision is permitted.")
    if proposal.operation == MutationOperation.SUPERSEDE_MEMORY:
        if not evidence:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Memory supersession requires source evidence.")
        if memory_ids is not None and proposal.target not in memory_ids:
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Memory supersession targets an unknown memory.")
        return ValidatedMutation(proposal=proposal, status="versioned", reason="A sourced memory supersession is permitted.")
    if proposal.operation == MutationOperation.LINK_EVENTS:
        data = proposal.value if isinstance(proposal.value, dict) else {}
        from_id, to_id = str(data.get("from") or ""), str(data.get("to") or "")
        if not from_id or not to_id or (evidence_event_ids is not None and ({from_id, to_id} - evidence_event_ids)):
            return ValidatedMutation(proposal=proposal, status="rejected", reason="Event links must reference recorded events for this character.")
        return ValidatedMutation(proposal=proposal, status="allowed", reason="Recorded events may be linked.")
    return ValidatedMutation(proposal=proposal, status="allowed", reason="Operation is permitted by runtime policy.")


class MutationBatch(BaseModel):
    proposals: list[MutationProposal] = Field(default_factory=list)


class TurnCommit(BaseModel):
    """All durable outputs of a successful cognitive turn, committed together."""

    model_config = {"extra": "forbid"}

    character_event: EventRecord
    memories: list[MemoryRecord] = Field(default_factory=list, max_length=20)
    proposals: list[MutationProposal] = Field(default_factory=list, max_length=25)


def _apply_mutations(
    character_id: str,
    batch: MutationBatch,
    conn: sqlite3.Connection,
) -> list[ValidatedMutation]:
    """Apply already-validated proposals inside a caller-owned transaction."""

    results: list[ValidatedMutation] = []
    ts = now_iso()
    character_row = conn.execute("SELECT document_json FROM characters WHERE id=?", (character_id,)).fetchone()
    if not character_row:
        raise HTTPException(404, "Character not found")
    character = CharacterDocument.model_validate_json(character_row["document_json"])
    allowed_mutable_keys = set(character.mutable_state) | {"topic_defensiveness"}
    evidence_event_ids = {
        str(row["id"])
        for row in conn.execute("SELECT id FROM events WHERE character_id=?", (character_id,))
    }
    goal_ids = {
        str(row["id"])
        for row in conn.execute("SELECT id FROM goals WHERE character_id=?", (character_id,))
    }
    memory_ids = {
        str(row["id"])
        for row in conn.execute("SELECT id FROM memories WHERE character_id=?", (character_id,))
    }
    for proposal in batch.proposals:
        checked = validate_proposal(
            proposal,
            allowed_mutable_keys=allowed_mutable_keys,
            evidence_event_ids=evidence_event_ids,
            goal_ids=goal_ids,
            memory_ids=memory_ids,
        )
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
                """INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, '{}', ?, ?)""",
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
                    """INSERT OR IGNORE INTO event_links VALUES (?, ?, ?, ?, ?, ?, ?)""",
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


@app.post("/mutations/{character_id}")
def apply_mutations(character_id: str, batch: MutationBatch) -> list[ValidatedMutation]:
    with db() as conn:
        return _apply_mutations(character_id, batch, conn)


@app.post("/sessions/{session_id}/turn")
def commit_turn(session_id: str, turn: TurnCommit) -> dict[str, Any]:
    """Commit the answer, its memories, and allowed state changes atomically."""

    event = turn.character_event
    if event.id is None or event.event_type != "character_message" or event.session_id != session_id:
        raise HTTPException(422, "A turn commit requires an identified character message for this session.")
    if any(memory.character_id != event.character_id for memory in turn.memories):
        raise HTTPException(422, "Turn memories must belong to the replying character.")
    with db() as conn:
        session = conn.execute("SELECT character_id, status FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        if session["status"] != "open" or session["character_id"] != event.character_id:
            raise HTTPException(409, "The turn does not belong to an open session for this character.")
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        responds_to = str(metadata.get("responds_to") or "")
        user_event = conn.execute(
            "SELECT 1 FROM events WHERE id=? AND character_id=? AND session_id=? AND event_type='user_message'",
            (responds_to, event.character_id, session_id),
        ).fetchone()
        if not user_event:
            raise HTTPException(422, "The character message must answer a recorded user event in this session.")
        stored_event = _add_event(event, conn)
        claim_audit = metadata.get("claim_verification", [])
        if not isinstance(claim_audit, list):
            raise HTTPException(422, "Character event claim verification must be a list.")
        for claim in claim_audit:
            if not isinstance(claim, dict) or claim.get("status") != "verified":
                raise HTTPException(422, "Only verified Executive claims may be committed.")
            conn.execute(
                "INSERT INTO executive_claim_audit VALUES (?, ?, ?, ?, 'verified', ?)",
                (f"claim_{uuid.uuid4().hex}", stored_event.id, event.character_id, json.dumps(claim), now_iso()),
            )
        stored_memories = [_add_memory(memory, conn) for memory in turn.memories]
        mutation_results = _apply_mutations(event.character_id, MutationBatch(proposals=turn.proposals), conn)
    return {
        "character_event": stored_event.model_dump(mode="json"),
        "memories": [memory.model_dump(mode="json") for memory in stored_memories],
        "mutation_results": [result.model_dump(mode="json") for result in mutation_results],
    }


@app.get("/debug/{character_id}")
def debug_state(character_id: str) -> dict[str, Any]:
    with db() as conn:
        events = [dict(r) for r in conn.execute("SELECT * FROM events WHERE character_id=? ORDER BY created_at, rowid", (character_id,))]
        memories = [dict(r) for r in conn.execute("SELECT * FROM memories WHERE character_id=? ORDER BY created_at, rowid", (character_id,))]
        mutations = [dict(r) for r in conn.execute("SELECT * FROM mutation_audit WHERE character_id=? ORDER BY created_at, rowid", (character_id,))]
        links = [dict(r) for r in conn.execute("SELECT * FROM event_links WHERE character_id=? ORDER BY created_at, rowid", (character_id,))]
    return {"events": events, "memories": memories, "mutations": mutations, "links": links}
