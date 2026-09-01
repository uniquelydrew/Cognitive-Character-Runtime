from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError

from services.common import (
    CharacterDocument,
    EventRecord,
    GeneralKnowledgeRecord,
    KnowledgeCatalog,
    KnowledgeClassification,
    MemoryRecord,
    ValidatedMutation,
)
from services.memory.migrations import apply_migrations
from services.memory.models import (
    KnowledgeCatalogRequest,
    ProfileDiffRequest,
    ProfileImportRequest,
    ReflectionRetryClaim,
    ReflectionRetrySchedule,
)
from services.memory.mutations import (
    MutationBatch,
    TurnCommit,
    apply_mutations as apply_mutation_batch,
    validate_proposal,
)
from services.memory.profiles import ProfileStore
from services.memory.records import (
    add_event as store_event,
    add_memory as store_memory,
    get_memories as query_memories,
    interaction_history as query_interaction_history,
)
from services.memory.sessions import SessionCreate, SessionStore
from services.memory.snapshots import SnapshotStore, diff_display_value, snapshot_diff
from services.memory.storage import connection, now_iso

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


def db(*, before_commit: Any | None = None, on_abort: Any | None = None):
    """Compatibility façade while routes migrate to the storage module."""
    return connection(DB_PATH, before_commit=before_commit, on_abort=on_abort)


def _profiles() -> ProfileStore:
    """Build the source-backed profile store from the current testable settings."""

    return ProfileStore(
        db=db,
        character_dir=CHARACTER_DIR,
        profile_id_re=PROFILE_ID_RE,
        snapshot_format=SNAPSHOT_FORMAT,
    )


def _sessions() -> SessionStore:
    return SessionStore(db=db)


def _snapshots() -> SnapshotStore:
    return SnapshotStore(db=db, profiles=_profiles(), snapshot_format=SNAPSHOT_FORMAT)


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
    _profiles().load_character_files()


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
    return _profiles().profile_path(character_id)


def _safe_biography_path(profile: Path, raw_path: Any) -> Path:
    return ProfileStore.safe_biography_path(profile, raw_path)


def _character_source_text(char: CharacterDocument) -> str:
    return ProfileStore.character_source_text(char)


def _stage_character_source(char: CharacterDocument) -> tuple[Path, Path]:
    return _profiles().stage_character_source(char)


def _restore_source_bytes(path: Path, original: bytes | None) -> None:
    ProfileStore.restore_source_bytes(path, original)


def write_character_source(char: CharacterDocument) -> Path:
    return _profiles().write_character_source(char)


def profile_summary(char: CharacterDocument) -> dict[str, Any]:
    return ProfileStore.profile_summary(char)


def read_character_source(character_id: str) -> CharacterDocument:
    return _profiles().read_character_source(character_id)


def upsert_character(
    char: CharacterDocument,
    initialize: bool = False,
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    _profiles().upsert_character(char, initialize=initialize, conn=conn)


def persist_profile_and_runtime(profile: CharacterDocument, *, initialize: bool) -> None:
    # Preserve the injectable app-level seam used to verify file/DB rollback.
    _profiles().persist_profile_and_runtime(
        profile,
        initialize=initialize,
        upsert=upsert_character,
    )



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "database": str(DB_PATH)}


@app.get("/characters", response_model=list[CharacterDocument])
def list_characters() -> list[CharacterDocument]:
    return _profiles().list_characters()


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
    return _profiles().get_character(character_id)


def _snapshot_runtime(character_id: str) -> dict[str, Any]:
    return _profiles().snapshot_runtime(character_id)


def character_snapshot(character_id: str) -> dict[str, Any]:
    return _profiles().character_snapshot(character_id)


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
    return _snapshots().safe_load_uploaded_yaml(source_text)


def _snapshot_import_payload(source_text: str) -> tuple[CharacterDocument, dict[str, Any] | None]:
    return _snapshots().import_payload(source_text)


def _restore_snapshot(source: CharacterDocument, runtime: dict[str, Any]) -> None:
    _snapshots().restore(source, runtime)

def _diff_display_value(value: Any) -> Any:
    return diff_display_value(value)


def _snapshot_diff(
    before: Any,
    after: Any,
    path: str = "",
    changes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return snapshot_diff(before, after, path, changes)


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


@app.post("/sessions")
def create_session(req: SessionCreate) -> dict[str, Any]:
    return _sessions().create(req)


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    return _sessions().get(session_id)


@app.get("/sessions/{session_id}/events")
def session_events(
    session_id: str,
    limit: int = Query(MAX_SESSION_EVENTS_RETURNED, ge=1, le=MAX_SESSION_EVENTS_RETURNED),
) -> list[dict[str, Any]]:
    return _sessions().events(session_id, limit)


@app.post("/sessions/{session_id}/close")
def close_session(session_id: str) -> dict[str, str]:
    return _sessions().close(session_id)


def _reflection_job_payload(row: sqlite3.Row) -> dict[str, Any]:
    return SessionStore.job_payload(row)


@app.post("/reflection-jobs/{session_id}/schedule")
def schedule_reflection_retry(session_id: str, request: ReflectionRetrySchedule) -> dict[str, Any]:
    return _sessions().schedule_reflection_retry(session_id, request)


@app.post("/reflection-jobs/claim")
def claim_reflection_retry(request: ReflectionRetryClaim) -> dict[str, Any]:
    return _sessions().claim_reflection_retry(request)


@app.post("/reflection-jobs/{session_id}/complete")
def complete_reflection_retry(session_id: str) -> dict[str, Any]:
    return _sessions().complete_reflection_retry(session_id)


@app.post("/reflection-jobs/{session_id}/reschedule")
def reschedule_reflection_retry(session_id: str, request: ReflectionRetrySchedule) -> dict[str, Any]:
    return _sessions().reschedule_reflection_retry(session_id, request)


@app.get("/reflection-jobs")
def reflection_jobs(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    return _sessions().reflection_jobs(limit)


def _add_event(event: EventRecord, conn: sqlite3.Connection) -> EventRecord:
    """Compatibility façade for transactional event persistence."""

    return store_event(event, conn)


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
        return query_interaction_history(character_id, topic, limit, conn)


def _add_memory(memory: MemoryRecord, conn: sqlite3.Connection) -> MemoryRecord:
    """Compatibility façade for transactional memory persistence."""

    return store_memory(memory, conn)


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
    with db() as conn:
        return query_memories(character_id, topic, limit, conn)


def _apply_mutations(
    character_id: str,
    batch: MutationBatch,
    conn: sqlite3.Connection,
) -> list[ValidatedMutation]:
    return apply_mutation_batch(character_id, batch, conn)


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
