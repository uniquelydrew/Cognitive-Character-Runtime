"""Source-backed character profiles and portable runtime snapshots.

The memory HTTP module supplies the database context factory and current paths;
this module owns the profile/source consistency rules and has no FastAPI routes.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Pattern

import yaml
from fastapi import HTTPException

from services.common import CharacterDocument, EpistemicType
from services.memory.storage import now_iso


class ProfileStore:
    """Keep canonical YAML primers and their runtime projections consistent."""

    def __init__(
        self,
        *,
        db: Callable[..., Any],
        character_dir: Path,
        profile_id_re: Pattern[str],
        snapshot_format: str,
    ) -> None:
        self.db = db
        self.character_dir = character_dir
        self.profile_id_re = profile_id_re
        self.snapshot_format = snapshot_format

    def load_character_files(self) -> None:
        if not self.character_dir.exists():
            return
        for path in sorted(self.character_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "biography_file" in raw:
                bio_path = self.safe_biography_path(path, raw.pop("biography_file"))
                if bio_path.exists():
                    raw["biography"] = bio_path.read_text(encoding="utf-8").strip()
            self.upsert_character(CharacterDocument.model_validate(raw), initialize=True)

    def profile_path(self, character_id: str) -> Path:
        if not self.profile_id_re.fullmatch(character_id):
            raise HTTPException(
                422,
                "Character IDs must start with a lowercase letter and use only lowercase letters, numbers, _ or -.",
            )
        return self.character_dir / f"{character_id}.yaml"

    @staticmethod
    def safe_biography_path(profile: Path, raw_path: Any) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise HTTPException(422, "biography_file must be a non-empty relative path.")
        root = profile.parent.resolve()
        candidate = (root / raw_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise HTTPException(422, "biography_file must stay inside the character source directory.")
        return candidate

    @staticmethod
    def character_source_text(character: CharacterDocument) -> str:
        return yaml.safe_dump(
            character.model_dump(mode="json", exclude_none=True),
            allow_unicode=True,
            sort_keys=False,
        )

    def stage_character_source(self, character: CharacterDocument) -> tuple[Path, Path]:
        path = self.profile_path(character.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".yaml.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(self.character_source_text(character), encoding="utf-8")
        except OSError as exc:
            raise HTTPException(503, "The canonical profile source could not be saved.") from exc
        return path, temporary

    @staticmethod
    def restore_source_bytes(path: Path, original: bytes | None) -> None:
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

    def write_character_source(self, character: CharacterDocument) -> Path:
        path, temporary = self.stage_character_source(character)
        try:
            temporary.replace(path)
        except OSError as exc:
            raise HTTPException(503, "The canonical profile source could not be saved.") from exc
        return path

    @staticmethod
    def profile_summary(character: CharacterDocument) -> dict[str, Any]:
        identity = character.identity
        return {
            "id": character.id,
            "name": str(identity.get("name", character.id)),
            "occupation": str(identity.get("occupation", "")),
            "faction": str(identity.get("faction", "")),
            "source_file": f"{character.id}.yaml",
        }

    def read_character_source(self, character_id: str) -> CharacterDocument:
        path = self.profile_path(character_id)
        if not path.exists():
            raise HTTPException(404, "Canonical profile source was not found.")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "biography_file" in raw:
                bio_path = self.safe_biography_path(path, raw.pop("biography_file"))
                if bio_path.exists():
                    raw["biography"] = bio_path.read_text(encoding="utf-8").strip()
            return CharacterDocument.model_validate(raw)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise HTTPException(422, "Canonical profile source is not a valid character document.") from exc

    def upsert_character(
        self,
        character: CharacterDocument,
        initialize: bool = False,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if conn is None:
            with self.db() as owned_connection:
                self.upsert_character(character, initialize=initialize, conn=owned_connection)
            return
        timestamp = now_iso()
        exists = conn.execute("SELECT 1 FROM characters WHERE id=?", (character.id,)).fetchone()
        conn.execute(
            """
            INSERT INTO characters(id, document_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET document_json=excluded.document_json, updated_at=excluded.updated_at
            """,
            (character.id, character.model_dump_json(), timestamp, timestamp),
        )
        if initialize and not exists:
            for key, value in character.mutable_state.items():
                conn.execute(
                    "INSERT OR REPLACE INTO mutable_state VALUES (?, ?, ?, ?)",
                    (character.id, key, json.dumps(value), timestamp),
                )
            for key, value in character.beliefs.items():
                conn.execute(
                    "INSERT OR REPLACE INTO beliefs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (character.id, key, json.dumps(value), 1.0, EpistemicType.BELIEF.value, "[]", 1, timestamp),
                )
            for goal in character.initial_goals:
                conn.execute(
                    "INSERT INTO goals VALUES (?, ?, ?, 'active', '{}', ?, ?)",
                    (f"goal_{uuid.uuid4().hex}", character.id, goal, timestamp, timestamp),
                )

    def persist_profile_and_runtime(
        self,
        profile: CharacterDocument,
        *,
        initialize: bool,
        upsert: Callable[..., None] | None = None,
    ) -> None:
        path, temporary = self.stage_character_source(profile)
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
                self.restore_source_bytes(path, original)

        try:
            with self.db(before_commit=commit_source, on_abort=abort_source) as conn:
                (upsert or self.upsert_character)(profile, initialize=initialize, conn=conn)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def list_characters(self) -> list[CharacterDocument]:
        with self.db() as conn:
            rows = conn.execute("SELECT document_json FROM characters ORDER BY id").fetchall()
        return [CharacterDocument.model_validate_json(row["document_json"]) for row in rows]

    def get_character(self, character_id: str) -> dict[str, Any]:
        with self.db() as conn:
            row = conn.execute("SELECT document_json FROM characters WHERE id=?", (character_id,)).fetchone()
            if not row:
                raise HTTPException(404, "Character not found")
            character = CharacterDocument.model_validate_json(row["document_json"])
            mutable = {
                row["key"]: json.loads(row["value_json"])
                for row in conn.execute("SELECT key, value_json FROM mutable_state WHERE character_id=?", (character_id,))
            }
            beliefs = {
                row["key"]: {
                    "value": json.loads(row["value_json"]),
                    "confidence": row["confidence"],
                    "epistemic_type": row["epistemic_type"],
                    "evidence": json.loads(row["evidence_json"]),
                    "revision": row["revision"],
                }
                for row in conn.execute("SELECT * FROM beliefs WHERE character_id=?", (character_id,))
            }
            goals = [dict(row) for row in conn.execute("SELECT * FROM goals WHERE character_id=? ORDER BY created_at", (character_id,))]
        return {"character": character.model_dump(), "mutable_state": mutable, "beliefs": beliefs, "goals": goals}

    def snapshot_runtime(self, character_id: str) -> dict[str, Any]:
        state = self.get_character(character_id)
        with self.db() as conn:
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
                    {**{key: row[key] for key in row.keys() if key != "metadata_json"}, "metadata": json.loads(row["metadata_json"])}
                    for row in goal_rows
                ],
            },
            "memories": [
                {
                    **{key: row[key] for key in row.keys() if key not in {"source_event_ids_json", "metadata_json"}},
                    "source_event_ids": json.loads(row["source_event_ids_json"]),
                    "metadata": json.loads(row["metadata_json"]),
                }
                for row in memory_rows
            ],
            "sessions": sessions,
            "events": [
                {**{key: row[key] for key in row.keys() if key != "metadata_json"}, "metadata": json.loads(row["metadata_json"])}
                for row in event_rows
            ],
            "event_links": [
                {**{key: row[key] for key in row.keys() if key != "evidence_json"}, "evidence": json.loads(row["evidence_json"])}
                for row in link_rows
            ],
            "mutation_audit": [
                {**{key: row[key] for key in row.keys() if key != "proposal_json"}, "proposal": json.loads(row["proposal_json"])}
                for row in mutation_rows
            ],
            "belief_history": [
                {
                    **{key: row[key] for key in row.keys() if key not in {"old_value_json", "new_value_json", "evidence_json"}},
                    "old_value": json.loads(row["old_value_json"]) if row["old_value_json"] is not None else None,
                    "new_value": json.loads(row["new_value_json"]),
                    "evidence": json.loads(row["evidence_json"]),
                }
                for row in belief_history_rows
            ],
        }

    def character_snapshot(self, character_id: str) -> dict[str, Any]:
        source = self.read_character_source(character_id)
        return {
            "format": self.snapshot_format,
            "source": source.model_dump(mode="json", exclude_none=True),
            "runtime": self.snapshot_runtime(character_id),
        }
