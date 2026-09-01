"""Portable character snapshot validation, restoration, and comparison."""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

import yaml
from fastapi import HTTPException
from pydantic import ValidationError
from yaml.tokens import AliasToken

from services.common import CharacterDocument, EpistemicType
from services.memory.models import SnapshotRuntime
from services.memory.profiles import ProfileStore
from services.memory.storage import now_iso


class SnapshotStore:
    """Keep snapshot import/export mechanics out of the HTTP application."""

    def __init__(
        self,
        *,
        db: Callable[..., Any],
        profiles: ProfileStore,
        snapshot_format: str,
    ) -> None:
        self.db = db
        self.profiles = profiles
        self.snapshot_format = snapshot_format

    @staticmethod
    def safe_load_uploaded_yaml(source_text: str) -> dict[str, Any]:
        """Reject parser-amplification inputs before loading uploaded YAML."""

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

    def import_payload(self, source_text: str) -> tuple[CharacterDocument, dict[str, Any] | None]:
        """Accept either a source YAML primer or a complete exported snapshot."""

        loaded = self.safe_load_uploaded_yaml(source_text)
        if loaded.get("format") != self.snapshot_format:
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

    def restore(self, source: CharacterDocument, runtime: dict[str, Any]) -> None:
        """Atomically replace one character runtime with its exported snapshot."""

        character_id = source.id
        self.profiles.profile_path(character_id)
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

        # Stage source first, then replace it immediately before the database
        # commit. If either side fails, rollback restores the prior YAML.
        path, staged_source = self.profiles.stage_character_source(source)
        original_source = path.read_bytes() if path.exists() else None
        source_replaced = False

        def replace_source_before_commit() -> None:
            nonlocal source_replaced
            staged_source.replace(path)
            source_replaced = True

        def compensate_source() -> None:
            if source_replaced:
                self.profiles.restore_source_bytes(path, original_source)
            elif staged_source.exists():
                staged_source.unlink()

        timestamp = now_iso()
        with self.db(before_commit=replace_source_before_commit, on_abort=compensate_source) as conn:
            for table in (
                "event_links", "mutation_audit", "belief_history", "memories", "events",
                "sessions", "goals", "beliefs", "mutable_state",
            ):
                conn.execute(f"DELETE FROM {table} WHERE character_id=?", (character_id,))
            conn.execute("DELETE FROM characters WHERE id=?", (character_id,))
            conn.execute(
                "INSERT INTO characters(id, document_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (
                    character_id, source.model_dump_json(),
                    str(character_meta.get("created_at") or timestamp),
                    str(character_meta.get("updated_at") or timestamp),
                ),
            )
            for key, value in mutable_state.items():
                conn.execute(
                    "INSERT INTO mutable_state(character_id, key, value_json, updated_at) VALUES (?, ?, ?, ?)",
                    (character_id, str(key), json.dumps(value), timestamp),
                )
            for key, belief in beliefs.items():
                conn.execute(
                    """
                    INSERT INTO beliefs(character_id, key, value_json, confidence, epistemic_type, evidence_json, revision, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        character_id, str(key), json.dumps(belief.get("value")),
                        float(belief.get("confidence", 1.0)),
                        str(belief.get("epistemic_type", EpistemicType.BELIEF.value)),
                        json.dumps(belief.get("evidence", [])), int(belief.get("revision", 1)),
                        str(belief.get("updated_at") or timestamp),
                    ),
                )
            for row in goals:
                conn.execute(
                    "INSERT INTO goals VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(row.get("id") or f"goal_{uuid.uuid4().hex}"), character_id,
                        str(row.get("goal", "")), str(row.get("status", "active")),
                        json.dumps(row.get("metadata", {})), str(row.get("created_at") or timestamp),
                        str(row.get("updated_at") or timestamp),
                    ),
                )
            for row in sessions:
                conn.execute(
                    "INSERT INTO sessions(id, character_id, status, created_at, closed_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        str(row.get("id") or f"sess_{uuid.uuid4().hex}"), character_id,
                        str(row.get("status", "open")), str(row.get("created_at") or timestamp), row.get("closed_at"),
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
                        row.get("topic"), json.dumps(row.get("metadata", {})), str(row.get("created_at") or timestamp),
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
                        str(row.get("id") or f"mem_{uuid.uuid4().hex}"), character_id,
                        str(row.get("kind", "self_history")), row.get("topic"), str(row.get("content", "")),
                        str(row.get("epistemic_type", "observation")), float(row.get("confidence", 1.0)),
                        float(row.get("salience", 0.5)), json.dumps(row.get("source_event_ids", [])),
                        str(row.get("status", "active")), row.get("superseded_by"), json.dumps(row.get("metadata", {})),
                        str(row.get("created_at") or timestamp), str(row.get("updated_at") or timestamp),
                    ),
                )
            for row in links:
                conn.execute(
                    "INSERT INTO event_links VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(row.get("id") or f"lnk_{uuid.uuid4().hex}"), character_id,
                        str(row.get("from_event_id", "")), str(row.get("to_event_id", "")),
                        str(row.get("relationship", "related_to")), json.dumps(row.get("evidence", [])),
                        str(row.get("created_at") or timestamp),
                    ),
                )
            for row in mutations:
                conn.execute(
                    "INSERT INTO mutation_audit VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(row.get("id") or f"mut_{uuid.uuid4().hex}"), character_id,
                        json.dumps(row.get("proposal", {})), str(row.get("status", "allowed")),
                        str(row.get("reason", "Imported snapshot record.")), str(row.get("created_at") or timestamp),
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
                        str(row.get("created_at") or timestamp),
                    ),
                )


def diff_display_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 400:
        return value[:397] + "..."
    if isinstance(value, list) and len(value) > 12:
        return {"items": len(value), "preview": [diff_display_value(item) for item in value[:12]]}
    if isinstance(value, dict) and len(value) > 20:
        keys = sorted(value)[:20]
        return {key: diff_display_value(value[key]) for key in keys} | {"_truncated_keys": len(value) - len(keys)}
    return value


def snapshot_diff(
    before: Any,
    after: Any,
    path: str = "",
    changes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Produce a bounded, ID-aware structural diff suitable for profile review."""

    changes = changes if changes is not None else []
    if len(changes) >= 250:
        return changes
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}" if path else key
            if key not in before:
                changes.append({"kind": "added", "path": child_path, "after": diff_display_value(after[key])})
            elif key not in after:
                changes.append({"kind": "removed", "path": child_path, "before": diff_display_value(before[key])})
            else:
                snapshot_diff(before[key], after[key], child_path, changes)
            if len(changes) >= 250:
                break
        return changes
    if isinstance(before, list) and isinstance(after, list):
        if all(isinstance(item, dict) and "id" in item for item in before + after):
            before_by_id = {str(item["id"]): item for item in before}
            after_by_id = {str(item["id"]): item for item in after}
            return snapshot_diff(before_by_id, after_by_id, path, changes)
        if before != after:
            changes.append({
                "kind": "changed", "path": path,
                "before": diff_display_value(before), "after": diff_display_value(after),
            })
        return changes
    if before != after:
        changes.append({
            "kind": "changed", "path": path,
            "before": diff_display_value(before), "after": diff_display_value(after),
        })
    return changes
