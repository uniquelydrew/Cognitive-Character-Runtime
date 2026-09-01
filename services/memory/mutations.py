"""Runtime mutation policy and transactional application for character state."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from fastapi import HTTPException
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
from services.memory.storage import now_iso


class MutationBatch(BaseModel):
    proposals: list[MutationProposal] = Field(default_factory=list)


class TurnCommit(BaseModel):
    """All durable outputs of a successful cognitive turn, committed together."""

    model_config = {"extra": "forbid"}

    character_event: EventRecord
    memories: list[MemoryRecord] = Field(default_factory=list, max_length=20)
    proposals: list[MutationProposal] = Field(default_factory=list, max_length=25)


def validate_proposal(
    proposal: MutationProposal,
    *,
    allowed_mutable_keys: set[str] | None = None,
    evidence_event_ids: set[str] | None = None,
    goal_ids: set[str] | None = None,
    memory_ids: set[str] | None = None,
) -> ValidatedMutation:
    """Apply the narrow state-transition policy to an Executive proposal."""

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


def apply_mutations(
    character_id: str,
    batch: MutationBatch,
    conn: sqlite3.Connection,
) -> list[ValidatedMutation]:
    """Apply already-validated proposals inside a caller-owned transaction."""

    results: list[ValidatedMutation] = []
    timestamp = now_iso()
    character_row = conn.execute("SELECT document_json FROM characters WHERE id=?", (character_id,)).fetchone()
    if not character_row:
        raise HTTPException(404, "Character not found")
    character = CharacterDocument.model_validate_json(character_row["document_json"])
    allowed_mutable_keys = set(character.mutable_state) | {"topic_defensiveness"}
    evidence_event_ids = {str(row["id"]) for row in conn.execute("SELECT id FROM events WHERE character_id=?", (character_id,))}
    goal_ids = {str(row["id"]) for row in conn.execute("SELECT id FROM goals WHERE character_id=?", (character_id,))}
    memory_ids = {str(row["id"]) for row in conn.execute("SELECT id FROM memories WHERE character_id=?", (character_id,))}
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
            (f"mut_{uuid.uuid4().hex}", character_id, proposal.model_dump_json(), checked.status, checked.reason, timestamp),
        )
        if checked.status == "rejected":
            continue
        if proposal.operation == MutationOperation.SET_MUTABLE_STATE:
            conn.execute(
                """
                INSERT INTO mutable_state(character_id, key, value_json, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(character_id, key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (character_id, proposal.target, json.dumps(proposal.value), timestamp),
            )
        elif proposal.operation == MutationOperation.SET_BELIEF:
            old = conn.execute(
                "SELECT value_json, revision FROM beliefs WHERE character_id=? AND key=?", (character_id, proposal.target)
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
                (character_id, proposal.target, json.dumps(proposal.value), proposal.confidence,
                 proposal.epistemic_type.value, json.dumps(proposal.evidence), revision, timestamp),
            )
            conn.execute(
                "INSERT INTO belief_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"bh_{uuid.uuid4().hex}", character_id, proposal.target, old_json,
                 json.dumps(proposal.value), proposal.confidence, json.dumps(proposal.evidence), proposal.reason, timestamp),
            )
        elif proposal.operation == MutationOperation.ADD_MEMORY:
            conn.execute(
                """INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, '{}', ?, ?)""",
                (f"mem_{uuid.uuid4().hex}", character_id, proposal.target, proposal.target,
                 str(proposal.value), proposal.epistemic_type.value, proposal.confidence, 0.6,
                 json.dumps(proposal.evidence), timestamp, timestamp),
            )
        elif proposal.operation == MutationOperation.ADD_GOAL:
            conn.execute(
                "INSERT INTO goals VALUES (?, ?, ?, 'active', '{}', ?, ?)",
                (f"goal_{uuid.uuid4().hex}", character_id, str(proposal.value), timestamp, timestamp),
            )
        elif proposal.operation == MutationOperation.UPDATE_GOAL:
            conn.execute(
                "UPDATE goals SET status=?, updated_at=? WHERE id=? AND character_id=?",
                (str(proposal.value), timestamp, proposal.target, character_id),
            )
        elif proposal.operation == MutationOperation.LINK_EVENTS:
            data = proposal.value if isinstance(proposal.value, dict) else {}
            from_id, to_id = data.get("from"), data.get("to")
            relationship = data.get("relationship", "related_to")
            if from_id and to_id:
                conn.execute(
                    """INSERT OR IGNORE INTO event_links VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (f"lnk_{uuid.uuid4().hex}", character_id, from_id, to_id, relationship,
                     json.dumps(proposal.evidence), timestamp),
                )
        elif proposal.operation == MutationOperation.SUPERSEDE_MEMORY:
            conn.execute(
                "UPDATE memories SET status='superseded', superseded_by=?, updated_at=? WHERE id=? AND character_id=?",
                (str(proposal.value), timestamp, proposal.target, character_id),
            )
    return results
