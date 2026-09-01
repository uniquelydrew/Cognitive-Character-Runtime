"""HTTP request and portable-snapshot contracts for the memory service.

Keeping these models separate from route registration makes the persistence
contract inspectable and reusable without importing the SQLite application.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from services.common import EpistemicType, MutationProposal


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
