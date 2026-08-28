from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EpistemicType(StrEnum):
    FACT = "fact"
    OBSERVATION = "observation"
    SELF_STATEMENT = "self_statement"
    OTHER_STATEMENT = "other_statement"
    BELIEF = "belief"
    INFERENCE = "inference"
    SUSPICION = "suspicion"
    RUMOR = "rumor"
    LIE = "lie"
    UNKNOWN = "unknown"


class MutationOperation(StrEnum):
    SET_MUTABLE_STATE = "set_mutable_state"
    SET_BELIEF = "set_belief"
    ADD_GOAL = "add_goal"
    UPDATE_GOAL = "update_goal"
    ADD_MEMORY = "add_memory"
    LINK_EVENTS = "link_events"
    SUPERSEDE_MEMORY = "supersede_memory"
    UPDATE_CORE = "update_core"


class MutationProposal(BaseModel):
    operation: MutationOperation
    target: str
    value: Any = None
    old_value: Any = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    epistemic_type: EpistemicType = EpistemicType.INFERENCE
    reason: str = ""


class ValidatedMutation(BaseModel):
    proposal: MutationProposal
    status: Literal["allowed", "versioned", "rejected"]
    reason: str


class MemoryRecord(BaseModel):
    id: str | None = None
    character_id: str
    kind: str
    topic: str | None = None
    content: str
    epistemic_type: EpistemicType = EpistemicType.OBSERVATION
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(default_factory=list)
    status: Literal["active", "superseded"] = "active"
    superseded_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventRecord(BaseModel):
    id: str | None = None
    character_id: str
    session_id: str | None = None
    event_type: str
    actor: str | None = None
    content: str
    topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CharacterDocument(BaseModel):
    id: str
    identity: dict[str, Any]
    traits: dict[str, float] = Field(default_factory=dict)
    cognition: dict[str, Any] = Field(default_factory=dict)
    speech: dict[str, Any] = Field(default_factory=dict)
    values: list[str] = Field(default_factory=list)
    inhibitions: list[str] = Field(default_factory=list)
    initial_goals: list[str] = Field(default_factory=list)
    mutable_state: dict[str, Any] = Field(default_factory=dict)
    beliefs: dict[str, Any] = Field(default_factory=dict)
    biography: str = ""


class InteractionClassification(BaseModel):
    interaction_type: Literal[
        "new_subject",
        "repeated_question",
        "paraphrase",
        "challenge",
        "contradiction",
        "follow_up",
    ] = "new_subject"
    topic: str | None = None
    prior_answer: str | None = None
    times_asked: int = 0
    related_event_ids: list[str] = Field(default_factory=list)


class CognitiveRequest(BaseModel):
    character: CharacterDocument
    user_input: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    left_result: dict[str, Any] | None = None
    right_result: dict[str, Any] | None = None
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    mode: Literal["turn", "reflection"] = "turn"


class CognitiveResponse(BaseModel):
    role: str
    result: dict[str, Any]
