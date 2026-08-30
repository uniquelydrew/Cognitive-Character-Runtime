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


class RepeatDynamics(BaseModel):
    """Deterministic conversational pressure supplied to the executive role."""

    conversation_patience: float = Field(default=1.0, ge=0.0, le=1.0)
    subject_defensiveness: float = Field(default=0.0, ge=0.0, le=1.0)
    intersection_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    response_posture: Literal["normal", "reclarify", "confused", "defensive"] = "normal"
    semantic_repeat: bool = False
    consecutive_repeats: int = Field(default=1, ge=1)
    subject_key: str | None = None
    review_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


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
    repeat_dynamics: RepeatDynamics = Field(default_factory=RepeatDynamics)


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


class ModelOutput(BaseModel):
    """Base class for data returned by a cognitive model.

    Model output is untrusted input.  The workers validate it against one of the
    concrete contracts below before it can reach the orchestrator or memory API.
    """

    model_config = {"extra": "forbid"}


class LeftAnalysis(ModelOutput):
    topic: str = Field(min_length=1, max_length=160)
    observations: list[str] = Field(default_factory=list, max_length=12)
    consistency_constraints: list[str] = Field(default_factory=list, max_length=12)
    recommended_strategy: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class RightAnalysis(ModelOutput):
    social_read: str = Field(min_length=1, max_length=500)
    affect: dict[str, float] = Field(default_factory=dict)
    recommended_tone: str = Field(min_length=1, max_length=160)
    associations: list[str] = Field(default_factory=list, max_length=12)


class MemoryWrite(ModelOutput):
    kind: str = Field(min_length=1, max_length=120)
    topic: str | None = Field(default=None, max_length=160)
    content: str = Field(min_length=1, max_length=8_000)
    epistemic_type: EpistemicType = EpistemicType.SELF_STATEMENT
    confidence: float = Field(ge=0.0, le=1.0)
    salience: float = Field(ge=0.0, le=1.0)


class ExecutiveTurn(ModelOutput):
    goal: str = Field(min_length=1, max_length=500)
    strategy: str = Field(min_length=1, max_length=500)
    speech: str = Field(min_length=1, max_length=8_000)
    topic: str = Field(min_length=1, max_length=160)
    mutations: list[MutationProposal] = Field(default_factory=list, max_length=20)
    memory_writes: list[MemoryWrite] = Field(default_factory=list, max_length=10)


class EventLink(ModelOutput):
    source_event_id: str = Field(min_length=1, max_length=120)
    target_event_id: str = Field(min_length=1, max_length=120)
    relationship: str = Field(min_length=1, max_length=120)


class ExecutiveReflection(ModelOutput):
    summary: str = Field(min_length=1, max_length=8_000)
    related_event_ids: list[str] = Field(default_factory=list, max_length=100)
    mutations: list[MutationProposal] = Field(default_factory=list, max_length=50)
    links: list[EventLink] = Field(default_factory=list, max_length=50)


def output_model_for(role: str, mode: str) -> type[ModelOutput]:
    """Return the only permitted output type for a worker role/mode pair."""

    if role == "left" and mode == "turn":
        return LeftAnalysis
    if role == "right" and mode == "turn":
        return RightAnalysis
    if role == "executive" and mode == "turn":
        return ExecutiveTurn
    if role == "executive" and mode == "reflection":
        return ExecutiveReflection
    raise ValueError(f"Unsupported cognitive role/mode: {role}/{mode}")
