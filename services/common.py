from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    model_config = {"extra": "forbid"}
    operation: MutationOperation
    target: str = Field(min_length=1, max_length=160)
    value: Any = None
    old_value: Any = None
    evidence: list[str] = Field(default_factory=list, max_length=100)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    epistemic_type: EpistemicType = EpistemicType.INFERENCE
    reason: str = Field(default="", max_length=4_000)


class ValidatedMutation(BaseModel):
    model_config = {"extra": "forbid"}
    proposal: MutationProposal
    status: Literal["allowed", "versioned", "rejected"]
    reason: str


class MemoryRecord(BaseModel):
    model_config = {"extra": "forbid"}
    id: str | None = Field(default=None, min_length=1, max_length=120)
    character_id: str = Field(min_length=1, max_length=64)
    kind: str = Field(min_length=1, max_length=120)
    topic: str | None = Field(default=None, min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=8_000)
    epistemic_type: EpistemicType = EpistemicType.OBSERVATION
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(default_factory=list, max_length=100)
    status: Literal["active", "superseded"] = "active"
    superseded_by: str | None = Field(default=None, min_length=1, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventRecord(BaseModel):
    model_config = {"extra": "forbid"}
    id: str | None = None
    character_id: str
    session_id: str | None = None
    event_type: str
    actor: str | None = None
    content: str
    topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CharacterDocument(BaseModel):
    # Profiles are executable inputs to the model pipeline. Silently dropping
    # unknown YAML keys hides typos and creates a path for unreviewed behaviour.
    model_config = {"extra": "forbid"}
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
    # Explicit knowledge affiliations supplement labels derived from identity,
    # occupation, faction, and birthplace. They are never conversation mutations.
    knowledge_labels: list[str] = Field(default_factory=list, max_length=100)
    biography: str = ""


KnowledgeLabel = Annotated[
    str,
    Field(min_length=2, max_length=96, pattern=r"^[a-z][a-z0-9_.-]*$"),
]


class KnowledgeClassification(BaseModel):
    """One node in the source-of-truth general-knowledge classification DAG."""

    model_config = {"extra": "forbid"}

    id: KnowledgeLabel
    parents: list[KnowledgeLabel] = Field(default_factory=list, max_length=12)
    aliases: list[str] = Field(default_factory=list, max_length=24)
    description: str = Field(default="", max_length=500)


class KnowledgeAccessRule(BaseModel):
    """Access conditions evaluated against a character's derived label closure."""

    model_config = {"extra": "forbid"}

    require_all: list[KnowledgeLabel] = Field(default_factory=list, max_length=24)
    require_any: list[KnowledgeLabel] = Field(default_factory=list, max_length=24)


class GeneralKnowledgeRecord(BaseModel):
    """A reusable fact set, deliberately distinct from events and memories."""

    model_config = {"extra": "forbid"}

    id: KnowledgeLabel
    labels: list[KnowledgeLabel] = Field(min_length=1, max_length=24)
    access: KnowledgeAccessRule = Field(default_factory=KnowledgeAccessRule)
    assertions: list[str] = Field(min_length=1, max_length=24)
    epistemic_type: EpistemicType = EpistemicType.FACT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="", max_length=500)


class KnowledgeCatalog(BaseModel):
    """A YAML source file containing taxonomy nodes and labeled knowledge records."""

    model_config = {"extra": "forbid"}

    classifications: list[KnowledgeClassification] = Field(default_factory=list, max_length=10_000)
    records: list[GeneralKnowledgeRecord] = Field(default_factory=list, max_length=100_000)


class RepeatDynamics(BaseModel):
    """Measured repeat pressure and a non-binding escalation recommendation."""

    conversation_patience: float = Field(default=1.0, ge=0.0, le=1.0)
    subject_defensiveness: float = Field(default=0.0, ge=0.0, le=1.0)
    intersection_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    response_posture: Literal["normal", "reclarify", "confused", "defensive"] = "normal"
    suggested_posture: Literal["normal", "reclarify", "confused", "defensive"] = "normal"
    escalation_recommendation: Literal["hold", "increase", "deescalate"] = "hold"
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
    mode: Literal["turn", "reflection", "repeat_assessment"] = "turn"


class CognitiveResponse(BaseModel):
    role: str
    result: dict[str, Any]


class ModelOutput(BaseModel):
    """Base class for data returned by a cognitive model.

    Model output is untrusted input.  The workers validate it against one of the
    concrete contracts below before it can reach the orchestrator or memory API.
    """

    model_config = {"extra": "forbid"}


CompactAtom = Annotated[str, Field(min_length=1, max_length=96)]
# Small local models occasionally expand a requested code into a short phrase.
# Preserve the bounded control-artifact contract without turning that recoverable
# phrasing variation into a failed user turn.
CompactCode = Annotated[str, Field(min_length=1, max_length=96)]


class LeftAnalysis(ModelOutput):
    """A compact analytic control artifact, not prose for the user."""

    # The orchestrator already has a deterministic topic for every turn. A small
    # local model may omit this redundant field after otherwise completing JSON.
    topic: CompactAtom = "topic.general"
    fact_refs: list[CompactAtom] = Field(default_factory=list, max_length=4)
    constraints: list[CompactCode] = Field(default_factory=list, max_length=3)
    action: CompactCode
    confidence: float = Field(ge=0.0, le=1.0)


class RightAnalysis(ModelOutput):
    """A compact relational control artifact, not prose for the user."""

    action: CompactCode
    affect: dict[CompactCode, float] = Field(default_factory=dict, max_length=4)
    tone: CompactCode
    risk: CompactCode
    association_keys: list[CompactAtom] = Field(default_factory=list, max_length=4)

    @field_validator("action", mode="before")
    @classmethod
    def coerce_single_action(cls, value: Any) -> Any:
        """Accept a local model's one-item action list as its scalar control key."""

        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            return value[0]
        return value


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
    # Repeat pressure is evidence, not an automatic emotional reaction. The
    # executive decides whether this turn should alter durable defensiveness.
    repeat_escalation: Literal["hold", "increase", "deescalate"] = "hold"
    mutations: list[MutationProposal] = Field(default_factory=list, max_length=20)
    memory_writes: list[MemoryWrite] = Field(default_factory=list, max_length=10)


class ExecutiveRepeatAssessment(ModelOutput):
    """Bounded executive hypothesis work for a repeated user question.

    These are possible explanations for the repeat, not claims about the user.
    Keeping them as compact labels avoids storing private-style reasoning while
    giving the speaking executive a concrete decision aid.
    """

    primary_hypothesis: CompactCode = "unclear_repeat_intent"
    alternative_hypotheses: list[CompactCode] = Field(default_factory=list, max_length=4)
    evidence_codes: list[CompactCode] = Field(default_factory=list, max_length=4)
    response_mode: Literal[
        "new_angle",
        "check_understanding",
        "invite_specificity",
        "test_consistency",
        "set_boundary",
    ] = "invite_specificity"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


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
    if role == "executive" and mode == "repeat_assessment":
        return ExecutiveRepeatAssessment
    if role == "executive" and mode == "reflection":
        return ExecutiveReflection
    raise ValueError(f"Unsupported cognitive role/mode: {role}/{mode}")
