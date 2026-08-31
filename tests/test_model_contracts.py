import pytest
from pydantic import ValidationError

from services.common import (
    ExecutiveRepeatAssessment,
    ExecutiveTurn,
    LeftAnalysis,
    RightAnalysis,
    output_model_for,
)


def test_left_contract_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LeftAnalysis.model_validate({
            "topic": "self.birthplace",
            "fact_refs": ["identity.birthplace"],
            "constraints": ["preserve_core"],
            "action": "answer",
            "confidence": 0.8,
            "untrusted_extra": "must not pass through a worker",
        })


def test_left_contract_bounds_compact_fact_references():
    with pytest.raises(ValidationError):
        LeftAnalysis.model_validate({
            "topic": "self.birthplace",
            "fact_refs": ["identity.birthplace"] * 5,
            "constraints": ["preserve_core"],
            "action": "answer",
            "confidence": 0.8,
        })


def test_compact_contract_recovers_local_model_field_variants():
    left = LeftAnalysis.model_validate({
        "fact_refs": ["identity.birthplace"],
        "constraints": ["preserve_core"],
        "action": "answer",
        "confidence": 0.8,
    })
    right = RightAnalysis.model_validate({
        "action": ["inform"],
        "affect": {"curiosity": 0.5},
        "tone": "warm",
        "risk": "low",
        "association_keys": [],
    })

    assert left.topic == "topic.general"
    assert right.action == "inform"


def test_executive_turn_contract_requires_nonempty_speech():
    with pytest.raises(ValidationError):
        ExecutiveTurn.model_validate({
            "goal": "maintain continuity",
            "strategy": "answer from context",
            "speech": "",
            "topic": "topic.general",
            "factual_claims": [],
            "mutations": [],
            "memory_writes": [],
        })


def test_executive_turn_contract_requires_an_explicit_claim_list():
    payload = {
        "goal": "maintain continuity",
        "strategy": "answer from context",
        "speech": "How can I help?",
        "topic": "topic.general",
        "mutations": [],
        "memory_writes": [],
    }
    with pytest.raises(ValidationError, match="factual_claims"):
        ExecutiveTurn.model_validate(payload)

    assert ExecutiveTurn.model_validate({**payload, "factual_claims": []}).factual_claims == []


def test_only_executive_can_reflect():
    assert output_model_for("executive", "reflection").__name__ == "ExecutiveReflection"
    with pytest.raises(ValueError):
        output_model_for("left", "reflection")


def test_only_executive_can_produce_a_repeat_intent_assessment():
    assessment = ExecutiveRepeatAssessment.model_validate({
        "primary_hypothesis": "wants_a_different_angle",
        "alternative_hypotheses": ["checking_consistency"],
        "evidence_codes": ["exact_question_repeated"],
        "response_mode": "new_angle",
        "confidence": 0.6,
    })

    assert assessment.response_mode == "new_angle"
    assert output_model_for("executive", "repeat_assessment").__name__ == "ExecutiveRepeatAssessment"
    with pytest.raises(ValueError):
        output_model_for("right", "repeat_assessment")
