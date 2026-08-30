import pytest
from pydantic import ValidationError

from services.common import ExecutiveTurn, LeftAnalysis, RightAnalysis, output_model_for


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
            "mutations": [],
            "memory_writes": [],
        })


def test_only_executive_can_reflect():
    assert output_model_for("executive", "reflection").__name__ == "ExecutiveReflection"
    with pytest.raises(ValueError):
        output_model_for("left", "reflection")
