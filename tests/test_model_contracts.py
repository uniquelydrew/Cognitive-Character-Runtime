import pytest
from pydantic import ValidationError

from services.common import ExecutiveTurn, LeftAnalysis, output_model_for


def test_left_contract_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LeftAnalysis.model_validate({
            "topic": "self.birthplace",
            "observations": [],
            "consistency_constraints": [],
            "recommended_strategy": "answer from established state",
            "confidence": 0.8,
            "untrusted_extra": "must not pass through a worker",
        })


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
