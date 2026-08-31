from __future__ import annotations

import pytest
from fastapi import HTTPException

from services.orchestrator.claims import verify_factual_claims
from services.orchestrator.relationships import historical_relationships


def test_claim_verification_keeps_a_durable_citation_audit() -> None:
    audit = verify_factual_claims(
        {"factual_claims": [{"text": "I was born in Northbridge.", "evidence_refs": ["identity.birthplace"]}]},
        {"identity.birthplace": "Northbridge"},
    )
    assert audit[0]["status"] == "verified"


def test_claim_verification_rejects_unknown_citations() -> None:
    with pytest.raises(HTTPException, match="rejected"):
        verify_factual_claims(
            {"factual_claims": [{"text": "Unsupported.", "evidence_refs": ["made.up"]}]},
            {},
        )


def test_historical_relationships_are_not_limited_to_revisits() -> None:
    relationships = historical_relationships(
        message="But that contradicts what you said.",
        topic="topic.history",
        review={"matched_event_id": "evt_prior", "semantic_repeat_candidate": True, "confidence": 0.8},
    )
    assert relationships[0]["relationship"] == "challenges"
