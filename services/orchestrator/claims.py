"""Claim-level verification for the only model output that speaks to users."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def claim_evidence_catalog(character: Any, state: dict[str, Any], knowledge: list[dict[str, Any]]) -> dict[str, str]:
    """Build stable, model-visible citations without exposing write capabilities."""
    evidence: dict[str, str] = {}
    for key, value in character.identity.items():
        evidence[f"identity.{key}"] = str(value)
    for key, value in character.beliefs.items():
        evidence[f"primer.belief.{key}"] = str(value)
    for key, value in state.get("beliefs", {}).items():
        if isinstance(value, dict):
            evidence[f"belief.{key}"] = str(value.get("value"))
    for item in knowledge:
        if isinstance(item, dict) and item.get("id"):
            evidence[f"knowledge.{item['id']}"] = "; ".join(map(str, item.get("assertions", [])))
    return evidence


def verify_factual_claims(executive: dict[str, Any], evidence: dict[str, str]) -> list[dict[str, Any]]:
    """Return an audit record or reject claims that cite unavailable evidence.

    Verification is deliberately provenance-based: it proves a claim is tied to
    authorized source material, rather than pretending lexical matching proves
    semantic truth.
    """
    claims = executive.get("factual_claims", [])
    if not isinstance(claims, list):
        raise HTTPException(502, "Executive factual_claims must be a list.")
    audit: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise HTTPException(502, "Executive returned an invalid factual claim.")
        refs = claim.get("evidence_refs", [])
        unknown = [ref for ref in refs if ref not in evidence]
        status = "verified" if not unknown else "rejected"
        audit.append({"text": claim.get("text", ""), "evidence_refs": refs, "status": status, "unknown_refs": unknown})
    rejected = [claim for claim in audit if claim["status"] == "rejected"]
    if rejected:
        raise HTTPException(422, "Executive response rejected: a factual claim cited unavailable evidence.")
    return audit
