"""Claim-level verification for the only model output that speaks to users."""

from __future__ import annotations

import re
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


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOP = {"a", "an", "the", "i", "you", "was", "is", "are", "in", "of", "to", "and", "my", "your", "that"}
_FACT_CUE = re.compile(r"\b(?:born|from|name|occupation|work|am|is|are|was|were|has|have)\b", re.I)


def _terms(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOP and len(token) > 1}


def _entails(claim: str, source: str) -> bool:
    """Conservative lexical entailment for source-grounded runtime facts.

    An identity/catalog value must appear in the asserted proposition. This
    rejects a citation to ``Northbridge`` for an assertion about Greyhaven and
    avoids treating mere source selection as proof.
    """
    claim_terms, source_terms = _terms(claim), _terms(source)
    return bool(source_terms) and source_terms.issubset(claim_terms)


def _factual_sentences(speech: str) -> list[str]:
    """Return conservative declarative factual candidates requiring coverage."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", speech)
        if sentence.strip() and not sentence.rstrip().endswith("?") and _FACT_CUE.search(sentence)
    ]


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
        text = str(claim.get("text", ""))
        unknown = [ref for ref in refs if ref not in evidence]
        contradictions = []
        if not unknown and not any(_entails(text, evidence[ref]) for ref in refs):
            contradictions = refs
        status = "verified" if not unknown and not contradictions else "rejected"
        audit.append({"text": text, "evidence_refs": refs, "status": status, "unknown_refs": unknown, "contradicted_refs": contradictions})
    rejected = [claim for claim in audit if claim["status"] == "rejected"]
    if rejected:
        raise HTTPException(422, "Executive response rejected: a factual claim is unsupported or contradicts its cited evidence.")
    covered = [claim["text"] for claim in audit]
    uncovered = [
        sentence for sentence in _factual_sentences(str(executive.get("speech", "")))
        if not any(_terms(sentence) & _terms(claim) for claim in covered)
    ]
    if uncovered:
        raise HTTPException(422, "Executive response rejected: factual speech is not covered by a verified claim.")
    return audit
