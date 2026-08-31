"""Executive-turn safeguards that sit between model output and persistence."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException

from services.common import CognitiveRequest
from services.orchestrator.claims import verify_factual_claims
from services.orchestrator.inference import infer_timed


async def infer_verified_turn(
    client: httpx.AsyncClient,
    executive_url: str,
    request: CognitiveRequest,
    evidence: dict[str, str],
    metrics: dict[str, dict[str, int | float | None]],
) -> tuple[dict[str, Any], list[dict[str, Any]], int, bool]:
    """Generate and verify executive speech, allowing one focused repair pass.

    The repair is not a fallback citation generator: the Executive must emit a
    fresh structured turn whose own claim text entails the cited source.  This
    keeps unsupported speech rejectable while making the contract usable with
    compact local models that occasionally omit an otherwise-required field.
    """

    executive, elapsed = await infer_timed(client, executive_url, request, "executive", metrics)
    speech = str(executive.get("speech", "")).strip()
    if not speech:
        raise HTTPException(502, "Executive produced no speech")
    executive = {**executive, "speech": speech}
    try:
        return executive, verify_factual_claims(executive, evidence), elapsed, False
    except HTTPException as exc:
        if exc.status_code != 422:
            raise

    retry_context = {
        **request.context,
        "claim_coverage_retry": {
            "rejected_speech": speech[:1600],
            "requirement": (
                "Return a fresh turn. Every declarative factual sentence in speech must have a "
                "factual_claims item whose text includes the fact and whose evidence_refs names an "
                "available context.claim_evidence key. If you cannot cite it, remove the factual sentence."
            ),
        },
    }
    retry_request = request.model_copy(update={"context": retry_context})
    executive, retry_elapsed = await infer_timed(client, executive_url, retry_request, "executive", metrics)
    speech = str(executive.get("speech", "")).strip()
    if not speech:
        raise HTTPException(502, "Executive produced no speech on claim-coverage revision")
    executive = {**executive, "speech": speech}
    return executive, verify_factual_claims(executive, evidence), elapsed + retry_elapsed, True
