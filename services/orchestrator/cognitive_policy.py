"""Deterministic cognitive policy used by the orchestrator turn pipeline.

The policy is independent of FastAPI and receives runtime-dependent limits as
arguments, so the route module only composes configuration and I/O.
"""

from __future__ import annotations

import re
from math import isfinite, sqrt
from typing import Any

import httpx

from services.common import CharacterDocument


CONTENT_TOKEN_RE = re.compile(r"[a-z0-9']+")
CONTENT_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "did", "does", "you", "your",
    "yours", "i", "me", "my", "what", "where", "when", "who", "why", "how", "again",
    "tell", "said", "say", "about", "to", "of", "in", "on", "please", "could", "would",
    "it", "that", "this", "and", "or", "for", "with", "have", "has", "had", "be", "been",
}


def normalize_topic(text: str) -> str:
    """Produce a cheap deterministic topic key for the bootstrap runtime."""

    lower = text.lower().strip()
    if any(term in lower for term in ("where were you born", "where are you from", "birthplace", "hometown", "birth town")):
        return "self.birthplace"
    if any(term in lower for term in ("what is your name", "what's your name", "who are you")):
        return "self.name"
    if any(term in lower for term in ("what do you do", "your job", "occupation", "your work", "work as")):
        return "self.occupation"
    kept = sorted(content_tokens(lower))
    return "topic." + (".".join(kept[:8]) or "general")


def content_tokens(text: str) -> set[str]:
    """Return the small deterministic lexical signal used by policy checks."""

    return {
        token
        for token in CONTENT_TOKEN_RE.findall(text.lower())
        if token not in CONTENT_STOP_WORDS and len(token) > 1
    }


def token_similarity(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def resolve_cognitive_priorities(character: CharacterDocument | None) -> dict[str, Any]:
    """Normalize authored Left/Right priorities into an enforceable turn policy."""

    cognition = character.cognition if character is not None else {}

    def weight(name: str) -> float:
        value = cognition.get(name, 0.5) if isinstance(cognition, dict) else 0.5
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.5
        return numeric if isfinite(numeric) and numeric >= 0 else 0.5

    left_raw = weight("left_weight")
    right_raw = weight("right_weight")
    total = left_raw + right_raw
    left = 0.5 if total <= 0 else left_raw / total
    right = 0.5 if total <= 0 else right_raw / total
    difference = left - right
    primary_role = "balanced" if abs(difference) < 0.08 else ("left" if difference > 0 else "right")
    return {
        "left_weight": round(left, 4),
        "right_weight": round(right, 4),
        "primary_role": primary_role,
        "weight_gap": round(abs(difference), 4),
        "invariant": "left_constraints_bind",
        "enforcement": "weighted_arbitration_plan_and_role_attention_budget",
    }


def bounded_lobe_transcript(
    session_events: list[dict[str, Any]],
    *,
    max_events: int,
    max_characters: int,
    max_event_characters: int,
) -> list[dict[str, Any]]:
    """Return the recent raw conversation window within hard context limits."""

    selected: list[dict[str, Any]] = []
    remaining = max_characters
    conversation = [
        event
        for event in session_events
        if event.get("event_type") in {"user_message", "character_message"}
    ]
    for event in reversed(conversation[-max_events:]):
        content = str(event.get("content") or "").strip()
        if not content or remaining <= 0:
            continue
        permitted = min(max_event_characters, remaining)
        clipped = content[:permitted]
        selected.append(
            {
                "event_id": str(event.get("id") or ""),
                "event_type": str(event.get("event_type") or ""),
                "actor": event.get("actor"),
                "content": clipped,
                "topic": event.get("topic"),
                "content_truncated": len(clipped) < len(content),
            }
        )
        remaining -= len(clipped)
    return list(reversed(selected))


def weighted_arbitration_plan(
    priorities: dict[str, Any],
    left_result: dict[str, Any],
    right_result: dict[str, Any],
) -> dict[str, Any]:
    """Materialize authored weighting as bounded, auditable executive input."""

    primary = str(priorities["primary_role"])
    left_packet = {
        "weight": priorities["left_weight"],
        "action": str(left_result.get("action") or "answer"),
        "fact_refs": [str(item) for item in left_result.get("fact_refs", [])[:4]],
        "constraints": [str(item) for item in left_result.get("constraints", [])[:3]],
    }
    right_packet = {
        "weight": priorities["right_weight"],
        "action": str(right_result.get("action") or "inform"),
        "tone": str(right_result.get("tone") or "neutral"),
        "risk": str(right_result.get("risk") or "low"),
        "association_keys": [str(item) for item in right_result.get("association_keys", [])[:4]],
    }
    primary_packet = left_packet if primary == "left" else right_packet if primary == "right" else None
    return {
        "priorities": priorities,
        "primary_role": primary,
        "primary_packet": primary_packet,
        "left": left_packet,
        "right": right_packet,
        "binding_rules": [
            "left.constraints are binding; right affect or tone cannot override them",
            (
                f"when non-factual response choices conflict, favor the {primary} packet"
                if primary in {"left", "right"}
                else "when non-factual response choices conflict, balance both packets"
            ),
        ],
    }


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    try:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(a * a for a in right))
    except (TypeError, ValueError):
        return None
    if not isfinite(dot) or left_norm <= 0 or right_norm <= 0:
        return None
    score = dot / (left_norm * right_norm)
    return score if isfinite(score) else None


async def semantic_repeat_evidence(
    client: httpx.AsyncClient,
    message: str,
    prior_users: list[dict[str, Any]],
    *,
    embedding_url: str,
    embedding_model: str,
    timeout_seconds: float,
    similarity_threshold: float,
    max_candidates: int,
) -> dict[str, Any]:
    """Return bounded embedding evidence and fail closed on provider errors."""

    candidates = [
        event for event in prior_users[-max_candidates:]
        if str(event.get("content") or "").strip() and str(event.get("id") or "")
    ]
    if not embedding_url or not embedding_model or not candidates:
        return {
            "available": False,
            "model": embedding_model or None,
            "threshold": similarity_threshold,
            "matches": {},
            "reason": "no_embedding_candidates_or_configuration",
        }
    inputs = [message, *[str(event["content"]) for event in candidates]]
    try:
        response = await client.post(
            embedding_url,
            json={"model": embedding_model, "input": inputs, "truncate": True},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = payload.get("embeddings", []) if isinstance(payload, dict) else []
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise ValueError("embedding provider returned an unexpected vector count")
        parsed = [[float(value) for value in vector] for vector in vectors if isinstance(vector, list)]
        if len(parsed) != len(inputs):
            raise ValueError("embedding provider returned a non-vector value")
    except (httpx.HTTPError, TypeError, ValueError):
        return {
            "available": False,
            "model": embedding_model,
            "threshold": similarity_threshold,
            "matches": {},
            "reason": "embedding_provider_unavailable",
        }

    matches: dict[str, float] = {}
    for event, vector in zip(candidates, parsed[1:]):
        score = cosine_similarity(parsed[0], vector)
        if score is not None:
            matches[str(event["id"])] = round(score, 4)
    return {
        "available": True,
        "model": str(payload.get("model") or embedding_model),
        "threshold": similarity_threshold,
        "matches": matches,
        "reason": "embedding_similarity",
    }
