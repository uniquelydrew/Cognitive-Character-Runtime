"""Historical-relationship inference, of which repetition is one relationship."""

from __future__ import annotations

from typing import Any


def historical_relationships(*, message: str, topic: str, review: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose explicit relationships for executive audit and durable event links."""
    matched = review.get("matched_event_id")
    if not matched:
        return []
    relationship = "revisits" if review.get("semantic_repeat_candidate") else "related_to"
    lower = message.lower()
    if any(word in lower for word in ("but", "actually", "contradict", "different")):
        relationship = "challenges"
    return [{
        "relationship": relationship,
        "target_event_id": str(matched),
        "subject_key": review.get("subject_key") or topic,
        "confidence": float(review.get("confidence", 0.0)),
        "evidence": list(review.get("evidence", [])),
    }]


def merge_historical_relationships(
    *, heuristic: list[dict[str, Any]], proposed: Any, allowed_event_ids: set[str]
) -> list[dict[str, Any]]:
    """Accept only Executive relationships pointing to visible recorded events.

    The deterministic candidate remains a useful fallback. Executive proposals
    add non-repeat semantics such as clarification and support, but cannot
    manufacture event identifiers or overwrite immutable history.
    """
    merged = {(item["target_event_id"], item["relationship"]): item for item in heuristic}
    if not isinstance(proposed, list):
        return list(merged.values())
    for item in proposed:
        if not isinstance(item, dict):
            continue
        prior_id = str(item.get("prior_event_id") or "")
        relation = str(item.get("relationship") or "")
        if prior_id not in allowed_event_ids or relation not in {
            "revisits", "follows_up", "clarifies", "challenges", "contradicts", "supports",
        }:
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        merged[(prior_id, relation)] = {
            "relationship": relation,
            "target_event_id": prior_id,
            "subject_key": None,
            "confidence": confidence,
            "evidence": ["executive_historical_relationship"],
        }
    return list(merged.values())
