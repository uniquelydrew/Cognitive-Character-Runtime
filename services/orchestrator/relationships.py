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
