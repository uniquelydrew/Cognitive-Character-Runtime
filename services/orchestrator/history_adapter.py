"""Compatibility between evidence-driven history matching and the current turn contract.

Delete this module once the turn pipeline consumes ``HistoryMatch`` directly.
It contains no matching policy of its own; it only translates field names and
computes contiguous thread depth for the existing repeat-dynamics interface.
"""

from __future__ import annotations

from typing import Any

from services.orchestrator.history_matching import answered_turns, history_report, match_history


def _recent_turns(events: list[dict[str, Any]], current_event_id: str) -> list[dict[str, Any]]:
    return [
        {
            "event_id": event.get("id"),
            "event_type": event.get("event_type"),
            "actor": event.get("actor"),
            "content": event.get("content"),
            "topic": event.get("topic"),
        }
        for event in events
        if event.get("id") != current_event_id
        and event.get("event_type") in {"user_message", "character_message"}
    ][-8:]


def _thread_depth(prior_turns: list[Any], root_user_event_id: str, matched_event_id: str) -> int:
    depth = 1
    for prior in reversed(prior_turns):
        prior_root = str(prior.prior_match.get("root_user_event_id") or "")
        belongs = prior.user_event_id in {root_user_event_id, matched_event_id} or prior_root == root_user_event_id
        if not belongs:
            break
        depth += 1
    return depth


def evidence_history_review(
    *,
    message: str,
    topic: str,
    current_event_id: str,
    session_events: list[dict[str, Any]],
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    prior_times: int,
    embedding_matches: dict[str, float] | None = None,
    embedding_threshold: float = 0.80,
) -> dict[str, Any]:
    """Translate a ``HistoryMatch`` into the legacy Executive review shape.

    ``topic`` and ``prior_times`` remain in the signature only because the current
    route still supplies them. They are explicitly excluded from match evidence.
    """

    del topic, prior_times
    prior = answered_turns(session_events, current_user_event_id=current_event_id)
    match = match_history(
        message=message,
        current_left=left_result,
        current_right=right_result,
        prior_turns=prior,
        embedding_matches=embedding_matches,
        embedding_threshold=embedding_threshold,
    )
    report = history_report(match)
    if match is None:
        return {
            "semantic_repeat_candidate": False,
            "subject_key": "thread:new",
            "matched_event_id": None,
            "matched_answer": None,
            "confidence": 0.0,
            "embedding_similarity": None,
            "embedding_threshold": embedding_threshold,
            "reason": "no evidenced historical match",
            "consecutive_repeats": 1,
            "recent_turns": _recent_turns(session_events, current_event_id),
            "root_user_event_id": None,
            "signals": [],
            "history_match": report,
        }

    embedding_signal = next(
        (signal for signal in match.signals if signal.kind == "embedding_similarity"),
        None,
    )
    embedding_similarity = None
    if embedding_signal is not None and "=" in embedding_signal.detail:
        try:
            embedding_similarity = float(embedding_signal.detail.split("=", 1)[1])
        except ValueError:
            embedding_similarity = None

    subject_key = match.subject_hint or f"thread:{match.root_user_event_id}"
    return {
        "semantic_repeat_candidate": True,
        "subject_key": subject_key,
        "matched_event_id": match.user_event_id,
        "matched_answer": match.answer,
        "confidence": round(match.score, 4),
        "embedding_similarity": embedding_similarity,
        "embedding_threshold": embedding_threshold,
        "reason": "+".join(signal.kind for signal in match.signals),
        "consecutive_repeats": _thread_depth(prior, match.root_user_event_id, match.user_event_id),
        "recent_turns": _recent_turns(session_events, current_event_id),
        "root_user_event_id": match.root_user_event_id,
        "signals": report["signals"],
        "history_match": report,
    }


def no_repeat_lobe_reuse(**_: Any) -> None:
    """Force current cognition even when surface text is exactly repeated."""

    return None
