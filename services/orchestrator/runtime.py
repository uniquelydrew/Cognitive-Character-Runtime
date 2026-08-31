"""Runtime composition for the orchestrator.

This module is intentionally thin.  It installs the evidence-driven historical
matcher while the large legacy route module is decomposed.  Keeping composition
here lets deployed behavior move to the new matcher without duplicating the chat
endpoint or maintaining two HTTP applications.

The compatibility adapter can be deleted when ``app.py`` is split into turn,
history, and reflection services; its output shape exists only because the
current turn pipeline still consumes the older ``repeat_review`` field names.
"""

from __future__ import annotations

from typing import Any

from services.orchestrator import app as orchestrator
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
    """Count a contiguous return to one historical thread.

    This replaces topic-name and lexical streak counting. A turn belongs to the
    thread only when it is the root itself or its own persisted match explicitly
    points to that root. As soon as conversation moves elsewhere, the streak ends.
    """

    depth = 1  # current user turn
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
    """Adapt the new matcher to the current Executive-facing review contract.

    ``topic`` and ``prior_times`` are accepted only for call-site compatibility.
    Neither participates in candidate selection. In particular, equal normalized
    topic strings no longer imply that a user repeated a question.
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
    """Always perform current Left/Right cognition, including on exact repeats.

    Reusing the previous lobe artifacts made repeated wording a shortcut around
    changed memory, goals, mutable state, and conversational context. The history
    matcher is cheap enough to recognize exact repetition without freezing the
    cognitive inputs that will be used to answer it.
    """

    return None


# Compatibility installation while the HTTP route module is decomposed. Python
# resolves these globals when ``chat`` executes, so the route itself need not be
# duplicated and there remains exactly one deployed turn implementation.
orchestrator.executive_repeat_review = evidence_history_review
orchestrator.immediate_repeat_lobe_reuse = no_repeat_lobe_reuse

app = orchestrator.app
