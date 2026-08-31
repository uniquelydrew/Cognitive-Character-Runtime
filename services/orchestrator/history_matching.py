"""Evidence-driven historical turn matching for persistent conversations.

The matcher deliberately does not try to infer why a user returned to earlier
material.  Its job is narrower: identify answered prior turns that the current
turn may relate to, preserve the independent evidence supporting each candidate,
and expose one bounded best match for Executive adjudication.

A repeated question is therefore not a primitive.  Exact repetition, semantic
similarity, shared grounded facts, shared associations, and lexical overlap are
independent signals of historical relatedness.  The Executive remains responsible
for deciding whether the relationship is a repeat, clarification, continuation,
challenge, contradiction, or something else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "did", "does", "you", "your",
    "yours", "i", "me", "my", "what", "where", "when", "who", "why", "how", "again",
    "tell", "said", "say", "about", "to", "of", "in", "on", "please", "could", "would",
    "it", "that", "this", "and", "or", "for", "with", "have", "has", "had", "be", "been",
}


@dataclass(frozen=True)
class AnsweredTurn:
    """One user event paired with the character reply that completed it."""

    user_event_id: str
    user_text: str
    user_topic: str | None
    answer_event_id: str
    answer_text: str
    answer_topic: str | None
    left: dict[str, Any] = field(default_factory=dict)
    right: dict[str, Any] = field(default_factory=dict)
    prior_match: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchSignal:
    """One independently observable reason a historical turn may be related."""

    kind: str
    strength: float
    detail: str


@dataclass(frozen=True)
class HistoryMatch:
    """A bounded historical candidate report consumed by the Executive."""

    user_event_id: str
    answer_event_id: str
    answer: str
    score: float
    signals: tuple[MatchSignal, ...]
    root_user_event_id: str
    subject_hint: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": True,
            "matched_event_id": self.user_event_id,
            "matched_answer_event_id": self.answer_event_id,
            "matched_answer": self.answer,
            "confidence": round(self.score, 4),
            "root_user_event_id": self.root_user_event_id,
            "subject_hint": self.subject_hint,
            "signals": [
                {"kind": signal.kind, "strength": round(signal.strength, 4), "detail": signal.detail}
                for signal in self.signals
            ],
        }


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOP_WORDS and len(token) > 1
    }


def canonical_text(text: str) -> str:
    """Normalize surface form only; this intentionally carries no semantics."""

    return " ".join(_TOKEN_RE.findall(text.lower()))


def lexical_similarity(left: str, right: str) -> float:
    """Jaccard similarity used only as one weak evidence provider."""

    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _compact_keys(analysis: dict[str, Any], *names: str) -> set[str]:
    values: set[str] = set()
    for name in names:
        raw = analysis.get(name, [])
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            values.update(
                str(item).strip().lower()
                for item in raw
                if isinstance(item, str) and item.strip()
            )
    return values


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def answered_turns(
    events: Iterable[dict[str, Any]], *, current_user_event_id: str | None = None, limit: int = 24,
) -> list[AnsweredTurn]:
    """Pair only completed user turns with their replies, preserving chronology.

    Unanswered user events are intentionally invisible to historical matching: a
    user retry after a failed turn should not be interpreted as conversational
    repetition merely because an input event was persisted.
    """

    materialized = list(events)
    users = {
        str(event.get("id")): event
        for event in materialized
        if event.get("event_type") == "user_message"
        and event.get("id")
        and str(event.get("id")) != str(current_user_event_id or "")
    }
    paired: list[AnsweredTurn] = []
    for reply in materialized:
        if reply.get("event_type") != "character_message":
            continue
        metadata = reply.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        user_id = str(metadata.get("responds_to") or "")
        user = users.get(user_id)
        if user is None:
            continue
        paired.append(
            AnsweredTurn(
                user_event_id=user_id,
                user_text=str(user.get("content") or ""),
                user_topic=str(user.get("topic")) if user.get("topic") else None,
                answer_event_id=str(reply.get("id") or ""),
                answer_text=str(reply.get("content") or ""),
                answer_topic=str(reply.get("topic")) if reply.get("topic") else None,
                left=dict(metadata.get("left", {})) if isinstance(metadata.get("left"), dict) else {},
                right=dict(metadata.get("right", {})) if isinstance(metadata.get("right"), dict) else {},
                prior_match=(
                    dict(metadata.get("history_match", {}))
                    if isinstance(metadata.get("history_match"), dict)
                    else dict(metadata.get("repeat_review", {}))
                    if isinstance(metadata.get("repeat_review"), dict)
                    else {}
                ),
            )
        )
    return paired[-max(1, limit):]


def candidate_signals(
    *,
    message: str,
    current_left: dict[str, Any],
    current_right: dict[str, Any],
    prior: AnsweredTurn,
    embedding_similarity: float | None = None,
    embedding_threshold: float = 0.80,
) -> list[MatchSignal]:
    """Collect independent evidence without deciding user intent."""

    signals: list[MatchSignal] = []
    if canonical_text(message) and canonical_text(message) == canonical_text(prior.user_text):
        signals.append(MatchSignal("exact_text", 1.0, "normalized user text is identical"))

    lexical = lexical_similarity(message, prior.user_text)
    if lexical >= 0.30:
        # Lexical overlap is deliberately weak; it can support a candidate but
        # should not dominate grounded or semantic evidence.
        signals.append(MatchSignal("lexical_overlap", min(0.62, 0.35 + lexical * 0.27), f"jaccard={lexical:.4f}"))

    current_facts = _compact_keys(current_left, "fact_refs")
    prior_facts = _compact_keys(prior.left, "fact_refs")
    shared_facts = sorted(current_facts & prior_facts)
    if shared_facts:
        signals.append(MatchSignal("shared_fact_reference", 0.90, ",".join(shared_facts[:4])))

    current_associations = _compact_keys(current_right, "association_keys")
    prior_associations = _compact_keys(prior.right, "association_keys")
    shared_associations = sorted(current_associations & prior_associations)
    if shared_associations:
        signals.append(MatchSignal("shared_association", 0.70, ",".join(shared_associations[:4])))

    embedding = _safe_score(embedding_similarity)
    if embedding is not None and embedding >= embedding_threshold:
        # Preserve the actual similarity while mapping it into a strong but not
        # absolute confidence contribution. Exact text remains uniquely decisive.
        strength = min(0.95, 0.72 + 0.23 * ((embedding - embedding_threshold) / max(1e-6, 1.0 - embedding_threshold)))
        signals.append(MatchSignal("embedding_similarity", strength, f"cosine={embedding:.4f}"))

    return signals


def _combined_score(signals: list[MatchSignal]) -> float:
    """Combine independent evidence without allowing duplicate weak signals to win.

    The strongest signal establishes the candidate. Additional independent signal
    families increase confidence with diminishing returns. This avoids the old
    pattern where several correlated lexical heuristics could accidentally amount
    to semantic certainty.
    """

    if not signals:
        return 0.0
    strongest = max(signal.strength for signal in signals)
    families = {signal.kind for signal in signals}
    bonus = min(0.08, 0.025 * max(0, len(families) - 1))
    return min(1.0, strongest + bonus)


def _subject_hint(prior: AnsweredTurn, signals: list[MatchSignal]) -> str | None:
    shared_fact = next((signal for signal in signals if signal.kind == "shared_fact_reference"), None)
    if shared_fact:
        first = shared_fact.detail.split(",", 1)[0]
        return f"fact:{first}"
    prior_root = str(prior.prior_match.get("root_user_event_id") or "").strip()
    if prior_root:
        return f"thread:{prior_root}"
    # Topic is retained only as an observational hint after a match has already
    # been established. It is never itself evidence that two turns are related.
    return prior.answer_topic or prior.user_topic


def match_history(
    *,
    message: str,
    current_left: dict[str, Any],
    current_right: dict[str, Any],
    prior_turns: Iterable[AnsweredTurn],
    embedding_matches: dict[str, float] | None = None,
    embedding_threshold: float = 0.80,
    minimum_score: float = 0.70,
) -> HistoryMatch | None:
    """Return the best evidenced historical candidate, if one clears policy.

    Recency is used only as the final tie-breaker. A recent weak lexical overlap
    cannot outrank an older turn sharing the same grounded fact or a strong
    embedding match.
    """

    embeddings = embedding_matches or {}
    candidates: list[tuple[float, int, AnsweredTurn, list[MatchSignal]]] = []
    materialized = list(prior_turns)
    for index, prior in enumerate(materialized):
        signals = candidate_signals(
            message=message,
            current_left=current_left,
            current_right=current_right,
            prior=prior,
            embedding_similarity=embeddings.get(prior.user_event_id),
            embedding_threshold=embedding_threshold,
        )
        score = _combined_score(signals)
        if score >= minimum_score:
            candidates.append((score, index, prior, signals))
    if not candidates:
        return None

    score, _, prior, signals = max(candidates, key=lambda item: (item[0], item[1]))
    root = str(prior.prior_match.get("root_user_event_id") or prior.user_event_id)
    return HistoryMatch(
        user_event_id=prior.user_event_id,
        answer_event_id=prior.answer_event_id,
        answer=prior.answer_text,
        score=score,
        signals=tuple(signals),
        root_user_event_id=root,
        subject_hint=_subject_hint(prior, signals),
    )


def history_report(match: HistoryMatch | None) -> dict[str, Any]:
    """Stable Executive-facing shape with no inferred motive or emotional state."""

    if match is None:
        return {
            "matched": False,
            "matched_event_id": None,
            "matched_answer_event_id": None,
            "matched_answer": None,
            "confidence": 0.0,
            "root_user_event_id": None,
            "subject_hint": None,
            "signals": [],
        }
    return match.as_dict()
