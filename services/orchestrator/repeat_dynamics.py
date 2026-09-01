"""Executive-facing repeat-response safeguards and durable pressure dynamics."""

from __future__ import annotations

from typing import Any

from services.common import CharacterDocument, RepeatDynamics
from services.orchestrator.cognitive_policy import CONTENT_TOKEN_RE, content_tokens


def question_signature(message: str) -> str:
    return " ".join(CONTENT_TOKEN_RE.findall(message.lower()))


def response_terms(text: str) -> set[str]:
    """Normalize limited common inflections for answer-echo prevention."""

    terms: set[str] = set()
    for token in content_tokens(text):
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ied"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if len(token) > 1:
            terms.add(token)
    return terms


def response_substantially_repeats_prior_answer(speech: str, prior_speech: str) -> bool:
    if question_signature(speech) == question_signature(prior_speech):
        return True
    speech_terms = response_terms(speech)
    prior_terms = response_terms(prior_speech)
    if len(speech_terms) < 3 or len(prior_terms) < 3:
        return False
    shared = len(speech_terms & prior_terms)
    return shared >= 3 and (shared / min(len(speech_terms), len(prior_terms))) >= 0.50


def response_substantially_repeats_recent_answers(speech: str, prior_speeches: list[str]) -> bool:
    return any(
        response_substantially_repeats_prior_answer(speech, prior_speech)
        for prior_speech in prior_speeches
        if prior_speech.strip()
    )


def repeat_intent_fallback(assessment: dict[str, Any], consecutive_repeats: int) -> str:
    """Offer a varied last-resort response after planned repeat reframing fails."""

    response_mode = str(assessment.get("response_mode") or "")
    primary = str(assessment.get("primary_hypothesis") or "")
    stage = max(consecutive_repeats - 2, 0) % 4
    if response_mode == "test_consistency" or "consisten" in primary:
        return [
            "If you are checking whether my answer changes when you ask again, it does not. Is there a circumstance you want to test?",
            "My answer is consistent. Are you trying to see whether a different situation would change it?",
            "I cannot tell whether you are testing my consistency or looking for more detail. Which is it?",
            "If consistency is the point, I have been clear. Tell me what condition you think might alter the answer.",
        ][stage]
    if response_mode == "check_understanding" or any(key in primary for key in ("understand", "clear")):
        return [
            "I may not have explained the point clearly. Do you want the principle, or an example of it in practice?",
            "Perhaps I have answered too broadly. Which word or part of the answer is unclear?",
            "We may be using the same words for different questions. What do you mean by it here?",
            "I do not want to keep guessing at the gap. Name the part you want me to unpack.",
        ][stage]
    if response_mode == "set_boundary":
        return [
            "I've answered the question directly. If you mean something more specific, say which part you want to examine.",
            "I can approach the subject another way, but repeating the same words does not tell me what is missing.",
            "If you are looking for a different answer, be direct about the distinction you want to explore.",
            "I have tried to meet the question as asked. Tell me the actual point you want to press.",
        ][stage]
    return [
        "Perhaps the broad answer is not the useful one. Are you asking what I value, or how that value guides a decision?",
        "It sounds as though the principle alone is not enough. Do you want to know how it guides a decision, or whether another concern can outrank it?",
        "We may be talking past each other. Are you asking about the value itself, the work it leads me to do, or a situation where it is tested?",
        "If the answer is still not reaching you, tell me what distinction you need me to make.",
    ][stage]


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def derive_repeat_dynamics(
    *,
    character: CharacterDocument,
    mutable_state: dict[str, Any],
    review: dict[str, Any],
    user_turn_count: int,
    escalation_decision: str = "hold",
) -> tuple[RepeatDynamics, dict[str, float], bool]:
    """Measure repeat pressure; only Executive choice may raise defensiveness."""

    raw_topics = mutable_state.get("topic_defensiveness", {})
    if not isinstance(raw_topics, dict):
        raw_topics = {}
    topic_defensiveness = {
        str(key): clamp(float(value))
        for key, value in raw_topics.items()
        if isinstance(value, (int, float))
    }
    subject_key = str(review["subject_key"])
    prior_defensiveness = topic_defensiveness.get(subject_key, 0.0)
    semantic_repeat = bool(review["semantic_repeat_candidate"])
    consecutive_repeats = int(review["consecutive_repeats"])
    if semantic_repeat:
        added_pressure = 0.24 + (0.04 * min(max(consecutive_repeats - 2, 0), 3))
        projected_defensiveness = clamp((prior_defensiveness * 0.96) + added_pressure)
    elif prior_defensiveness:
        projected_defensiveness = clamp(prior_defensiveness * 0.97)
    else:
        projected_defensiveness = 0.0
    if semantic_repeat and escalation_decision == "increase":
        subject_defensiveness = projected_defensiveness
    elif escalation_decision == "deescalate":
        subject_defensiveness = clamp(prior_defensiveness * 0.70)
    elif semantic_repeat:
        subject_defensiveness = prior_defensiveness
    else:
        subject_defensiveness = projected_defensiveness
    updated_topics = dict(topic_defensiveness)
    changed = abs(subject_defensiveness - prior_defensiveness) >= 0.001
    if changed:
        updated_topics[subject_key] = round(subject_defensiveness, 4)
    trait_patience = float(character.traits.get("patient", 0.5))
    trait_irritability = float(character.traits.get("irritable", 0.5))
    baseline_patience = clamp(0.72 + (0.20 * trait_patience) - (0.10 * trait_irritability))
    conversation_drain = 0.025 * max(user_turn_count - 1, 0)
    repetition_drain = 0.115 * max(consecutive_repeats - 1, 0)
    conversation_patience = clamp(baseline_patience - conversation_drain - repetition_drain)
    intersection_pressure = clamp(subject_defensiveness * (1.0 - conversation_patience))
    suggested_pressure = clamp(projected_defensiveness * (1.0 - conversation_patience))

    def posture_for(pressure: float) -> str:
        if pressure >= 0.36:
            return "defensive"
        if pressure >= 0.16:
            return "confused"
        return "reclarify" if semantic_repeat else "normal"

    posture = posture_for(intersection_pressure)
    suggested_posture = posture_for(suggested_pressure)
    escalation_recommendation = "increase" if semantic_repeat and consecutive_repeats >= 2 else "hold"
    return (
        RepeatDynamics(
            conversation_patience=round(conversation_patience, 4),
            subject_defensiveness=round(subject_defensiveness, 4),
            intersection_pressure=round(intersection_pressure, 4),
            response_posture=posture,
            suggested_posture=suggested_posture,
            escalation_recommendation=escalation_recommendation,
            semantic_repeat=semantic_repeat,
            consecutive_repeats=consecutive_repeats,
            subject_key=subject_key,
            review_confidence=round(float(review["confidence"]), 4),
        ),
        updated_topics,
        changed,
    )
