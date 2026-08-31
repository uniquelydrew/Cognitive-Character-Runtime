from services.common import CharacterDocument
from services.orchestrator.app import (
    bounded_lobe_transcript,
    derive_repeat_dynamics,
    executive_repeat_review,
    immediate_repeat_lobe_reuse,
    repeat_intent_fallback,
    resolve_cognitive_priorities,
    response_substantially_repeats_prior_answer,
    response_substantially_repeats_recent_answers,
    weighted_arbitration_plan,
)


CHARACTER = CharacterDocument.model_validate(
    {
        "id": "test_character",
        "identity": {"name": "Test Character", "occupation": "Harbormaster"},
        "traits": {"patient": 0.62, "irritable": 0.28},
    }
)


def test_executive_review_detects_rephrased_repeat_from_lobe_agreement():
    review = executive_repeat_review(
        message="Where did the shipment end up?",
        topic="topic.shipment",
        current_event_id="evt_current",
        session_events=[
            {
                "id": "evt_previous",
                "event_type": "user_message",
                "content": "Tell me about the lost crates.",
                "topic": "topic.cargo",
                "metadata": {},
            },
            {
                "id": "evt_reply",
                "event_type": "character_message",
                "content": "The cargo is still unaccounted for.",
                "topic": "topic.cargo",
                "metadata": {
                    "responds_to": "evt_previous",
                    "left": {"topic": "topic.missing_cargo"},
                    "right": {"associations": ["missing cargo"]},
                },
            },
            {
                "id": "evt_current",
                "event_type": "user_message",
                "content": "Where did the shipment end up?",
                "topic": "topic.shipment",
                "metadata": {},
            },
        ],
        left_result={"topic": "topic.missing_cargo"},
        right_result={"associations": ["lost shipment"]},
        prior_times=0,
    )

    assert review["semantic_repeat_candidate"] is True
    assert review["matched_event_id"] == "evt_previous"
    assert review["subject_key"] == "topic.cargo"
    assert review["reason"] == "left analyses selected the same subject"
    assert review["confidence"] == 0.84


def test_executive_review_uses_subject_anchors_when_lobe_topic_labels_differ():
    review = executive_repeat_review(
        message="What town did he grow up in?",
        topic="topic.grow.town",
        current_event_id="evt_current",
        session_events=[
            {
                "id": "evt_previous",
                "event_type": "user_message",
                "content": "Where was he born?",
                "topic": "self.birthplace",
                "metadata": {},
            },
            {
                "id": "evt_reply",
                "event_type": "character_message",
                "content": "He was born in Greyhaven.",
                "topic": "self.birthplace",
                "metadata": {
                    "responds_to": "evt_previous",
                    "left": {
                        "topic": "birthplace",
                        "observations": ["Greyhaven is the birthplace."],
                        "recommended_strategy": "describe Greyhaven",
                    },
                },
            },
            {
                "id": "evt_current",
                "event_type": "user_message",
                "content": "What town did he grow up in?",
                "topic": "topic.grow.town",
                "metadata": {},
            },
        ],
        left_result={
            "topic": "grow.town",
            "observations": ["Greyhaven is where he grew up."],
            "recommended_strategy": "explain his birthplace",
        },
        right_result={"associations": []},
        prior_times=0,
    )

    assert review["semantic_repeat_candidate"] is True
    assert review["matched_event_id"] == "evt_previous"
    assert review["reason"] == "left analyses share subject-specific fact anchors"
    assert review["confidence"] == 0.76


def test_executive_review_uses_compact_fact_reference_without_losing_dotted_key():
    review = executive_repeat_review(
        message="Could you explain that place again?",
        topic="topic.explain.place",
        current_event_id="evt_current",
        session_events=[
            {
                "id": "evt_previous",
                "event_type": "user_message",
                "content": "Where were you born?",
                "topic": "self.birthplace",
                "metadata": {},
            },
            {
                "id": "evt_reply",
                "event_type": "character_message",
                "content": "I was born in Greyhaven.",
                "topic": "self.birthplace",
                "metadata": {
                    "responds_to": "evt_previous",
                    "left": {
                        "topic": "birthplace",
                        "fact_refs": ["identity.birthplace"],
                        "constraints": ["preserve_core"],
                        "action": "answer",
                    },
                },
            },
            {
                "id": "evt_current",
                "event_type": "user_message",
                "content": "Could you explain that place again?",
                "topic": "topic.explain.place",
                "metadata": {},
            },
        ],
        left_result={
            "topic": "place.explanation",
            "fact_refs": ["identity.birthplace"],
            "constraints": ["preserve_core"],
            "action": "reclarify",
        },
        right_result={"association_keys": []},
        prior_times=0,
    )

    assert review["semantic_repeat_candidate"] is True
    assert review["reason"] == "left analyses reference the same established fact"
    assert review["confidence"] == 0.78


def test_executive_review_keeps_the_prior_semantic_subject_key():
    review = executive_repeat_review(
        message="Could you say that another way?",
        topic="topic.say.way",
        current_event_id="evt_current",
        session_events=[
            {
                "id": "evt_previous",
                "event_type": "user_message",
                "content": "Remind me which town you grew up in.",
                "topic": "topic.grew.remind.town",
                "metadata": {},
            },
            {
                "id": "evt_reply",
                "event_type": "character_message",
                "content": "Greyhaven.",
                "topic": "stable topic identifier",
                "metadata": {
                    "responds_to": "evt_previous",
                    "left": {"topic": "grew.up.town"},
                    "repeat_review": {"subject_key": "self.birthplace"},
                },
            },
            {
                "id": "evt_current",
                "event_type": "user_message",
                "content": "Could you say that another way?",
                "topic": "topic.say.way",
                "metadata": {},
            },
        ],
        left_result={"topic": "grew.up.town"},
        right_result={"associations": []},
        prior_times=0,
    )

    assert review["semantic_repeat_candidate"] is True
    assert review["subject_key"] == "self.birthplace"


def test_executive_review_ignores_a_prior_user_turn_without_a_reply():
    review = executive_repeat_review(
        message="Where was it again?",
        topic="self.birthplace",
        current_event_id="evt_current",
        session_events=[
            {
                "id": "evt_unanswered",
                "event_type": "user_message",
                "content": "Where were you born?",
                "topic": "self.birthplace",
                "metadata": {},
            },
            {
                "id": "evt_current",
                "event_type": "user_message",
                "content": "Where was it again?",
                "topic": "self.birthplace",
                "metadata": {},
            },
        ],
        left_result={"topic": "self.birthplace"},
        right_result={"association_keys": []},
        prior_times=0,
    )

    assert review["semantic_repeat_candidate"] is False


def test_embedding_similarity_identifies_a_nonlexical_rephrased_repeat():
    review = executive_repeat_review(
        message="Which city was her early home?",
        topic="topic.early.home",
        current_event_id="evt_current",
        session_events=[
            {
                "id": "evt_previous",
                "event_type": "user_message",
                "content": "Where was she born?",
                "topic": "topic.birth.question",
                "metadata": {},
            },
            {
                "id": "evt_reply",
                "event_type": "character_message",
                "content": "She was born in Greyhaven.",
                "topic": "topic.birth.question",
                "metadata": {"responds_to": "evt_previous", "left": {}, "right": {}},
            },
            {
                "id": "evt_current",
                "event_type": "user_message",
                "content": "Which city was her early home?",
                "topic": "topic.early.home",
                "metadata": {},
            },
        ],
        left_result={"topic": "topic.early.home"},
        right_result={"association_keys": []},
        prior_times=0,
        embedding_matches={"evt_previous": 0.91},
        embedding_threshold=0.80,
    )

    assert review["semantic_repeat_candidate"] is True
    assert review["reason"] == "embedding similarity"
    assert review["embedding_similarity"] == 0.91


def test_bounded_lobe_transcript_keeps_recent_raw_turns_under_hard_limits(monkeypatch):
    monkeypatch.setattr("services.orchestrator.app.MAX_LOBE_TRANSCRIPT_EVENTS", 2)
    monkeypatch.setattr("services.orchestrator.app.MAX_LOBE_TRANSCRIPT_CHARS", 15)
    monkeypatch.setattr("services.orchestrator.app.MAX_LOBE_TRANSCRIPT_EVENT_CHARS", 12)
    transcript = bounded_lobe_transcript([
        {"id": "old", "event_type": "user_message", "actor": "user", "content": "old context", "topic": "old"},
        {"id": "question", "event_type": "user_message", "actor": "user", "content": "recent question", "topic": "new"},
        {"id": "reply", "event_type": "character_message", "actor": "character", "content": "recent answer", "topic": "new"},
    ])

    assert [event["event_id"] for event in transcript] == ["question", "reply"]
    assert sum(len(event["content"]) for event in transcript) <= 15
    assert transcript[-1]["content_truncated"] is True


def test_profile_priorities_drive_a_bounded_arbitration_plan():
    character = CHARACTER.model_copy(update={"cognition": {"left_weight": 3, "right_weight": 1}})
    priorities = resolve_cognitive_priorities(character)
    plan = weighted_arbitration_plan(
        priorities,
        {"action": "answer", "fact_refs": ["identity.birthplace"], "constraints": ["preserve_core"]},
        {"action": "reassure", "tone": "warm", "risk": "low", "association_keys": ["home"]},
    )

    assert priorities["left_weight"] == 0.75
    assert priorities["right_weight"] == 0.25
    assert priorities["primary_role"] == "left"
    assert plan["primary_packet"] == plan["left"]
    assert "left.constraints are binding" in plan["binding_rules"][0]


def test_immediate_answered_exact_repeat_reuses_prior_lobe_artifacts():
    reuse = immediate_repeat_lobe_reuse(
        message="Where were you born?",
        topic="self.birthplace",
        session_events=[
            {
                "id": "evt_question",
                "event_type": "user_message",
                "content": "Where were you born?",
                "topic": "self.birthplace",
                "metadata": {},
            },
            {
                "id": "evt_reply",
                "event_type": "character_message",
                "content": "Northbridge.",
                "topic": "self.birthplace",
                "metadata": {
                    "responds_to": "evt_question",
                    "left": {"topic": "self.birthplace", "action": "answer"},
                    "right": {"action": "inform", "tone": "warm"},
                },
            },
        ],
    )

    assert reuse is not None
    assert reuse["reason"] == "immediate_answered_exact_repeat"
    assert reuse["left"]["action"] == "answer"
    assert reuse["prior_speech"] == "Northbridge."


def test_repeat_reframe_guard_catches_a_close_paraphrase_but_allows_a_new_angle():
    prior = "As a harbormaster, my duty is to protect the port and its workers."

    assert response_substantially_repeats_prior_answer(
        "Protecting the port and its workers is my top priority as harbormaster.",
        prior,
    )
    assert not response_substantially_repeats_prior_answer(
        "The dawn watch is where I make that responsibility practical. Is there a particular decision you mean?",
        prior,
    )
    assert response_substantially_repeats_recent_answers(
        "The sense of duty and responsibility that comes with being a harbormaster is what truly matters to me.",
        [
            "I understand that my duty is my top priority, but what I value most is the sense of duty and responsibility that comes with it.",
            "We may be talking past each other. What distinction are you looking for?",
        ],
    )


def test_repeat_intent_fallback_varies_with_the_executive_response_mode():
    new_angle = repeat_intent_fallback({"response_mode": "new_angle"}, 2)
    consistency = repeat_intent_fallback({"response_mode": "test_consistency"}, 2)
    unknown = repeat_intent_fallback({"response_mode": "invite_specificity"}, 2)
    next_new_angle = repeat_intent_fallback({"response_mode": "new_angle"}, 3)

    assert new_angle != consistency != unknown != next_new_angle
    assert "broad answer" in unknown
    assert "circumstance" in consistency


def test_rephrased_or_unanswered_turns_do_not_bypass_lobe_reasoning():
    events = [
        {
            "id": "evt_question",
            "event_type": "user_message",
            "content": "Where were you born?",
            "topic": "self.birthplace",
            "metadata": {},
        },
        {
            "id": "evt_reply",
            "event_type": "character_message",
            "content": "Northbridge.",
            "topic": "self.birthplace",
            "metadata": {
                "responds_to": "evt_question",
                "left": {"topic": "self.birthplace"},
                "right": {"action": "inform"},
            },
        },
    ]

    assert immediate_repeat_lobe_reuse(
        message="What is your hometown?", topic="self.birthplace", session_events=events
    ) is None
    assert immediate_repeat_lobe_reuse(
        message="Where were you born?", topic="self.birthplace", session_events=events[:1]
    ) is None


def test_repeat_dynamics_preserve_subject_pressure_and_require_executive_escalation():
    state: dict[str, object] = {}
    baseline, _, _ = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state=state,
        review={
            "semantic_repeat_candidate": False,
            "subject_key": "topic.weather",
            "consecutive_repeats": 1,
            "confidence": 0.0,
        },
        user_turn_count=1,
    )
    repeat_review = {
        "semantic_repeat_candidate": True,
        "subject_key": "topic.missing_cargo",
        "consecutive_repeats": 2,
        "confidence": 0.84,
    }

    held_repeat, topics, changed = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state=state,
        review=repeat_review,
        user_turn_count=2,
    )
    assert changed is False
    assert held_repeat.response_posture == "reclarify"
    assert held_repeat.suggested_posture == "reclarify"
    assert held_repeat.escalation_recommendation == "increase"
    assert held_repeat.subject_defensiveness == 0

    increased_repeat, topics, changed = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state=state,
        review=repeat_review,
        user_turn_count=2,
        escalation_decision="increase",
    )
    assert changed is True
    assert increased_repeat.subject_defensiveness > 0
    assert increased_repeat.response_posture == "reclarify"

    escalated_again, topics, _ = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state={"topic_defensiveness": topics},
        review={**repeat_review, "consecutive_repeats": 3},
        user_turn_count=3,
        escalation_decision="increase",
    )
    assert escalated_again.response_posture == "confused"

    held_after_increase, preserved_topics, changed = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state={"topic_defensiveness": topics},
        review={**repeat_review, "consecutive_repeats": 4},
        user_turn_count=4,
    )
    assert changed is False
    assert held_after_increase.subject_defensiveness == topics["topic.missing_cargo"]
    assert held_after_increase.suggested_posture == "defensive"
    assert held_after_increase.conversation_patience < held_repeat.conversation_patience

    new_subject, preserved_topics, _ = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state={"topic_defensiveness": preserved_topics},
        review={
            "semantic_repeat_candidate": False,
            "subject_key": "topic.weather",
            "consecutive_repeats": 1,
            "confidence": 0.0,
        },
        user_turn_count=5,
    )
    assert new_subject.subject_defensiveness == 0
    # Changing subject clears only the active repeat penalty; it does not restore
    # the session to its original patience baseline.
    assert new_subject.conversation_patience < baseline.conversation_patience
    assert preserved_topics["topic.missing_cargo"] == topics["topic.missing_cargo"]
