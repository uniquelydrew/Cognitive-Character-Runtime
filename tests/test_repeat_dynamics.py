from services.common import CharacterDocument
from services.orchestrator.app import apply_repeat_posture, derive_repeat_dynamics, executive_repeat_review


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


def test_repeat_dynamics_preserve_subject_pressure_and_intersect_with_patience():
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

    first_repeat, topics, changed = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state=state,
        review=repeat_review,
        user_turn_count=2,
    )
    assert changed is True
    assert first_repeat.response_posture == "reclarify"
    assert first_repeat.subject_defensiveness > 0

    second_repeat, topics, _ = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state={"topic_defensiveness": topics},
        review={**repeat_review, "consecutive_repeats": 3},
        user_turn_count=3,
    )
    assert second_repeat.response_posture == "confused"

    third_repeat, topics, _ = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state={"topic_defensiveness": topics},
        review={**repeat_review, "consecutive_repeats": 4},
        user_turn_count=4,
    )
    assert third_repeat.response_posture == "defensive"
    assert third_repeat.conversation_patience < first_repeat.conversation_patience

    new_subject, preserved_topics, _ = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state={"topic_defensiveness": topics},
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


def test_repeat_posture_guard_makes_escalation_visible():
    confused, _, _ = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state={"topic_defensiveness": {"topic.cargo": 0.24}},
        review={
            "semantic_repeat_candidate": True,
            "subject_key": "topic.cargo",
            "consecutive_repeats": 3,
            "confidence": 0.8,
        },
        user_turn_count=3,
    )
    defensive, _, _ = derive_repeat_dynamics(
        character=CHARACTER,
        mutable_state={"topic_defensiveness": {"topic.cargo": confused.subject_defensiveness}},
        review={
            "semantic_repeat_candidate": True,
            "subject_key": "topic.cargo",
            "consecutive_repeats": 4,
            "confidence": 0.8,
        },
        user_turn_count=4,
    )

    assert "confused" in apply_repeat_posture("Greyhaven.", confused).lower()
    assert "already answered" in apply_repeat_posture("Greyhaven.", defensive).lower()
