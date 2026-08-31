from services.orchestrator.history_adapter import (
    evidence_history_review,
    neutral_interaction_classification,
    no_repeat_lobe_reuse,
)


def _answered(user_id, answer_id, user_text, *, fact_ref=None, root=None):
    left = {"action": "answer", "fact_refs": [fact_ref] if fact_ref else []}
    metadata = {"responds_to": user_id, "left": left, "right": {"association_keys": []}}
    if root:
        metadata["repeat_review"] = {"root_user_event_id": root}
    return [
        {"id": user_id, "event_type": "user_message", "actor": "user", "content": user_text, "topic": "ignored"},
        {"id": answer_id, "event_type": "character_message", "actor": "character", "content": "answer", "topic": "ignored", "metadata": metadata},
    ]


def test_pre_cognition_classification_cannot_declare_a_repeat_from_topic_history():
    classification = neutral_interaction_classification(
        interaction_type="repeated_question",
        topic="self.birthplace",
        prior_answer="Northbridge",
        times_asked=8,
        related_event_ids=["u1"],
    )
    assert classification.interaction_type == "new_subject"
    assert classification.times_asked == 1
    assert classification.prior_answer is None
    assert classification.related_event_ids == []


def test_adapter_does_not_use_equal_topics_or_prior_times_as_repeat_evidence():
    events = _answered("u1", "a1", "Tell me about cargo")
    review = evidence_history_review(
        message="How is the weather?",
        topic="ignored",
        current_event_id="u2",
        session_events=events,
        left_result={"fact_refs": []},
        right_result={"association_keys": []},
        prior_times=99,
    )
    assert review["semantic_repeat_candidate"] is False
    assert review["confidence"] == 0.0


def test_adapter_propagates_one_root_across_a_contiguous_history_thread():
    events = []
    events += _answered("u1", "a1", "Where were you born?", fact_ref="identity.birthplace")
    events += _answered(
        "u2",
        "a2",
        "Which city was that?",
        fact_ref="identity.birthplace",
        root="u1",
    )
    review = evidence_history_review(
        message="And that is still your place of origin?",
        topic="irrelevant",
        current_event_id="u3",
        session_events=events,
        left_result={"fact_refs": ["identity.birthplace"]},
        right_result={"association_keys": []},
        prior_times=0,
    )
    assert review["semantic_repeat_candidate"] is True
    assert review["root_user_event_id"] == "u1"
    assert review["consecutive_repeats"] == 3
    assert review["subject_key"] == "fact:identity.birthplace"


def test_thread_depth_resets_after_unrelated_answered_turn():
    events = []
    events += _answered("u1", "a1", "Where were you born?", fact_ref="identity.birthplace")
    events += _answered("u2", "a2", "What is your job?", fact_ref="identity.occupation")
    review = evidence_history_review(
        message="Which city were you born in?",
        topic="irrelevant",
        current_event_id="u3",
        session_events=events,
        left_result={"fact_refs": ["identity.birthplace"]},
        right_result={"association_keys": []},
        prior_times=0,
    )
    assert review["semantic_repeat_candidate"] is True
    assert review["root_user_event_id"] == "u1"
    assert review["consecutive_repeats"] == 1


def test_exact_repeat_keeps_topic_only_as_output_label_not_match_evidence():
    events = _answered("u1", "a1", "Where were you born?")
    review = evidence_history_review(
        message="Where were you born?",
        topic="surface.same-question",
        current_event_id="u2",
        session_events=events,
        left_result={},
        right_result={},
        prior_times=0,
    )
    assert review["semantic_repeat_candidate"] is True
    assert review["subject_key"] == "surface.same-question"
    assert "exact_text" in review["reason"]


def test_exact_repeat_no_longer_reuses_stale_lobe_artifacts():
    assert no_repeat_lobe_reuse(message="same", topic="anything", session_events=[]) is None
