from services.orchestrator.history_matching import (
    answered_turns,
    candidate_signals,
    canonical_text,
    history_report,
    lexical_similarity,
    match_history,
)


def _events():
    return [
        {
            "id": "u1",
            "event_type": "user_message",
            "content": "Where were you born?",
            "topic": "legacy.birthplace",
            "metadata": {},
        },
        {
            "id": "a1",
            "event_type": "character_message",
            "content": "I was born in Greyhaven.",
            "topic": "legacy.birthplace",
            "metadata": {
                "responds_to": "u1",
                "left": {"fact_refs": ["identity.birthplace"], "action": "answer"},
                "right": {"association_keys": ["home.origin"], "action": "inform"},
            },
        },
        {
            "id": "u2",
            "event_type": "user_message",
            "content": "What work do you do?",
            "topic": "legacy.occupation",
            "metadata": {},
        },
        {
            "id": "a2",
            "event_type": "character_message",
            "content": "I manage the harbor.",
            "topic": "legacy.occupation",
            "metadata": {
                "responds_to": "u2",
                "left": {"fact_refs": ["identity.occupation"], "action": "answer"},
                "right": {"association_keys": ["harbor.work"], "action": "inform"},
            },
        },
    ]


def test_answered_turns_excludes_unanswered_inputs_and_current_event():
    events = _events() + [
        {"id": "u3", "event_type": "user_message", "content": "Unanswered", "topic": "x", "metadata": {}},
        {"id": "u4", "event_type": "user_message", "content": "Current", "topic": "x", "metadata": {}},
    ]
    paired = answered_turns(events, current_user_event_id="u4")
    assert [turn.user_event_id for turn in paired] == ["u1", "u2"]


def test_topic_equality_is_not_evidence_of_historical_relatedness():
    prior = answered_turns(_events())[0]
    signals = candidate_signals(
        message="Completely unrelated words",
        current_left={"fact_refs": []},
        current_right={"association_keys": []},
        prior=prior,
    )
    assert signals == []


def test_exact_text_is_decisive_without_special_case_topics():
    prior = answered_turns(_events())[0]
    match = match_history(
        message="Where were you born?",
        current_left={},
        current_right={},
        prior_turns=[prior],
    )
    assert match is not None
    assert match.user_event_id == "u1"
    assert match.score == 1.0
    assert match.signals[0].kind == "exact_text"


def test_shared_grounded_fact_can_match_semantically_different_wording():
    prior = answered_turns(_events())[0]
    match = match_history(
        message="Which city counts as your place of origin?",
        current_left={"fact_refs": ["identity.birthplace"]},
        current_right={},
        prior_turns=[prior],
    )
    assert match is not None
    assert match.user_event_id == "u1"
    assert any(signal.kind == "shared_fact_reference" for signal in match.signals)
    assert match.subject_hint == "fact:identity.birthplace"


def test_embedding_can_retrieve_a_nonlexical_prior_turn():
    prior = answered_turns(_events())[0]
    match = match_history(
        message="Which city was your early home?",
        current_left={},
        current_right={},
        prior_turns=[prior],
        embedding_matches={"u1": 0.92},
        embedding_threshold=0.80,
    )
    assert match is not None
    assert any(signal.kind == "embedding_similarity" for signal in match.signals)


def test_weak_lexical_overlap_does_not_create_a_repeat_by_accumulation():
    prior = answered_turns(_events())[1]
    assert lexical_similarity("Do you work near ships?", prior.user_text) > 0
    match = match_history(
        message="Do you work near ships?",
        current_left={},
        current_right={},
        prior_turns=[prior],
    )
    assert match is None


def test_best_candidate_prefers_grounded_evidence_over_recent_weak_overlap():
    prior = answered_turns(_events())
    match = match_history(
        message="Tell me about your origin work",
        current_left={"fact_refs": ["identity.birthplace"]},
        current_right={},
        prior_turns=prior,
    )
    assert match is not None
    assert match.user_event_id == "u1"


def test_prior_match_root_is_propagated_as_conversation_thread_identity():
    events = _events()
    events[1]["metadata"]["history_match"] = {
        "matched": True,
        "root_user_event_id": "u0",
    }
    prior = answered_turns(events)[0]
    match = match_history(
        message="Where were you born?",
        current_left={},
        current_right={},
        prior_turns=[prior],
    )
    assert match is not None
    assert match.root_user_event_id == "u0"


def test_report_contains_evidence_not_inferred_user_motive():
    prior = answered_turns(_events())[0]
    report = history_report(match_history(
        message="Which city was your early home?",
        current_left={},
        current_right={},
        prior_turns=[prior],
        embedding_matches={"u1": 0.91},
    ))
    assert report["matched"] is True
    assert report["signals"]
    assert "primary_hypothesis" not in report
    assert "repeat" not in report


def test_canonical_text_only_normalizes_surface_form():
    assert canonical_text("Where, WERE you born?!") == "where were you born"
