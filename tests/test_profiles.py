from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

from services.common import CharacterDocument, EventRecord, MemoryRecord, MutationOperation, MutationProposal
from services.memory import app as memory


@pytest.fixture
def isolated_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source_dir = tmp_path / "characters"
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "cognition.db")
    monkeypatch.setattr(memory, "CHARACTER_DIR", source_dir)
    memory.init_db()
    return source_dir


def profile() -> CharacterDocument:
    return CharacterDocument.model_validate(
        {
            "id": "profile_test",
            "identity": {
                "name": "Profile Test",
                "age": 29,
                "occupation": "Archivist",
                "birthplace": "Larkspur",
                "faction": "Public Archive",
                "siblings": ["Mira Test"],
            },
            "traits": {"patient": 0.7},
            "values": ["accuracy"],
            "initial_goals": ["preserve_records"],
            "mutable_state": {"mood": {"curiosity": 0.6}},
            "beliefs": {"records_matter": True},
            "biography": "An archivist who protects public records.",
        }
    )


def test_profile_create_and_edit_persist_canonical_yaml(isolated_profiles: Path):
    created = memory.create_profile(profile())

    source_path = isolated_profiles / "profile_test.yaml"
    assert source_path.exists()
    assert created["source"]["identity"]["name"] == "Profile Test"
    assert created["runtime"]["mutable_state"]["mood"]["curiosity"] == 0.6
    assert "Profile Test" in source_path.read_text(encoding="utf-8")

    changed = profile().model_copy(
        update={
            "identity": {**profile().identity, "occupation": "Chief Archivist"},
            "biography": "Now leads the public archive.",
        }
    )
    updated = memory.update_profile("profile_test", changed)

    assert updated["source"]["identity"]["occupation"] == "Chief Archivist"
    assert "Chief Archivist" in source_path.read_text(encoding="utf-8")
    # Editing a primer does not erase the separate, accumulated runtime state.
    assert updated["runtime"]["mutable_state"]["mood"]["curiosity"] == 0.6


def test_profile_id_is_stable_and_safe(isolated_profiles: Path):
    memory.create_profile(profile())
    renamed = profile().model_copy(update={"id": "renamed_profile"})

    with pytest.raises(HTTPException, match="cannot be renamed"):
        memory.update_profile("profile_test", renamed)
    with pytest.raises(HTTPException, match="Character IDs"):
        memory.profile_path("../unsafe")


def test_profile_update_keeps_source_and_runtime_aligned_when_db_write_fails(
    isolated_profiles: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    memory.create_profile(profile())
    source_path = isolated_profiles / "profile_test.yaml"
    original_source = source_path.read_text(encoding="utf-8")

    def fail_upsert(*_args, **_kwargs):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(memory, "upsert_character", fail_upsert)
    updated = profile().model_copy(update={"biography": "This must not be committed."})
    with pytest.raises(RuntimeError, match="simulated database failure"):
        memory.update_profile("profile_test", updated)

    assert source_path.read_text(encoding="utf-8") == original_source
    assert memory.get_character("profile_test")["character"]["biography"] == profile().biography


def test_memory_provenance_cannot_reference_another_or_unknown_event(isolated_profiles: Path):
    memory.create_profile(profile())
    with pytest.raises(HTTPException, match="provenance"):
        memory.add_memory(MemoryRecord(
            character_id="profile_test",
            kind="derived_conclusion",
            content="This should be rejected.",
            source_event_ids=["evt_not_recorded"],
        ))


def test_turn_commit_rolls_back_answer_when_any_derived_write_is_invalid(isolated_profiles: Path):
    memory.create_profile(profile())
    session = memory.create_session(memory.SessionCreate(character_id="profile_test"))
    user_event = memory.add_event(EventRecord(
        character_id="profile_test",
        session_id=session["id"],
        event_type="user_message",
        actor="user",
        content="What conclusion have you drawn?",
        topic="topic.conclusions",
    ))
    turn = memory.TurnCommit(
        character_event=EventRecord(
            id="evt_atomic_reply",
            character_id="profile_test",
            session_id=session["id"],
            event_type="character_message",
            actor="character",
            content="The records need care.",
            topic="topic.conclusions",
            metadata={"responds_to": user_event.id},
        ),
        memories=[MemoryRecord(
            character_id="profile_test",
            kind="derived_conclusion",
            content="This bad source must abort every write.",
            source_event_ids=["evt_not_recorded"],
        )],
    )

    with pytest.raises(HTTPException, match="provenance"):
        memory.commit_turn(session["id"], turn)

    events = memory.session_events(session["id"], limit=20)
    assert [event["event_type"] for event in events] == ["user_message"]
    assert memory.get_memories("profile_test", limit=20) == []


def test_full_yaml_snapshot_round_trip_preserves_runtime_conclusions(isolated_profiles: Path):
    memory.create_profile(profile())
    session = memory.create_session(memory.SessionCreate(character_id="profile_test"))
    user_event = memory.add_event(EventRecord(
        character_id="profile_test",
        session_id=session["id"],
        event_type="user_message",
        actor="user",
        content="What conclusion have you drawn?",
        topic="topic.conclusions",
    ))
    memory.add_event(EventRecord(
        character_id="profile_test",
        session_id=session["id"],
        event_type="character_message",
        actor="character",
        content="Records deserve careful handling.",
        topic="topic.conclusions",
        metadata={"responds_to": user_event.id},
    ))
    memory.add_memory(MemoryRecord(
        character_id="profile_test",
        kind="derived_conclusion",
        topic="topic.conclusions",
        content="Careful handling protects public trust.",
        epistemic_type="inference",
        confidence=0.78,
        salience=0.7,
        source_event_ids=[user_event.id or ""],
    ))
    memory.apply_mutations("profile_test", memory.MutationBatch(proposals=[MutationProposal(
        operation=MutationOperation.SET_MUTABLE_STATE,
        target="topic_defensiveness",
        value={"topic.conclusions": 0.24},
        evidence=[user_event.id or ""],
    )]))

    snapshot = memory.character_snapshot("profile_test")
    exported = memory.export_profile("profile_test")
    assert memory.SNAPSHOT_FORMAT.encode() in exported.body
    assert b"derived_conclusion" in exported.body

    # Change the live runtime after export. Restoring the snapshot should discard
    # this later state and recover the source, events, memories, and conclusions.
    memory.add_memory(MemoryRecord(
        character_id="profile_test",
        kind="later_change",
        content="This was created after export.",
        source_event_ids=[],
    ))
    diff = memory.diff_profile(
        "profile_test",
        memory.ProfileDiffRequest(yaml=yaml.safe_dump(snapshot, sort_keys=False)),
    )
    assert diff["changed"] is True
    assert any(
        change["kind"] == "added"
        and isinstance(change.get("after"), dict)
        and change["after"].get("kind") == "later_change"
        for change in diff["changes"]
    )
    memory.import_profile(memory.ProfileImportRequest(yaml=yaml.safe_dump(snapshot, sort_keys=False)))

    restored = memory.character_snapshot("profile_test")
    assert restored["source"] == snapshot["source"]
    assert restored["runtime"]["state"]["mutable_state"] == snapshot["runtime"]["state"]["mutable_state"]
    assert restored["runtime"]["events"] == snapshot["runtime"]["events"]
    assert restored["runtime"]["memories"] == snapshot["runtime"]["memories"]
    assert all(item["kind"] != "later_change" for item in restored["runtime"]["memories"])


def test_interaction_history_ignores_unanswered_user_turns(isolated_profiles: Path):
    memory.create_profile(profile())
    unanswered = memory.add_event(EventRecord(
        character_id="profile_test",
        session_id=None,
        event_type="user_message",
        actor="user",
        content="Where were you born?",
        topic="self.birthplace",
    ))
    answered = memory.add_event(EventRecord(
        character_id="profile_test",
        session_id=None,
        event_type="user_message",
        actor="user",
        content="What is your name?",
        topic="self.birthplace",
    ))
    reply = memory.add_event(EventRecord(
        character_id="profile_test",
        session_id=None,
        event_type="character_message",
        actor="character",
        content="Profile Test.",
        topic="self.birthplace",
        metadata={"responds_to": answered.id},
    ))

    history = memory.interaction_history("profile_test", "self.birthplace", limit=20)

    assert history["times_asked"] == 1
    assert history["prior_answer"] == "Profile Test."
    assert [event["id"] for event in history["events"]] == [answered.id, reply.id]
    assert unanswered.id not in {event["id"] for event in history["events"]}
