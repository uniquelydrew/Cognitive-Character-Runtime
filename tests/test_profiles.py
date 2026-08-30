from pathlib import Path

import pytest
from fastapi import HTTPException

from services.common import CharacterDocument, EventRecord
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
