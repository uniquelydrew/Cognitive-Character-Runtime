from pathlib import Path

import yaml

from services.common import CharacterDocument
from services.memory import app as memory


def _character(character_id: str, *, birthplace: str, occupation: str) -> CharacterDocument:
    return CharacterDocument.model_validate({
        "id": character_id,
        "identity": {
            "name": character_id,
            "birthplace": birthplace,
            "occupation": occupation,
            "faction": "Port Authority" if occupation == "Harbormaster" else "Visitor",
        },
        "biography": "Test character.",
    })


def test_general_knowledge_is_label_indexed_and_character_scoped(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "cognition.db")
    monkeypatch.setattr(memory, "CHARACTER_DIR", tmp_path / "characters")
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(memory, "KNOWLEDGE_DIR", knowledge_dir)
    (knowledge_dir / "catalog.yaml").write_text(yaml.safe_dump({
        "classifications": [
            {"id": "access.public"},
            {"id": "place.northbridge", "aliases": ["northbridge"]},
            {"id": "community.port_staff", "aliases": ["port staff"]},
            {"id": "role.harbormaster", "parents": ["community.port_staff"], "aliases": ["harbormaster"]},
            {"id": "domain.records", "aliases": ["manifest", "manifests", "records"]},
        ],
        "records": [
            {
                "id": "northbridge.public_delays",
                "labels": ["domain.records"],
                "access": {"require_all": ["access.public", "place.northbridge"]},
                "assertions": ["Northbridge records can be delayed."],
            },
            {
                "id": "northbridge.staff_triage",
                "labels": ["domain.records", "community.port_staff"],
                "access": {"require_all": ["community.port_staff"]},
                "assertions": ["Port staff reconcile manifests with berth logs."],
            },
        ],
    }, sort_keys=False), encoding="utf-8")
    memory.init_db()
    memory.load_knowledge_files()
    memory.create_profile(_character("harbor_test", birthplace="Northbridge", occupation="Harbormaster"))
    memory.create_profile(_character("visitor_test", birthplace="Greyhaven", occupation="Visitor"))

    harbormaster = memory.character_knowledge("harbor_test", "What is happening with the manifests?", limit=12)
    visitor = memory.character_knowledge("visitor_test", "What is happening with the manifests?", limit=12)

    assert harbormaster["query_labels"] == ["domain.records"]
    assert {item["id"] for item in harbormaster["items"]} == {
        "northbridge.public_delays", "northbridge.staff_triage"
    }
    assert all(item["access_reason"].startswith("derived:") for item in harbormaster["items"])
    assert visitor["items"] == []


def test_knowledge_taxonomy_rejects_cycles(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "cognition.db")
    monkeypatch.setattr(memory, "KNOWLEDGE_DIR", tmp_path / "knowledge")
    memory.KNOWLEDGE_DIR.mkdir()
    (memory.KNOWLEDGE_DIR / "cycle.yaml").write_text(yaml.safe_dump({
        "classifications": [
            {"id": "a.node", "parents": ["b.node"]},
            {"id": "b.node", "parents": ["a.node"]},
        ],
        "records": [],
    }), encoding="utf-8")
    memory.init_db()

    try:
        memory.load_knowledge_files()
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:  # pragma: no cover - assertion failure clarity
        raise AssertionError("Cyclic taxonomy must be rejected")
