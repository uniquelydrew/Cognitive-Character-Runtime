from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

from services.common import CharacterDocument, KnowledgeCatalog
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


def test_catalog_import_is_atomic_and_activates_one_managed_source(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "cognition.db")
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(memory, "KNOWLEDGE_DIR", knowledge_dir)
    (knowledge_dir / "starter.yaml").write_text(yaml.safe_dump({
        "classifications": [{"id": "access.public"}, {"id": "domain.starter"}],
        "records": [{
            "id": "starter.public_fact",
            "labels": ["domain.starter"],
            "access": {"require_all": ["access.public"]},
            "assertions": ["The starter source is active before an import."],
        }],
    }, sort_keys=False), encoding="utf-8")
    # This file is deliberately a valid YAML document, but it is a schema sample
    # and must never be co-loaded into the production corpus.
    (knowledge_dir / "catalog.example.yaml").write_text(yaml.safe_dump({
        "classifications": [{"id": "access.example"}], "records": [],
    }, sort_keys=False), encoding="utf-8")
    memory.init_db()
    memory.load_knowledge_files()
    assert [node.id for node in memory.current_knowledge_catalog().classifications] == [
        "access.public", "domain.starter"
    ]

    imported_yaml = yaml.safe_dump({
        "classifications": [
            {"id": "access.public", "aliases": ["common knowledge"]},
            {"id": "place.harbor", "aliases": ["harbor"]},
            {"id": "domain.shipping", "aliases": ["shipping"]},
        ],
        "records": [{
            "id": "harbor.public_shipping_fact",
            "labels": ["place.harbor", "domain.shipping"],
            "access": {"require_all": ["access.public", "place.harbor"]},
            "assertions": ["Harbor shipping notices are posted publicly."],
            "source": "Harbor notice board",
        }],
    }, sort_keys=False)
    result = memory.import_knowledge_catalog(memory.KnowledgeCatalogRequest(yaml=imported_yaml))

    assert result["managed"] is True
    assert result["source_file"] == "catalog.yaml"
    assert memory._knowledge_source_paths() == [knowledge_dir / "catalog.yaml"]
    assert [record.id for record in memory.current_knowledge_catalog().records] == [
        "harbor.public_shipping_fact"
    ]
    persisted = (knowledge_dir / "catalog.yaml").read_text(encoding="utf-8")

    invalid_yaml = yaml.safe_dump({
        "classifications": [{"id": "access.public"}],
        "records": [{
            "id": "bad.invalid_record",
            "labels": ["missing.label"],
            "assertions": ["This must not replace the current catalog."],
        }],
    }, sort_keys=False)
    with pytest.raises(HTTPException, match="Knowledge catalog is invalid") as error:
        memory.import_knowledge_catalog(memory.KnowledgeCatalogRequest(yaml=invalid_yaml))

    assert error.value.status_code == 422
    assert (knowledge_dir / "catalog.yaml").read_text(encoding="utf-8") == persisted
    assert [record.id for record in memory.current_knowledge_catalog().records] == [
        "harbor.public_shipping_fact"
    ]


def test_catalog_schema_example_is_valid_and_can_be_downloaded(tmp_path: Path, monkeypatch):
    sample_path = Path(__file__).resolve().parents[1] / "knowledge" / "catalog.example.yaml"
    example = KnowledgeCatalog.model_validate(yaml.safe_load(sample_path.read_text(encoding="utf-8")))
    memory._validate_knowledge_taxonomy(example.classifications, example.records)

    monkeypatch.setattr(memory, "KNOWLEDGE_DIR", tmp_path / "missing-knowledge")
    downloaded = memory._knowledge_sample_text()
    fallback = KnowledgeCatalog.model_validate(yaml.safe_load(downloaded))
    memory._validate_knowledge_taxonomy(fallback.classifications, fallback.records)


def test_catalog_accepts_and_preserves_setting_specific_epistemic_types(tmp_path: Path, monkeypatch):
    """Catalog categories are setting vocabulary, not the runtime memory enum."""

    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "cognition.db")
    monkeypatch.setattr(memory, "KNOWLEDGE_DIR", tmp_path / "knowledge")
    setting_types = [
        "canon_policy",
        "contested_principle",
        "provisional_model",
        "historical_fact",
        "partial_fact",
        "experienced_fact",
        "interpretive_fact",
        "realization",
        "principle",
        "civic_practice",
        "terminology",
        "unresolved",
    ]
    imported_yaml = yaml.safe_dump({
        "classifications": [
            {"id": "access.public"},
            {"id": "domain.civics"},
        ],
        "records": [
            {
                "id": f"civics.example_{index}",
                "labels": ["domain.civics"],
                "access": {"require_all": ["access.public"]},
                "assertions": [f"A setting assertion classified as {epistemic_type}."],
                "epistemic_type": epistemic_type,
            }
            for index, epistemic_type in enumerate(setting_types)
        ],
    }, sort_keys=False)

    memory.init_db()
    memory.import_knowledge_catalog(memory.KnowledgeCatalogRequest(yaml=imported_yaml))

    catalog = memory.current_knowledge_catalog()
    assert {record.epistemic_type for record in catalog.records} == set(setting_types)
