import asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient

from services.common import CharacterDocument, EventRecord
from services.memory import app as memory
from services.orchestrator import app as orchestrator


def test_orchestrator_requires_api_key_and_does_not_emit_permissive_cors(monkeypatch):
    monkeypatch.setattr(orchestrator, "API_AUTH_TOKEN", "test-api-token")
    client = TestClient(orchestrator.app)

    denied = client.get("/profiles", headers={"Origin": "https://untrusted.example"})
    preflight = client.options(
        "/profiles",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert denied.status_code == 401
    assert preflight.status_code == 401
    assert "access-control-allow-origin" not in denied.headers
    assert "access-control-allow-origin" not in preflight.headers


def test_close_succeeds_even_when_optional_reflection_fails(monkeypatch):
    async def failed_reflection(_session_id):
        raise HTTPException(502, "model response was incomplete")

    async def fake_post(_client, _url, _payload):
        return {"id": "sess_close_test", "status": "closed", "closed_at": "2026-01-01T00:00:00+00:00"}

    monkeypatch.setattr(orchestrator, "_reflect", failed_reflection)
    monkeypatch.setattr(orchestrator, "post_json", fake_post)

    result = asyncio.run(orchestrator.close("sess_close_test"))

    assert result["session"]["status"] == "closed"
    assert result["reflection"] is None
    assert "queued for retry" in result["reflection_warning"]


def test_reflection_retry_job_is_durable_and_allows_only_late_reflection_events(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "cognition.db")
    memory.init_db()
    character = CharacterDocument.model_validate({
        "id": "retry_test",
        "identity": {"name": "Retry Test"},
        "biography": "Test character.",
    })
    memory.upsert_character(character, initialize=True)
    session = memory.create_session(memory.SessionCreate(character_id=character.id))
    memory.close_session(session["id"])

    queued = memory.schedule_reflection_retry(
        session["id"], memory.ReflectionRetrySchedule(error="Executive temporarily unavailable.")
    )
    claimed = memory.claim_reflection_retry(memory.ReflectionRetryClaim(lease_seconds=30))

    assert queued["status"] == "pending"
    assert claimed["job"]["session_id"] == session["id"]
    assert claimed["job"]["status"] == "running"
    assert claimed["job"]["attempts"] == 1

    reflection = memory.add_event(EventRecord(
        character_id=character.id,
        session_id=session["id"],
        event_type="reflection",
        actor="executive",
        content="Deferred reflection completed.",
    ))
    completed = memory.complete_reflection_retry(session["id"])

    assert reflection.id
    assert completed["status"] == "completed"
    try:
        memory.add_event(EventRecord(
            character_id=character.id,
            session_id=session["id"],
            event_type="user_message",
            actor="user",
            content="This must remain rejected.",
        ))
    except HTTPException as error:
        assert error.status_code == 409
    else:  # pragma: no cover - assertion failure clarity
        raise AssertionError("Closed sessions must not accept new conversational events")


def test_deferred_reflection_processor_completes_a_claimed_job(monkeypatch):
    calls: list[str] = []

    async def fake_post(_client, url, _payload):
        calls.append(url)
        if url.endswith("/reflection-jobs/claim"):
            return {"job": {"session_id": "sess_retry", "character_id": "retry_test"}}
        return {"found": True}

    async def successful_reflection(_session_id):
        return orchestrator.ReflectionResult(
            session_id="sess_retry",
            summary="Deferred reflection completed.",
            mutation_results=[],
            executive={},
        )

    monkeypatch.setattr(orchestrator, "post_json", fake_post)
    monkeypatch.setattr(orchestrator, "_reflect", successful_reflection)

    result = asyncio.run(orchestrator.process_deferred_reflections_once())

    assert result["status"] == "completed"
    assert calls[0].endswith("/reflection-jobs/claim")
    assert calls[-1].endswith("/reflection-jobs/sess_retry/complete")
