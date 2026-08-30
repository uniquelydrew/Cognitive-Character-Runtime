import asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient

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
    assert "reflection could not be generated" in result["reflection_warning"]
