import asyncio
import json

import httpx

from services.orchestrator.app import semantic_repeat_evidence


def test_embedding_repeat_evidence_batches_current_and_prior_questions():
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["payload"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "test-embed",
            "embeddings": [
                [1.0, 0.0],
                [0.82, 0.5723635],
                [0.0, 1.0],
            ],
        })

    async def run() -> dict[str, object]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await semantic_repeat_evidence(
                client,
                "Which city was her early home?",
                [
                    {"id": "birth", "content": "Where was she born?"},
                    {"id": "cargo", "content": "What happened to the shipment?"},
                ],
            )

    evidence = asyncio.run(run())

    assert observed["path"] == "/api/embed"
    assert observed["payload"]["input"][0] == "Which city was her early home?"
    assert evidence["available"] is True
    assert evidence["model"] == "test-embed"
    assert evidence["matches"]["birth"] > 0.80
    assert evidence["matches"]["cargo"] == 0.0


def test_embedding_failures_leave_repeat_review_on_the_safe_fallback_path():
    async def run() -> dict[str, object]:
        transport = httpx.MockTransport(lambda _request: httpx.Response(503))
        async with httpx.AsyncClient(transport=transport) as client:
            return await semantic_repeat_evidence(
                client,
                "A question",
                [{"id": "prior", "content": "An earlier question"}],
            )

    evidence = asyncio.run(run())

    assert evidence["available"] is False
    assert evidence["matches"] == {}
    assert evidence["reason"] == "embedding_provider_unavailable"
