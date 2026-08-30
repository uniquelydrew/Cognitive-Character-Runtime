import httpx
import pytest

from services.orchestrator.app import put_json


@pytest.mark.asyncio
async def test_put_json_uses_put_method():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url == "http://profile.test/profiles/example"
        return httpx.Response(200, json={"updated": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await put_json(client, "http://profile.test/profiles/example", {"id": "example"})

    assert result == {"updated": True}
