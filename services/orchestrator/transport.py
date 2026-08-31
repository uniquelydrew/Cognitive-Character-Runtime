"""HTTP boundary helpers for orchestrator dependencies."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException


def upstream_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str) and detail:
            return detail
    except (ValueError, TypeError):
        pass
    return response.text or "The request could not be completed."


async def request_json(client: httpx.AsyncClient, method: str, url: str, payload: Any = None) -> Any:
    timeout_messages = {
        "GET": "A required service did not respond in time. Please retry.",
        "POST": "The character is taking longer than expected. Please retry.",
        "PUT": "The update took too long. Please retry.",
    }
    try:
        response = await client.request(method, url, json=payload) if payload is not None else await client.request(method, url)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, timeout_messages.get(method, "The service timed out. Please retry.")) from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "A required service is unavailable. Please retry shortly.") from exc
    if response.status_code >= 400:
        raise HTTPException(response.status_code, upstream_detail(response))
    return response.json()


async def get_json(client: httpx.AsyncClient, url: str) -> Any:
    return await request_json(client, "GET", url)


async def post_json(client: httpx.AsyncClient, url: str, payload: Any) -> Any:
    return await request_json(client, "POST", url, payload)


async def put_json(client: httpx.AsyncClient, url: str, payload: Any) -> Any:
    return await request_json(client, "PUT", url, payload)


async def get_text(client: httpx.AsyncClient, url: str) -> httpx.Response:
    try:
        response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "The export took too long. Please retry.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "The export service is unavailable. Please retry shortly.") from exc
    if response.status_code >= 400:
        raise HTTPException(response.status_code, upstream_detail(response))
    return response
