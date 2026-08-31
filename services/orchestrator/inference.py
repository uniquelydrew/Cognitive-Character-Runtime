"""Validated worker inference and health instrumentation."""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from services.common import CognitiveRequest, CognitiveResponse
from services.orchestrator.transport import post_json


async def infer(
    client: httpx.AsyncClient, base_url: str, request: CognitiveRequest, expected_role: str,
) -> dict[str, Any]:
    """Call a cognitive worker and validate its role-specific response envelope."""

    data = await post_json(client, f"{base_url}/infer", request.model_dump(mode="json"))
    try:
        response = CognitiveResponse.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(502, f"{expected_role} worker returned an invalid response envelope: {exc}") from exc
    if response.role != expected_role:
        raise HTTPException(502, f"Expected {expected_role} worker response, received {response.role!r}")
    return response.result


async def infer_timed(
    client: httpx.AsyncClient,
    base_url: str,
    request: CognitiveRequest,
    expected_role: str,
    metrics: dict[str, dict[str, int | float | None]],
) -> tuple[dict[str, Any], int]:
    """Call a worker while maintaining independent per-role latency metrics."""

    started = time.perf_counter()
    try:
        result = await infer(client, base_url, request, expected_role)
    except Exception:
        elapsed = round((time.perf_counter() - started) * 1000)
        metric = metrics[expected_role]
        metric["calls"] = int(metric["calls"] or 0) + 1
        metric["failures"] = int(metric["failures"] or 0) + 1
        metric["last_ms"] = elapsed
        raise
    elapsed = round((time.perf_counter() - started) * 1000)
    metric = metrics[expected_role]
    calls = int(metric["calls"] or 0) + 1
    previous_average = metric["average_ms"]
    metric["calls"] = calls
    metric["last_ms"] = elapsed
    metric["average_ms"] = round(
        elapsed if previous_average is None else ((float(previous_average) * (calls - 1)) + elapsed) / calls, 1,
    )
    return result, elapsed


async def probe_worker(
    client: httpx.AsyncClient,
    role: str,
    url: str,
    metrics: dict[str, dict[str, int | float | None]],
) -> dict[str, Any]:
    """Probe one role without permitting a degraded peer to hide it."""

    started = time.perf_counter()
    try:
        response = await client.get(f"{url}/health")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("worker health response was not an object")
        readiness, model = "ready", str(payload.get("model") or "configured")
    except (httpx.HTTPError, ValueError, TypeError):
        readiness, model = "unavailable", None
    metric = metrics[role]
    return {
        "role": role, "status": readiness, "model": model,
        "health_probe_ms": round((time.perf_counter() - started) * 1000),
        "calls": int(metric["calls"] or 0), "failures": int(metric["failures"] or 0),
        "last_inference_ms": metric["last_ms"], "average_inference_ms": metric["average_ms"],
    }
