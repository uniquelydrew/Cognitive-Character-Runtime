from __future__ import annotations

import asyncio
import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("COGNITIVE_ROLE", "left")

from services.cognitive_worker import app as worker  # noqa: E402
from services.common import CognitiveRequest, LeftAnalysis  # noqa: E402


def test_worker_retries_an_incomplete_json_completion(monkeypatch: pytest.MonkeyPatch):
    responses = iter(
        [
            '{"topic":"topic.general","observations":["partial',
            (
                '{"topic":"topic.general","observations":["established fact"],'
                '"consistency_constraints":["preserve biography"],'
                '"recommended_strategy":"answer concisely","confidence":0.8}'
            ),
        ]
    )
    retries: list[bool] = []

    async def fake_completion(*_args, corrective_retry: bool, **_kwargs) -> str:
        retries.append(corrective_retry)
        return next(responses)

    monkeypatch.setattr(worker, "_request_completion", fake_completion)
    request = CognitiveRequest.model_construct()
    result = asyncio.run(worker._request_model(request, LeftAnalysis))

    assert result.topic == "topic.general"
    assert retries == [False, True]


def test_worker_hides_raw_contract_trace_after_retries(monkeypatch: pytest.MonkeyPatch):
    async def incomplete_completion(*_args, **_kwargs) -> str:
        return '{"topic":"topic.general"'

    monkeypatch.setattr(worker, "_request_completion", incomplete_completion)
    request = CognitiveRequest.model_construct()

    with pytest.raises(HTTPException, match="incomplete response") as exc_info:
        asyncio.run(worker._request_model(request, LeftAnalysis))

    assert exc_info.value.status_code == 502
