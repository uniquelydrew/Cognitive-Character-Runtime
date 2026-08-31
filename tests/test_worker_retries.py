from __future__ import annotations

import asyncio
import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("COGNITIVE_ROLE", "left")

from services.cognitive_worker import app as worker  # noqa: E402
from services.common import CognitiveRequest, ExecutiveRepeatAssessment, LeftAnalysis, RightAnalysis  # noqa: E402


def test_worker_retries_an_incomplete_json_completion(monkeypatch: pytest.MonkeyPatch):
    responses = iter(
        [
            '{"topic":"topic.general","observations":["partial',
            (
                '{"topic":"topic.general","fact_refs":["identity.name"],'
                '"constraints":["preserve_core"],'
                '"action":"answer","confidence":0.8}'
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


def test_worker_distills_complete_legacy_lobe_json(monkeypatch: pytest.MonkeyPatch):
    responses = iter([
        (
            '{"topic":"self.occupation","observations":["identity.occupation"],'
            '"consistency_constraints":["preserve_core"],'
            '"recommended_strategy":"answer","confidence":0.8}'
        ),
        (
            '{"social_read":"ordinary_exchange","affect":{"curiosity":0.5},'
            '"recommended_tone":"warm","associations":["occupation.work"]}'
        ),
    ])

    async def legacy_completion(*_args, **_kwargs) -> str:
        return next(responses)

    monkeypatch.setattr(worker, "_request_completion", legacy_completion)
    request = CognitiveRequest.model_construct(context={"interaction": {"topic": "self.occupation"}})

    left = asyncio.run(worker._request_model(request, LeftAnalysis))
    right = asyncio.run(worker._request_model(request, RightAnalysis))

    assert left.fact_refs == ["identity.occupation"]
    assert right.action == "inform"
    assert right.association_keys == ["occupation.work"]


def test_worker_recovers_allowlisted_fields_from_interrupted_lobe_json(monkeypatch: pytest.MonkeyPatch):
    async def interrupted_completion(*_args, **_kwargs) -> str:
        return '{"topic":"topic.missing_cargo","observations":["missing cargo","lost shipment"'

    monkeypatch.setattr(worker, "_request_completion", interrupted_completion)
    request = CognitiveRequest.model_construct(context={"interaction": {"topic": "topic.cargo"}})

    result = asyncio.run(worker._request_model(request, LeftAnalysis))

    assert result.topic == "topic.missing_cargo"
    assert result.fact_refs == ["missing cargo", "lost shipment"]
    assert result.action == "answer"


def test_worker_recovers_a_repeat_assessment_from_a_near_miss(monkeypatch: pytest.MonkeyPatch):
    async def near_miss_completion(*_args, **_kwargs) -> str:
        return (
            '{"hypothesis":"wants_more_detail","alternatives":["checking_consistency"],'
            '"recommended_action":"different_angle","speech":"must not be retained"}'
        )

    monkeypatch.setattr(worker, "_request_completion", near_miss_completion)
    request = CognitiveRequest.model_construct(mode="repeat_assessment")

    result = asyncio.run(worker._request_model(request, ExecutiveRepeatAssessment))

    assert result.primary_hypothesis == "wants_more_detail"
    assert result.response_mode == "new_angle"
    assert result.alternative_hypotheses == ["checking_consistency"]


def test_lobe_priority_changes_the_enforced_generation_budget(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker, "ROLE", "left")
    monkeypatch.setattr(worker, "MODEL_MAX_TOKENS", 160)

    high = CognitiveRequest.model_construct(
        context={"role_attention": {"role": "left", "attention_budget": 1.5}}
    )
    low = CognitiveRequest.model_construct(
        context={"role_attention": {"role": "left", "attention_budget": 0.5}}
    )

    assert worker._priority_token_budget(high) == 240
    assert worker._priority_token_budget(low) == 80
