from __future__ import annotations

import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from services.common import CognitiveRequest, CognitiveResponse, ModelOutput, output_model_for

ROLE = os.getenv("COGNITIVE_ROLE", "").lower()
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://ollama:11434/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "unused")
MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "150"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "240"))
MODEL_OUTPUT_ATTEMPTS = int(os.getenv("MODEL_OUTPUT_ATTEMPTS", "2"))

if ROLE not in {"left", "right", "executive"}:
    raise RuntimeError("COGNITIVE_ROLE must be one of: left, right, executive")
if MODEL_TIMEOUT_SECONDS <= 0:
    raise RuntimeError("MODEL_TIMEOUT_SECONDS must be positive")
if MODEL_MAX_TOKENS <= 0:
    raise RuntimeError("MODEL_MAX_TOKENS must be positive")
if MODEL_OUTPUT_ATTEMPTS <= 0:
    raise RuntimeError("MODEL_OUTPUT_ATTEMPTS must be positive")

app = FastAPI(title=f"Cognitive Worker: {ROLE}", version="0.2.0")


def _system_prompt(mode: str, output_model: type[ModelOutput], *, corrective_retry: bool = False) -> str:
    role_instructions = {
        "left": (
            "You are the analytic hemisphere of a persistent fictional character. "
            "Identify relevant established facts, consistency constraints, causal implications, and a response strategy. "
            "Do not invent canonical facts or propose mutations."
        ),
        "right": (
            "You are the associative and social hemisphere of a persistent fictional character. "
            "Assess affect, tone, subtext, associations, and social consequences. "
            "Do not invent canonical facts or propose mutations."
        ),
        "executive": (
            "You are the executive function of a persistent fictional character. "
            "Arbitrate the supplied left and right analyses while maintaining continuity. "
            "Before speaking, inspect context.executive_repeat_review, which is prepared after both "
            "hemispheres finish. Treat semantic_repeat_candidate as a meaningful rephrased-repeat "
            "signal even when the wording differs. Follow context.conversation_dynamics.response_posture: "
            "normal means answer neutrally; reclarify means restate or clarify without escalation; "
            "confused means express sincere, non-accusatory confusion; defensive means set a polite, "
            "proportionate boundary. This is a tone constraint, not permission to change facts. "
            "Use only supplied character data and memories as established facts. "
            "Never rewrite raw history or immutable core biography. Any state change must be a typed, "
            "evidence-backed mutation proposal."
        ),
    }
    reflection = (
        " This is reflection mode: summarize the completed interaction and propose only provenance-backed "
        "derived memories, event links, or mutable revisions."
        if mode == "reflection"
        else " This is turn mode: produce the character's next spoken response."
    )
    concise_output = (
        " Keep all text concise. Do not repeat a value. For analytic lists, use at most three short items."
        if ROLE in {"left", "right"}
        else " Keep the response concise and do not repeat a value."
    )
    correction = (
        " This is a corrective retry: the previous response was incomplete. Return the minimal valid JSON object now."
        if corrective_retry
        else ""
    )
    return (
        f"{role_instructions[ROLE]}{reflection}{concise_output}{correction} "
        "Return one JSON object only: no markdown, no explanation, and no additional keys. "
        "Every key in this exact shape is required. Use empty arrays when there are no actions to propose. "
        f"Exact output shape: {_output_example(output_model)}"
    )


def _output_example(output_model: type[ModelOutput]) -> str:
    """Compact examples are more reliable for small local models than a full JSON Schema."""

    examples = {
        "LeftAnalysis": {
            "topic": "stable topic identifier",
            "observations": ["established observation"],
            "consistency_constraints": ["constraint to preserve"],
            "recommended_strategy": "brief strategy",
            "confidence": 0.8,
        },
        "RightAnalysis": {
            "social_read": "social interpretation",
            "affect": {"curiosity": 0.4, "annoyance": 0.1},
            "recommended_tone": "patient",
            "associations": ["relevant association"],
        },
        "ExecutiveTurn": {
            "goal": "maintain continuity",
            "strategy": "answer using established character context",
            "speech": "The character's spoken response as a plain string.",
            "topic": "stable topic identifier",
            "mutations": [],
            "memory_writes": [],
        },
        "ExecutiveReflection": {
            "summary": "brief interaction summary",
            "related_event_ids": ["evt_source"],
            "mutations": [],
            "links": [],
        },
    }
    return json.dumps(examples[output_model.__name__], separators=(",", ":"))


def _json_response_format() -> dict[str, str]:
    """The documented OpenAI-compatible JSON mode supported by Ollama."""

    return {"type": "json_object"}


def _model_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {MODEL_API_KEY}"} if MODEL_API_KEY else {}


async def _request_completion(
    req: CognitiveRequest,
    output_model: type[ModelOutput],
    *,
    corrective_retry: bool,
) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt(req.mode, output_model, corrective_retry=corrective_retry),
            },
            {"role": "user", "content": req.model_dump_json()},
        ],
        "temperature": 0.15 if ROLE in {"left", "executive"} else 0.55,
        # Every worker returns a small structured artifact. Bounding output avoids a
        # queued local-model request consuming the entire orchestration time budget.
        "max_tokens": MODEL_MAX_TOKENS,
        "response_format": _json_response_format(),
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(MODEL_TIMEOUT_SECONDS, connect=10)) as client:
            response = await client.post(
                f"{MODEL_BASE_URL}/chat/completions",
                json=payload,
                headers=_model_headers(),
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=f"The {ROLE} model did not respond within {MODEL_TIMEOUT_SECONDS:g} seconds. Please retry.",
        ) from exc
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Model provider request failed: {exc}") from exc

    if not isinstance(content, str):
        raise HTTPException(status_code=502, detail="Model provider returned a non-text completion.")
    return content


async def _request_model(req: CognitiveRequest, output_model: type[ModelOutput]) -> ModelOutput:
    """Validate live output, giving a small local model one concise repair attempt."""

    for attempt in range(MODEL_OUTPUT_ATTEMPTS):
        content = await _request_completion(req, output_model, corrective_retry=attempt > 0)
        try:
            return output_model.model_validate_json(content)
        except ValidationError:
            if attempt + 1 < MODEL_OUTPUT_ATTEMPTS:
                continue
    raise HTTPException(
        status_code=502,
        detail=(
            f"The {ROLE} model returned an incomplete response after "
            f"{MODEL_OUTPUT_ATTEMPTS} attempts. Please retry."
        ),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Readiness includes the configured live model provider and model registration."""

    try:
        async with httpx.AsyncClient(timeout=min(MODEL_TIMEOUT_SECONDS, 10)) as client:
            response = await client.get(f"{MODEL_BASE_URL}/models", headers=_model_headers())
            response.raise_for_status()
            models = response.json().get("data", [])
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise HTTPException(status_code=503, detail=f"Model provider is unavailable: {exc}") from exc

    available = {str(model.get("id")) for model in models if isinstance(model, dict)}
    if MODEL_NAME not in available:
        raise HTTPException(status_code=503, detail=f"Configured model is not available: {MODEL_NAME}")
    return {"status": "ready", "role": ROLE, "provider": MODEL_BASE_URL, "model": MODEL_NAME}


@app.post("/infer", response_model=CognitiveResponse)
async def infer(req: CognitiveRequest) -> CognitiveResponse:
    try:
        output_model = output_model_for(ROLE, req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = await _request_model(req, output_model)
    return CognitiveResponse(role=ROLE, result=result.model_dump(mode="json"))
