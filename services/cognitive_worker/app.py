from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from services.common import (
    CognitiveRequest,
    CognitiveResponse,
    ExecutiveRepeatAssessment,
    LeftAnalysis,
    ModelOutput,
    RightAnalysis,
    output_model_for,
)

ROLE = os.getenv("COGNITIVE_ROLE", "").lower()
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://ollama:11434/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "unused")
MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "40"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "240"))
MODEL_REPEAT_MAX_TOKENS = int(os.getenv("MODEL_REPEAT_MAX_TOKENS", str(MODEL_MAX_TOKENS)))
MODEL_REPEAT_TEMPERATURE = float(os.getenv("MODEL_REPEAT_TEMPERATURE", "0.32"))
MODEL_OUTPUT_ATTEMPTS = int(os.getenv("MODEL_OUTPUT_ATTEMPTS", "2"))

if MODEL_TIMEOUT_SECONDS <= 0 or MODEL_MAX_TOKENS <= 0 or MODEL_REPEAT_MAX_TOKENS <= 0:
    raise RuntimeError("Model timeout and output token budgets must be positive")
if not 1 <= MODEL_OUTPUT_ATTEMPTS <= 2:
    raise RuntimeError("MODEL_OUTPUT_ATTEMPTS must be between 1 and 2")

if ROLE not in {"left", "right", "executive"}:
    raise RuntimeError("COGNITIVE_ROLE must be one of: left, right, executive")
if MODEL_TIMEOUT_SECONDS <= 0:
    raise RuntimeError("MODEL_TIMEOUT_SECONDS must be positive")
if MODEL_MAX_TOKENS <= 0:
    raise RuntimeError("MODEL_MAX_TOKENS must be positive")
if MODEL_REPEAT_MAX_TOKENS <= 0:
    raise RuntimeError("MODEL_REPEAT_MAX_TOKENS must be positive")
if not 0.0 <= MODEL_REPEAT_TEMPERATURE <= 2.0:
    raise RuntimeError("MODEL_REPEAT_TEMPERATURE must be between 0 and 2")
if MODEL_OUTPUT_ATTEMPTS <= 0:
    raise RuntimeError("MODEL_OUTPUT_ATTEMPTS must be positive")

app = FastAPI(title=f"Cognitive Worker: {ROLE}", version="0.2.0")


def _system_prompt(mode: str, output_model: type[ModelOutput], *, corrective_retry: bool = False) -> str:
    role_instructions = {
        "left": (
            "You are the analytic hemisphere of a persistent fictional character. "
            "Identify relevant established facts, consistency constraints, causal implications, and a response action. "
            "context.role_attention is the source-controlled attention allocation for this role; it affects "
            "your bounded work budget but never allows facts outside authorized context. "
            "context.general_knowledge is the complete authorized general-knowledge view; do not infer or disclose "
            "facts outside it. "
            "Do not invent canonical facts or propose mutations."
        ),
        "right": (
            "You are the associative and social hemisphere of a persistent fictional character. "
            "Assess affect, tone, subtext, associations, and social consequences. "
            "context.role_attention is the source-controlled attention allocation for this role; it affects "
            "your bounded work budget but never allows facts outside authorized context. "
            "context.general_knowledge is the complete authorized general-knowledge view; do not infer or disclose "
            "facts outside it. "
            "Do not invent canonical facts or propose mutations."
        ),
        "executive": (
            "You are the executive function of a persistent fictional character. "
            "Arbitrate the supplied left and right analyses while maintaining continuity. "
            "context.weighted_arbitration is an enforced, source-controlled plan: left constraints are always "
            "binding, and when non-factual response choices conflict you must follow its primary_role and "
            "primary_packet. Do not substitute your own weighting. "
            "Before speaking, inspect context.executive_repeat_review, which is prepared after both "
            "hemispheres finish. Treat semantic_repeat_candidate as a meaningful rephrased-repeat "
            "signal even when the wording differs. context.conversation_dynamics contains measured pressure "
            "and a suggested posture, not a mandatory escalation. Use discretion: simple repeats, requests "
            "for clarification, or honest confusion normally merit a patient reframe and repeat_escalation=hold. "
            "Choose repeat_escalation=increase only when the user is clearly pressing an already-sufficient "
            "answer and the evidence supports proportionate suspicion or defensiveness. Choose deescalate "
            "when a charged subject is being handled constructively. If context.lobe_execution.mode is reused, "
            "reuse the supplied analysis and reframe; do not require new lobe reasoning. When "
            "context.repeat_deliberation.enabled is true, use its assessment as a fallible hypothesis about "
            "why the user repeated the question. previous_speech is the answer the user just saw: your speech "
            "must not repeat or closely paraphrase it. Instead, follow the assessment's response_mode with a "
            "different established facet, a focused question that tests the hypothesis, or a proportionate "
            "boundary. If repeat_deliberation contains rejected_speech, the preceding attempt was rejected for "
            "echoing the answer; make a meaningfully different response now. "
            "This is not permission to change facts. "
            "Use only supplied character data, memories, and context.general_knowledge as established facts. "
            "For every factual assertion in speech, emit a factual_claims item with the exact supporting "
            "context.claim_evidence key. Do not make a factual assertion if no authorized citation exists. "
            "When context.claim_coverage_retry is present, it records a rejected prior turn: produce a fresh "
            "response and satisfy its citation requirement exactly. "
            "Never rewrite raw history or immutable core biography. Any state change must be a typed, "
            "evidence-backed mutation proposal."
        ),
    }
    if ROLE == "executive" and mode == "repeat_assessment":
        role_instructions["executive"] = (
            "You are the executive function in an internal repeat-interpretation phase. "
            "Assess only observable conversation evidence and supplied character context to form tentative "
            "explanations for why the user repeated the question. Do not speak to the user, do not claim a "
            "motive as fact, do not propose mutations, and do not change emotional state. Select one response "
            "mode that gives the speaking executive a useful next move."
        )
    reflection = (
        " This is reflection mode: summarize the completed interaction and propose only provenance-backed "
        "derived memories, event links, or mutable revisions."
        if mode == "reflection"
        else (
            " This is repeat-assessment mode: return a compact decision artifact, never user-facing speech."
            if mode == "repeat_assessment"
            else " This is turn mode: produce the character's next spoken response."
        )
    )
    concise_output = (
        " Return a compact semantic control artifact, never user-facing prose or full sentences. "
        "Use short lower_snake_case or dot-separated keys: fact_refs such as identity.birthplace, "
        "constraints such as preserve_core, actions such as answer or clarify, and association_keys "
        "such as cargo.missing. action, intent, tone, and risk should be short labels, not explanations. "
        "Use at most four list entries."
        if ROLE in {"left", "right"}
        else (
            " Use short lower_snake_case labels and at most four alternatives or evidence codes."
            if mode == "repeat_assessment"
            else " Keep the response concise and do not repeat a value."
        )
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
            "topic": "self.birthplace",
            "fact_refs": ["identity.birthplace"],
            "constraints": ["preserve_core"],
            "action": "answer",
            "confidence": 0.8,
        },
        "RightAnalysis": {
            "action": "inform",
            "affect": {"curiosity": 0.4, "annoyance": 0.1},
            "tone": "warm",
            "risk": "low",
            "association_keys": ["home.birthplace"],
        },
        "ExecutiveTurn": {
            "goal": "maintain continuity",
            "strategy": "answer using established character context",
            "speech": "The character's spoken response as a plain string.",
            "topic": "stable topic identifier",
            "repeat_escalation": "hold",
            "factual_claims": [{"text": "I was born in Northbridge.", "evidence_refs": ["identity.birthplace"]}],
            "historical_relationships": [{"prior_event_id": "evt_prior", "relationship": "clarifies", "confidence": 0.8}],
            "mutations": [],
            "memory_writes": [],
        },
        "ExecutiveRepeatAssessment": {
            "primary_hypothesis": "wants_a_different_practical_angle",
            "alternative_hypotheses": ["did_not_find_prior_answer_specific_enough"],
            "evidence_codes": ["exact_question_repeated", "prior_answer_available"],
            "response_mode": "new_angle",
            "confidence": 0.55,
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


def _uses_extended_repeat_budget(req: CognitiveRequest) -> bool:
    if ROLE != "executive":
        return False
    if req.mode == "repeat_assessment":
        return True
    deliberation = req.context.get("repeat_deliberation", {})
    return isinstance(deliberation, dict) and bool(deliberation.get("enabled"))


def _priority_token_budget(req: CognitiveRequest) -> int:
    """Apply profile priority to lobe compute without starving either role."""

    if ROLE not in {"left", "right"}:
        return MODEL_MAX_TOKENS
    attention = req.context.get("role_attention", {})
    if not isinstance(attention, dict) or attention.get("role") != ROLE:
        return MODEL_MAX_TOKENS
    try:
        multiplier = float(attention.get("attention_budget", 1.0))
    except (TypeError, ValueError):
        multiplier = 1.0
    # A profile can prioritize one lobe, never switch the other one off. Compact
    # artifacts remain reliable with this 80-token floor.
    multiplier = max(0.5, min(1.5, multiplier))
    return max(80, round(MODEL_MAX_TOKENS * multiplier))


def _compact_items(value: Any, limit: int) -> list[str]:
    """Preserve the first useful model-proposed keys without accepting arbitrary types."""

    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:limit]


def _compact_label(value: Any, fallback: str) -> str:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _compact_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _repeat_response_mode(value: Any) -> str:
    """Normalize a small model's near-miss repeat response label safely."""

    label = _compact_label(value, "invite_specificity").lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clarify": "invite_specificity",
        "clarification": "invite_specificity",
        "ask_clarifying_question": "invite_specificity",
        "ask_for_specificity": "invite_specificity",
        "different_angle": "new_angle",
        "expand": "new_angle",
        "understanding_check": "check_understanding",
        "check_comprehension": "check_understanding",
        "consistency_check": "test_consistency",
        "boundary": "set_boundary",
    }
    label = aliases.get(label, label)
    return label if label in {
        "new_angle",
        "check_understanding",
        "invite_specificity",
        "test_consistency",
        "set_boundary",
    } else "invite_specificity"


def _distill_compact_artifact(
    content: str,
    req: CognitiveRequest,
    output_model: type[ModelOutput],
) -> dict[str, Any] | None:
    """Translate complete near-miss control JSON into the current safe shape.

    This permits only Left/Right artifacts and the non-user-facing Executive repeat
    assessment. It never repairs Executive speech, mutations, or memory writes.
    Unknown model keys are discarded; the resulting artifact is still validated
    against its Pydantic contract below.
    """

    if output_model not in {LeftAnalysis, RightAnalysis, ExecutiveRepeatAssessment}:
        return None
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    interaction = req.context.get("interaction", {})
    fallback_topic = interaction.get("topic") if isinstance(interaction, dict) else None
    if output_model is ExecutiveRepeatAssessment:
        # This artifact is internal control data only. Whitelisting known keys
        # makes the repeat path tolerant of a lightweight model adding prose or
        # retaining fields from its spoken-turn contract.
        return {
            "primary_hypothesis": _compact_label(
                payload.get("primary_hypothesis", payload.get("hypothesis")),
                "unclear_repeat_intent",
            ),
            "alternative_hypotheses": _compact_items(
                payload.get("alternative_hypotheses", payload.get("alternatives", [])), 4
            ),
            "evidence_codes": _compact_items(
                payload.get("evidence_codes", payload.get("evidence", [])), 4
            ),
            "response_mode": _repeat_response_mode(
                payload.get("response_mode", payload.get("recommended_action"))
            ),
            "confidence": _compact_confidence(payload.get("confidence")),
        }
    if output_model is LeftAnalysis:
        return {
            "topic": _compact_label(payload.get("topic"), str(fallback_topic or "topic.general")),
            "fact_refs": _compact_items(payload.get("fact_refs", payload.get("observations", [])), 4),
            "constraints": _compact_items(
                payload.get("constraints", payload.get("consistency_constraints", [])), 3
            ),
            "action": _compact_label(
                payload.get("action", payload.get("recommended_strategy")), "answer"
            ),
            "confidence": _compact_confidence(payload.get("confidence")),
        }
    return {
        "action": _compact_label(payload.get("action", payload.get("intent")), "inform"),
        "affect": payload.get("affect") if isinstance(payload.get("affect"), dict) else {},
        "tone": _compact_label(payload.get("tone", payload.get("recommended_tone")), "neutral"),
        "risk": _compact_label(payload.get("risk"), "low"),
        "association_keys": _compact_items(payload.get("association_keys", payload.get("associations", [])), 4),
    }


def _partial_string_field(content: str, field_name: str) -> str | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"', content)
    if not match:
        return None
    try:
        value = json.loads(f'"{match.group(1)}"')
    except ValueError:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _partial_string_array(content: str, field_name: str, limit: int) -> list[str]:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[([^\]]*)', content, flags=re.DOTALL)
    if not match:
        return []
    items: list[str] = []
    for raw in re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1)):
        try:
            value = json.loads(f'"{raw}"')
        except ValueError:
            continue
        if isinstance(value, str) and value.strip():
            items.append(value.strip())
        if len(items) == limit:
            break
    return items


def _recover_interrupted_lobe_json(
    content: str,
    req: CognitiveRequest,
    output_model: type[ModelOutput],
) -> dict[str, Any] | None:
    """Recover only complete, allowlisted keys from an interrupted lobe object.

    Local models occasionally repeat an array until the generation limit, leaving
    the JSON unclosed. An Executive response must never be repaired this way,
    because it can carry user-facing speech and state mutations.
    """

    if output_model not in {LeftAnalysis, RightAnalysis, ExecutiveRepeatAssessment} or not content.lstrip().startswith("{"):
        return None
    interaction = req.context.get("interaction", {})
    fallback_topic = interaction.get("topic") if isinstance(interaction, dict) else None
    if output_model is ExecutiveRepeatAssessment:
        # Unlike a spoken ExecutiveTurn, this never contains user-facing text or
        # mutations. A bounded best-effort recovery is therefore safe and avoids
        # turning an incomplete internal hypothesis into a failed conversation.
        confidence_match = re.search(r'"confidence"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', content)
        confidence: Any = confidence_match.group(1) if confidence_match else None
        return {
            "primary_hypothesis": _partial_string_field(content, "primary_hypothesis")
            or _partial_string_field(content, "hypothesis")
            or "unclear_repeat_intent",
            "alternative_hypotheses": _partial_string_array(content, "alternative_hypotheses", 4)
            or _partial_string_array(content, "alternatives", 4),
            "evidence_codes": _partial_string_array(content, "evidence_codes", 4)
            or _partial_string_array(content, "evidence", 4),
            "response_mode": _repeat_response_mode(
                _partial_string_field(content, "response_mode")
                or _partial_string_field(content, "recommended_action")
            ),
            "confidence": _compact_confidence(confidence),
        }
    if output_model is LeftAnalysis:
        topic = _partial_string_field(content, "topic")
        fact_refs = _partial_string_array(content, "fact_refs", 4) or _partial_string_array(content, "observations", 4)
        action = _partial_string_field(content, "action") or _partial_string_field(content, "recommended_strategy")
        # A topic on its own is too little evidence to trust; it remains a
        # retryable incomplete completion. A completed fact or action is enough
        # to recover the bounded lobe artifact safely.
        if not any((fact_refs, action)):
            return None
        confidence_match = re.search(r'"confidence"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', content)
        confidence: Any = confidence_match.group(1) if confidence_match else None
        return {
            "topic": topic or str(fallback_topic or "topic.general"),
            "fact_refs": fact_refs,
            "constraints": _partial_string_array(content, "constraints", 3)
            or _partial_string_array(content, "consistency_constraints", 3),
            "action": action or "answer",
            "confidence": _compact_confidence(confidence),
        }

    action = _partial_string_field(content, "action") or _partial_string_field(content, "intent")
    tone = _partial_string_field(content, "tone") or _partial_string_field(content, "recommended_tone")
    association_keys = _partial_string_array(content, "association_keys", 4) or _partial_string_array(
        content, "associations", 4
    )
    if not any((action, tone, association_keys)):
        return None
    return {
        "action": action or "inform",
        "affect": {},
        "tone": tone or "neutral",
        "risk": _partial_string_field(content, "risk") or "low",
        "association_keys": association_keys,
    }


def _model_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {MODEL_API_KEY}"} if MODEL_API_KEY else {}


async def _request_completion(
    req: CognitiveRequest,
    output_model: type[ModelOutput],
    *,
    corrective_retry: bool,
) -> str:
    extended_repeat_budget = _uses_extended_repeat_budget(req)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt(req.mode, output_model, corrective_retry=corrective_retry),
            },
            {"role": "user", "content": req.model_dump_json()},
        ],
        "temperature": (
            MODEL_REPEAT_TEMPERATURE
            if extended_repeat_budget
            else (0.15 if ROLE in {"left", "executive"} else 0.55)
        ),
        # Every worker returns a small structured artifact. Bounding output avoids a
        # queued local-model request consuming the entire orchestration time budget.
        "max_tokens": (
            MODEL_REPEAT_MAX_TOKENS
            if extended_repeat_budget
            else _priority_token_budget(req)
        ),
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
            distilled = _distill_compact_artifact(content, req, output_model)
            if distilled is None:
                distilled = _recover_interrupted_lobe_json(content, req, output_model)
            return output_model.model_validate(distilled) if distilled is not None else output_model.model_validate_json(content)
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
