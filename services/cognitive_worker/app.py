from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

from services.common import CognitiveRequest, CognitiveResponse

ROLE = os.getenv("COGNITIVE_ROLE", "left").lower()
BACKEND = os.getenv("WORKER_BACKEND", "mock").lower()
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "local-model")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "unused")

app = FastAPI(title=f"Cognitive Worker: {ROLE}", version="0.1.0")


def _name(req: CognitiveRequest) -> str:
    return str(req.character.identity.get("name", req.character.id))


def _topic_key(text: str) -> str:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "do", "did", "does", "you",
        "your", "yours", "i", "me", "my", "what", "where", "when", "who", "why",
        "how", "again", "tell", "said", "say", "about", "to", "of", "in", "on",
    }
    kept = [t for t in tokens if t not in stop]
    return ".".join(kept[:8]) or "general"


def _mock_left(req: CognitiveRequest) -> dict[str, Any]:
    classification = req.context.get("interaction", {})
    relevant = req.context.get("memories", [])
    facts = [
        m for m in relevant
        if m.get("epistemic_type") in {"fact", "observation", "self_statement", "belief"}
    ]
    return {
        "topic": classification.get("topic") or _topic_key(req.user_input),
        "observations": [m.get("content") for m in facts[:5]],
        "consistency_constraints": [
            m.get("content") for m in relevant if m.get("kind") == "self_history"
        ][:5],
        "recommended_strategy": "answer_consistently" if classification.get("prior_answer") else "answer_from_known_state",
        "confidence": 0.86 if facts else 0.55,
    }


def _mock_right(req: CognitiveRequest) -> dict[str, Any]:
    classification = req.context.get("interaction", {})
    traits = req.character.traits
    irritability = float(traits.get("irritable", 0.2))
    patience = float(traits.get("patient", 0.5))
    repeat_count = int(classification.get("times_asked", 0))
    annoyance = min(1.0, max(0.0, irritability * repeat_count - patience * 0.25))
    return {
        "social_read": "repetition_noticed" if repeat_count else "ordinary_exchange",
        "affect": {
            "annoyance": round(annoyance, 3),
            "curiosity": 0.35 if repeat_count else 0.5,
        },
        "recommended_tone": "pointed" if annoyance > 0.55 else "patient",
        "associations": [m.get("content") for m in req.context.get("memories", [])[:4]],
    }


def _canonical_answer(req: CognitiveRequest) -> str | None:
    q = req.user_input.lower()
    identity = req.character.identity
    mappings = {
        "born": "birthplace",
        "birthplace": "birthplace",
        "name": "name",
        "occupation": "occupation",
        "job": "occupation",
    }
    for needle, key in mappings.items():
        if needle in q and key in identity:
            return str(identity[key])
    return None


def _mock_executive_turn(req: CognitiveRequest) -> dict[str, Any]:
    interaction = req.context.get("interaction", {})
    prior_answer = interaction.get("prior_answer")
    answer = prior_answer or _canonical_answer(req)
    name = _name(req)
    repeated = interaction.get("interaction_type") in {"repeated_question", "paraphrase"}
    count = int(interaction.get("times_asked", 0))
    right = req.right_result or {}
    annoyance = float(right.get("affect", {}).get("annoyance", 0.0))

    if answer:
        if repeated and annoyance > 0.55:
            speech = f"{answer}. You've asked me that {max(2, count)} times now."
        elif repeated:
            speech = f"{answer}. You asked me that before."
        else:
            speech = answer
    else:
        biography = req.character.biography.strip()
        if biography:
            speech = f"From what I know of myself: {biography.splitlines()[0]}"
        else:
            speech = f"I'm {name}. I don't have enough established information to answer that without inventing it."

    topic = interaction.get("topic") or _topic_key(req.user_input)
    return {
        "goal": "maintain_continuity",
        "strategy": "reuse_prior_commitment" if prior_answer else "answer_from_character_state",
        "speech": speech,
        "topic": topic,
        "mutations": [],
        "memory_writes": [
            {
                "kind": "self_history",
                "topic": topic,
                "content": speech,
                "epistemic_type": "self_statement",
                "confidence": 1.0,
                "salience": 0.65,
            }
        ],
    }


def _mock_reflection(req: CognitiveRequest) -> dict[str, Any]:
    transcript = req.transcript
    source_ids = [str(t.get("event_id")) for t in transcript if t.get("event_id")]
    user_turns = [t for t in transcript if t.get("actor") == "user"]
    assistant_turns = [t for t in transcript if t.get("actor") == "character"]
    related_history = req.context.get("related_history", {})
    summary = (
        f"Interaction contained {len(user_turns)} user turns and {len(assistant_turns)} character turns."
    )
    proposals: list[dict[str, Any]] = []
    if source_ids:
        proposals.append({
            "operation": "add_memory",
            "target": "interaction_summary",
            "value": summary,
            "evidence": source_ids,
            "confidence": 1.0,
            "epistemic_type": "observation",
            "reason": "Consolidate completed interaction without rewriting source events.",
        })

    links: list[dict[str, Any]] = []
    current_by_topic: dict[str, list[str]] = {}
    for event in transcript:
        topic = event.get("topic")
        event_id = event.get("event_id")
        if topic and event_id:
            current_by_topic.setdefault(str(topic), []).append(str(event_id))

    for topic, prior_events in related_history.items():
        current_ids = current_by_topic.get(topic, [])
        if not current_ids or not prior_events:
            continue
        prior_id = str(prior_events[-1]["id"])
        current_id = current_ids[0]
        link = {"from": prior_id, "to": current_id, "relationship": "revisits"}
        links.append(link)
        proposals.append({
            "operation": "link_events",
            "target": topic,
            "value": link,
            "evidence": [prior_id, current_id],
            "confidence": 1.0,
            "epistemic_type": "observation",
            "reason": "Connect this interaction to earlier history on the same resolved topic.",
        })

    return {
        "summary": summary,
        "related_event_ids": source_ids,
        "mutations": proposals,
        "links": links,
    }


def _system_prompt() -> str:
    if ROLE == "left":
        return (
            "You are the analytic hemisphere of a persistent fictional character. "
            "Return only JSON. Focus on facts, consistency, causal reasoning, constraints, and plans. "
            "Never invent canonical facts. Treat supplied memories according to their epistemic type."
        )
    if ROLE == "right":
        return (
            "You are the associative/social hemisphere of a persistent fictional character. "
            "Return only JSON. Focus on affect, social interpretation, associations, subtext, and tone. "
            "Do not mutate memory or canonical facts."
        )
    return (
        "You are the executive function of a persistent fictional character. Return only JSON. "
        "Resolve analytic and associative recommendations, maintain self-continuity, and propose typed mutations. "
        "Never silently rewrite raw history or immutable core biography. In reflection mode, connect events and "
        "propose evidence-backed changes while preserving provenance."
    )


async def _openai_compatible(req: CognitiveRequest) -> dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": req.model_dump_json()},
        ],
        "temperature": 0.1 if ROLE in {"left", "executive"} else 0.65,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{MODEL_BASE_URL}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "role": ROLE, "backend": BACKEND, "model": MODEL_NAME}


@app.post("/infer", response_model=CognitiveResponse)
async def infer(req: CognitiveRequest) -> CognitiveResponse:
    if BACKEND == "openai_compatible":
        result = await _openai_compatible(req)
    elif ROLE == "left":
        result = _mock_left(req)
    elif ROLE == "right":
        result = _mock_right(req)
    elif ROLE == "executive" and req.mode == "reflection":
        result = _mock_reflection(req)
    else:
        result = _mock_executive_turn(req)
    return CognitiveResponse(role=ROLE, result=result)
