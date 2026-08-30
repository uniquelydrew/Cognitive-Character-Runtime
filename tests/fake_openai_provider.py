"""A protocol-level OpenAI-compatible provider used only by integration tests.

The production graph always talks to a live model provider. This test process
lets the suite verify the worker's provider API and output contracts without
downloading model weights in CI.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI

app = FastAPI()


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": "test-model"}]}


def _answer(request: dict[str, Any]) -> str | None:
    question = str(request.get("user_input", "")).lower()
    identity = request["character"]["identity"]
    for needle, key in {
        "born": "birthplace",
        "birthplace": "birthplace",
        "hometown": "birthplace",
        "name": "name",
        "occupation": "occupation",
        "job": "occupation",
    }.items():
        if needle in question and key in identity:
            return str(identity[key])
    return None


@app.post("/v1/chat/completions")
def completions(body: dict[str, Any]) -> dict[str, Any]:
    request = json.loads(body["messages"][1]["content"])
    system_prompt = body["messages"][0]["content"]
    interaction = request.get("context", {}).get("interaction", {})

    if "analytic hemisphere" in system_prompt:
        result = {
            "topic": interaction.get("topic", "topic.general"),
            "fact_refs": [],
            "constraints": ["preserve_core"],
            "action": "answer",
            "confidence": 0.8,
        }
    elif "associative and social hemisphere" in system_prompt:
        result = {
            "action": "reclarify" if interaction.get("times_asked") else "inform",
            "affect": {"annoyance": 0.1, "curiosity": 0.5},
            "tone": "patient",
            "risk": "low",
            "association_keys": [],
        }
    elif "reflection mode" not in system_prompt:
        answer = interaction.get("prior_answer") or _answer(request) or "I do not have enough established information."
        repeated = interaction.get("interaction_type") == "repeated_question"
        speech = f"{answer}. You asked me that before." if repeated else answer
        result = {
            "goal": "maintain_continuity",
            "strategy": "reuse_prior_commitment" if interaction.get("prior_answer") else "answer_from_character_state",
            "speech": speech,
            "topic": interaction.get("topic", "topic.general"),
            "repeat_escalation": "hold",
            "mutations": [],
            "memory_writes": [{
                "kind": "self_history",
                "topic": interaction.get("topic", "topic.general"),
                "content": speech,
                "epistemic_type": "self_statement",
                "confidence": 1.0,
                "salience": 0.65,
            }],
        }
    else:
        transcript = request.get("transcript", [])
        source_ids = [event["event_id"] for event in transcript if event.get("event_id")]
        related_history = request.get("context", {}).get("related_history", {})
        link_mutations = []
        links = []
        for topic, previous_events in related_history.items():
            current = next((event for event in transcript if event.get("topic") == topic), None)
            if not current or not previous_events:
                continue
            link = {
                "from": previous_events[-1]["id"],
                "to": current["event_id"],
                "relationship": "revisits",
            }
            links.append({
                "source_event_id": link["from"],
                "target_event_id": link["to"],
                "relationship": link["relationship"],
            })
            link_mutations.append({
                "operation": "link_events",
                "target": topic,
                "value": link,
                "evidence": [link["from"], link["to"]],
                "confidence": 1.0,
                "epistemic_type": "observation",
                "reason": "Connect this interaction to earlier topic history.",
            })
        summary = "Interaction contained test-provider-reviewed turns."
        result = {
            "summary": summary,
            "related_event_ids": source_ids,
            "mutations": ([{
                "operation": "add_memory",
                "target": "interaction_summary",
                "value": summary,
                "evidence": source_ids,
                "confidence": 1.0,
                "epistemic_type": "observation",
                "reason": "Consolidate immutable interaction events.",
            }] if source_ids else []) + link_mutations,
            "links": links,
        }
    return {"choices": [{"message": {"content": json.dumps(result)}}]}
