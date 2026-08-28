from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.common import (
    CharacterDocument,
    CognitiveRequest,
    EventRecord,
    InteractionClassification,
    MemoryRecord,
    MutationProposal,
)

MEMORY_URL = os.getenv("MEMORY_URL", "http://memory:8000").rstrip("/")
LEFT_URL = os.getenv("LEFT_URL", "http://left-model:8000").rstrip("/")
RIGHT_URL = os.getenv("RIGHT_URL", "http://right-model:8000").rstrip("/")
EXEC_URL = os.getenv("EXEC_URL", "http://executive-model:8000").rstrip("/")

app = FastAPI(title="Cognitive Character Orchestrator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionCreate(BaseModel):
    character_id: str


class ChatRequest(BaseModel):
    message: str


class ReflectionResult(BaseModel):
    session_id: str
    summary: str
    mutation_results: list[dict[str, Any]]
    executive: dict[str, Any]


def normalize_topic(text: str) -> str:
    """Cheap deterministic topic key for the bootstrap implementation.

    This intentionally does not pretend to be semantic retrieval. A later milestone
    replaces this with embedding/classifier-assisted topic resolution while preserving
    the same memory-service interface.
    """
    lower = text.lower().strip()
    if any(x in lower for x in ("where were you born", "where are you from", "birthplace", "hometown")):
        return "self.birthplace"
    if any(x in lower for x in ("what is your name", "what's your name", "who are you")):
        return "self.name"
    if any(x in lower for x in ("what do you do", "your job", "occupation")):
        return "self.occupation"

    tokens = re.findall(r"[a-z0-9']+", lower)
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "do", "did", "does", "you", "your",
        "yours", "i", "me", "my", "what", "where", "when", "who", "why", "how", "again",
        "tell", "said", "say", "about", "to", "of", "in", "on", "please", "could", "would",
    }
    kept = [t for t in tokens if t not in stop]
    return "topic." + (".".join(kept[:8]) or "general")


async def get_json(client: httpx.AsyncClient, url: str) -> Any:
    r = await client.get(url)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


async def post_json(client: httpx.AsyncClient, url: str, payload: Any) -> Any:
    r = await client.post(url, json=payload)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


async def load_character(client: httpx.AsyncClient, character_id: str) -> tuple[CharacterDocument, dict[str, Any]]:
    state = await get_json(client, f"{MEMORY_URL}/characters/{character_id}")
    char_payload = dict(state["character"])
    # Mutable state and beliefs are supplied separately to models; immutable identity remains in the primer.
    return CharacterDocument.model_validate(char_payload), state


async def infer(client: httpx.AsyncClient, base_url: str, req: CognitiveRequest) -> dict[str, Any]:
    data = await post_json(client, f"{base_url}/infer", req.model_dump(mode="json"))
    return data["result"]


@app.get("/health")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10) as client:
        checks = await asyncio.gather(
            get_json(client, f"{MEMORY_URL}/health"),
            get_json(client, f"{LEFT_URL}/health"),
            get_json(client, f"{RIGHT_URL}/health"),
            get_json(client, f"{EXEC_URL}/health"),
            return_exceptions=True,
        )
    return {
        "status": "ok" if all(not isinstance(x, Exception) for x in checks) else "degraded",
        "dependencies": [str(x) if isinstance(x, Exception) else x for x in checks],
    }


@app.get("/characters")
async def characters() -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/characters")


@app.get("/characters/{character_id}/state")
async def character_state(character_id: str) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/characters/{character_id}")


@app.post("/sessions")
async def create_session(req: SessionCreate) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await post_json(client, f"{MEMORY_URL}/sessions", req.model_dump())


@app.get("/sessions/{session_id}/events")
async def session_events(session_id: str) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/sessions/{session_id}/events")


@app.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest) -> dict[str, Any]:
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    async with httpx.AsyncClient(timeout=120) as client:
        # Resolve the session from its event stream if it already has events. For a new session,
        # the memory service doesn't expose session metadata yet, so resolve through a small helper call.
        # We store character_id in each event and also accept a private metadata route via SQLite service.
        session_meta = await _session_meta(client, session_id)
        if session_meta.get("status") != "open":
            raise HTTPException(409, "Interaction is already closed")
        character_id = session_meta["character_id"]
        character, state = await load_character(client, character_id)

        topic = normalize_topic(req.message)
        history = await get_json(client, f"{MEMORY_URL}/interaction-history/{character_id}?topic={topic}&limit=30")
        prior_times = int(history.get("times_asked", 0))
        prior_answer = history.get("prior_answer")
        classification = InteractionClassification(
            interaction_type="repeated_question" if prior_times > 0 else "new_subject",
            topic=topic,
            prior_answer=prior_answer,
            times_asked=prior_times + 1,
            related_event_ids=[e["id"] for e in history.get("events", [])],
        )

        user_event = EventRecord(
            character_id=character_id,
            session_id=session_id,
            event_type="user_message",
            actor="user",
            content=req.message,
            topic=topic,
            metadata={"interaction": classification.model_dump(mode="json")},
        )
        user_event = EventRecord.model_validate(
            await post_json(client, f"{MEMORY_URL}/events", user_event.model_dump(mode="json"))
        )

        memories = await get_json(client, f"{MEMORY_URL}/memories/{character_id}?topic={topic}&limit=20")
        common_context = {
            "interaction": classification.model_dump(mode="json"),
            "memories": memories,
            "mutable_state": state.get("mutable_state", {}),
            "beliefs": state.get("beliefs", {}),
            "goals": state.get("goals", []),
        }
        cognition_req = CognitiveRequest(character=character, user_input=req.message, context=common_context)

        left_task = infer(client, LEFT_URL, cognition_req)
        right_task = infer(client, RIGHT_URL, cognition_req)
        left_result, right_result = await asyncio.gather(left_task, right_task)

        exec_req = cognition_req.model_copy(update={"left_result": left_result, "right_result": right_result})
        executive = await infer(client, EXEC_URL, exec_req)
        speech = str(executive.get("speech", "")).strip()
        if not speech:
            raise HTTPException(502, "Executive produced no speech")

        character_event = EventRecord(
            character_id=character_id,
            session_id=session_id,
            event_type="character_message",
            actor="character",
            content=speech,
            topic=str(executive.get("topic") or topic),
            metadata={
                "left": left_result,
                "right": right_result,
                "executive": executive,
                "responds_to": user_event.id,
            },
        )
        character_event = EventRecord.model_validate(
            await post_json(client, f"{MEMORY_URL}/events", character_event.model_dump(mode="json"))
        )

        # Turn-level self-history is append-only. Reflection handles higher-order consolidation later.
        memory_writes = executive.get("memory_writes", [])
        stored_memories = []
        for write in memory_writes:
            record = MemoryRecord(
                character_id=character_id,
                kind=str(write.get("kind", "self_history")),
                topic=str(write.get("topic") or topic),
                content=str(write.get("content", speech)),
                epistemic_type=str(write.get("epistemic_type", "self_statement")),
                confidence=float(write.get("confidence", 1.0)),
                salience=float(write.get("salience", 0.5)),
                source_event_ids=[user_event.id or "", character_event.id or ""],
                metadata={"session_id": session_id},
            )
            stored_memories.append(await post_json(client, f"{MEMORY_URL}/memories", record.model_dump(mode="json")))

        immediate = executive.get("mutations", [])
        mutation_results = []
        if immediate:
            proposals = [MutationProposal.model_validate(p).model_dump(mode="json") for p in immediate]
            mutation_results = await post_json(
                client,
                f"{MEMORY_URL}/mutations/{character_id}",
                {"proposals": proposals},
            )

        return {
            "session_id": session_id,
            "character_id": character_id,
            "message": speech,
            "interaction": classification.model_dump(mode="json"),
            "cognition": {"left": left_result, "right": right_result, "executive": executive},
            "memory_writes": stored_memories,
            "mutation_results": mutation_results,
        }


async def _session_meta(client: httpx.AsyncClient, session_id: str) -> dict[str, Any]:
    # Endpoint deliberately kept on memory service to avoid orchestrator-side session state.
    return await get_json(client, f"{MEMORY_URL}/sessions/{session_id}")


@app.post("/sessions/{session_id}/reflect", response_model=ReflectionResult)
async def reflect(session_id: str) -> ReflectionResult:
    async with httpx.AsyncClient(timeout=120) as client:
        session_meta = await _session_meta(client, session_id)
        character_id = session_meta["character_id"]
        character, state = await load_character(client, character_id)
        transcript = await get_json(client, f"{MEMORY_URL}/sessions/{session_id}/events")

        # Reflection is idempotent until another conversational event is added. This prevents
        # repeated manual reflection/close operations from duplicating derived memory.
        last_reflection_index = max(
            (i for i, e in enumerate(transcript) if e["event_type"] == "reflection"),
            default=-1,
        )
        last_conversation_index = max(
            (i for i, e in enumerate(transcript) if e["event_type"] in {"user_message", "character_message"}),
            default=-1,
        )
        if last_reflection_index > last_conversation_index:
            prior = transcript[last_reflection_index]
            metadata = prior.get("metadata", {})
            return ReflectionResult(
                session_id=session_id,
                summary=prior.get("content", ""),
                mutation_results=metadata.get("mutation_results", []),
                executive=metadata.get("executive", {}),
            )

        memories = await get_json(client, f"{MEMORY_URL}/memories/{character_id}?limit=50")
        current_ids = {e["id"] for e in transcript}
        topics = sorted({e.get("topic") for e in transcript if e.get("topic")})
        related_history: dict[str, list[dict[str, Any]]] = {}
        for topic in topics:
            history = await get_json(
                client,
                f"{MEMORY_URL}/interaction-history/{character_id}?topic={topic}&limit=50",
            )
            prior = [e for e in history.get("events", []) if e["id"] not in current_ids]
            if prior:
                related_history[str(topic)] = prior

        reflection_req = CognitiveRequest(
            character=character,
            mode="reflection",
            transcript=[
                {
                    "event_id": e["id"],
                    "event_type": e["event_type"],
                    "actor": e["actor"],
                    "content": e["content"],
                    "topic": e.get("topic"),
                }
                for e in transcript
            ],
            context={
                "memories": memories,
                "mutable_state": state.get("mutable_state", {}),
                "beliefs": state.get("beliefs", {}),
                "goals": state.get("goals", []),
                "related_history": related_history,
            },
        )
        executive = await infer(client, EXEC_URL, reflection_req)
        proposals_raw = executive.get("mutations", [])
        proposals = [MutationProposal.model_validate(p).model_dump(mode="json") for p in proposals_raw]
        mutation_results = []
        if proposals:
            mutation_results = await post_json(
                client,
                f"{MEMORY_URL}/mutations/{character_id}",
                {"proposals": proposals},
            )

        reflection_event = EventRecord(
            character_id=character_id,
            session_id=session_id,
            event_type="reflection",
            actor="executive",
            content=str(executive.get("summary", "Interaction reflected.")),
            metadata={"executive": executive, "mutation_results": mutation_results},
        )
        await post_json(client, f"{MEMORY_URL}/events", reflection_event.model_dump(mode="json"))

        return ReflectionResult(
            session_id=session_id,
            summary=str(executive.get("summary", "")),
            mutation_results=mutation_results,
            executive=executive,
        )


@app.post("/sessions/{session_id}/close")
async def close(session_id: str) -> dict[str, Any]:
    reflection = await reflect(session_id)
    async with httpx.AsyncClient(timeout=20) as client:
        closed = await post_json(client, f"{MEMORY_URL}/sessions/{session_id}/close", {})
    return {"session": closed, "reflection": reflection.model_dump(mode="json")}


@app.get("/debug/{character_id}")
async def debug(character_id: str) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/debug/{character_id}")
