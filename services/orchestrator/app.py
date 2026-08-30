from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from services.common import (
    CharacterDocument,
    CognitiveRequest,
    CognitiveResponse,
    EpistemicType,
    EventRecord,
    InteractionClassification,
    MemoryRecord,
    MutationOperation,
    MutationProposal,
    RepeatDynamics,
)

MEMORY_URL = os.getenv("MEMORY_URL", "http://memory:8000").rstrip("/")
LEFT_URL = os.getenv("LEFT_URL", "http://left-model:8000").rstrip("/")
RIGHT_URL = os.getenv("RIGHT_URL", "http://right-model:8000").rstrip("/")
EXEC_URL = os.getenv("EXEC_URL", "http://executive-model:8000").rstrip("/")
WORKER_REQUEST_TIMEOUT_SECONDS = float(os.getenv("WORKER_REQUEST_TIMEOUT_SECONDS", "165"))
CONTENT_TOKEN_RE = re.compile(r"[a-z0-9']+")
CONTENT_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "did", "does", "you", "your",
    "yours", "i", "me", "my", "what", "where", "when", "who", "why", "how", "again",
    "tell", "said", "say", "about", "to", "of", "in", "on", "please", "could", "would",
    "it", "that", "this", "and", "or", "for", "with", "have", "has", "had", "be", "been",
}
ANALYSIS_STOP_WORDS = {
    "answer", "biography", "character", "consistent", "constraint", "constraints", "established",
    "fact", "facts", "identity", "information", "observation", "observations", "recommend", "response",
    "share", "strategy", "subject", "use",
}

if WORKER_REQUEST_TIMEOUT_SECONDS <= 0:
    raise RuntimeError("WORKER_REQUEST_TIMEOUT_SECONDS must be positive")

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
    if any(x in lower for x in ("where were you born", "where are you from", "birthplace", "hometown", "birth town")):
        return "self.birthplace"
    if any(x in lower for x in ("what is your name", "what's your name", "who are you")):
        return "self.name"
    if any(x in lower for x in ("what do you do", "your job", "occupation", "your work", "work as")):
        return "self.occupation"

    kept = sorted(_content_tokens(lower))
    return "topic." + (".".join(kept[:8]) or "general")


def _content_tokens(text: str) -> set[str]:
    """Small, deterministic lexical signal used before executive adjudication."""

    tokens = CONTENT_TOKEN_RE.findall(text.lower())
    return {token for token in tokens if token not in CONTENT_STOP_WORDS and len(token) > 1}


def _token_similarity(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _model_topic(value: Any) -> str | None:
    topic = str(value or "").strip()
    if not topic or topic in {"topic.general", "stable topic identifier"}:
        return None
    return topic


def _analysis_anchor_tokens(analysis: dict[str, Any]) -> set[str]:
    """Extract fact-bearing terms from a lobe result without generic reasoning words."""

    values = [str(analysis.get("topic", "")), str(analysis.get("recommended_strategy", ""))]
    values.extend(str(item) for item in analysis.get("observations", []) if isinstance(item, str))
    values.extend(str(item) for item in analysis.get("associations", []) if isinstance(item, str))
    return {
        token
        for value in values
        for token in _content_tokens(value)
        if token not in ANALYSIS_STOP_WORDS
    }


def executive_repeat_review(
    *,
    message: str,
    topic: str,
    current_event_id: str,
    session_events: list[dict[str, Any]],
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    prior_times: int,
) -> dict[str, Any]:
    """Review repeat evidence after both hemisphere analyses and before executive speech.

    This is deliberately a deterministic *candidate* review, not a replacement for the
    executive model. It supplies recent wording, lobe-topic agreement, and a confidence
    signal so the executive can recognize semantic rephrases without receiving an
    unbounded transcript on every turn.
    """

    prior_events = [event for event in session_events if event.get("id") != current_event_id]
    prior_users = [event for event in prior_events if event.get("event_type") == "user_message"]
    replies_by_user = {
        str(event.get("metadata", {}).get("responds_to")): event
        for event in prior_events
        if event.get("event_type") == "character_message" and event.get("metadata", {}).get("responds_to")
    }
    current_left_topic = _model_topic(left_result.get("topic"))
    current_associations = " ".join(str(item) for item in right_result.get("associations", []))

    best: dict[str, Any] | None = None
    for user_event in prior_users:
        score = 0.0
        reason = ""
        event_topic = str(user_event.get("topic") or "")
        if event_topic and event_topic == topic:
            score, reason = 0.98, "same normalized subject"

        lexical_score = _token_similarity(message, str(user_event.get("content", "")))
        if lexical_score > score:
            score, reason = lexical_score, "overlapping subject language"

        prior_reply = replies_by_user.get(str(user_event.get("id")))
        prior_review = (prior_reply or {}).get("metadata", {}).get("repeat_review", {})
        if not isinstance(prior_review, dict):
            prior_review = {}
        inherited_subject_key = str(prior_review.get("subject_key") or event_topic or topic)
        prior_left_topic = _model_topic((prior_reply or {}).get("metadata", {}).get("left", {}).get("topic"))
        if current_left_topic and prior_left_topic and current_left_topic == prior_left_topic and score < 0.84:
            score, reason = 0.84, "left analyses selected the same subject"

        if prior_reply:
            current_anchors = _analysis_anchor_tokens(left_result)
            prior_anchors = _analysis_anchor_tokens(prior_reply.get("metadata", {}).get("left", {}))
            shared_anchors = current_anchors & prior_anchors
            if len(shared_anchors) >= 2 and score < 0.76:
                score, reason = 0.76, "left analyses share subject-specific fact anchors"

        if current_associations and prior_reply:
            prior_associations = " ".join(
                str(item) for item in prior_reply.get("metadata", {}).get("right", {}).get("associations", [])
            )
            association_score = _token_similarity(current_associations, prior_associations)
            if association_score >= 0.5 and score < 0.66:
                score, reason = 0.66, "right analyses share the same association"

        candidate = {
            "event_id": user_event.get("id"),
            # Carry forward an earlier semantic resolution when this turn itself
            # was a paraphrase. That prevents a chain of new phrasings from
            # fragmenting the durable subject-specific gauge into lexical keys.
            "topic": inherited_subject_key,
            "answer": (prior_reply or {}).get("content"),
            "score": round(score, 3),
            "reason": reason,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    semantic_repeat = bool(best and best["score"] >= 0.56)
    if not semantic_repeat and prior_times > 0:
        semantic_repeat = True
        best = {
            "event_id": None,
            "topic": topic,
            "answer": None,
            "score": 0.9,
            "reason": "established same-subject history",
        }

    subject_key = str((best or {}).get("topic") or topic)
    consecutive_repeats = 1
    if semantic_repeat:
        for prior_user in reversed(prior_users):
            same_subject = str(prior_user.get("topic") or "") == subject_key
            similar_wording = _token_similarity(message, str(prior_user.get("content", ""))) >= 0.56
            if same_subject or similar_wording:
                consecutive_repeats += 1
            else:
                break

    recent_turns = [
        {
            "event_id": event.get("id"),
            "event_type": event.get("event_type"),
            "actor": event.get("actor"),
            "content": event.get("content"),
            "topic": event.get("topic"),
        }
        for event in prior_events[-8:]
        if event.get("event_type") in {"user_message", "character_message"}
    ]
    return {
        "semantic_repeat_candidate": semantic_repeat,
        "subject_key": subject_key,
        "matched_event_id": (best or {}).get("event_id"),
        "matched_answer": (best or {}).get("answer"),
        "confidence": float((best or {}).get("score", 0.0)),
        "reason": (best or {}).get("reason", "no prior semantic match"),
        "consecutive_repeats": consecutive_repeats,
        "recent_turns": recent_turns,
    }


def derive_repeat_dynamics(
    *,
    character: CharacterDocument,
    mutable_state: dict[str, Any],
    review: dict[str, Any],
    user_turn_count: int,
) -> tuple[RepeatDynamics, dict[str, float], bool]:
    """Intersect conversation patience with durable, subject-specific defensiveness."""

    raw_topics = mutable_state.get("topic_defensiveness", {})
    if not isinstance(raw_topics, dict):
        raw_topics = {}
    topic_defensiveness = {
        str(key): _clamp(float(value))
        for key, value in raw_topics.items()
        if isinstance(value, (int, float))
    }
    subject_key = str(review["subject_key"])
    prior_defensiveness = topic_defensiveness.get(subject_key, 0.0)
    semantic_repeat = bool(review["semantic_repeat_candidate"])
    consecutive_repeats = int(review["consecutive_repeats"])

    if semantic_repeat:
        added_pressure = 0.24 + (0.04 * min(max(consecutive_repeats - 2, 0), 3))
        subject_defensiveness = _clamp((prior_defensiveness * 0.96) + added_pressure)
    elif prior_defensiveness:
        # Returning to a charged subject can cool it slowly, but never erase it merely
        # because the user changed wording or briefly moved to another subject.
        subject_defensiveness = _clamp(prior_defensiveness * 0.97)
    else:
        subject_defensiveness = 0.0

    updated_topics = dict(topic_defensiveness)
    changed = abs(subject_defensiveness - prior_defensiveness) >= 0.001
    if changed:
        updated_topics[subject_key] = round(subject_defensiveness, 4)

    trait_patience = float(character.traits.get("patient", 0.5))
    trait_irritability = float(character.traits.get("irritable", 0.5))
    baseline_patience = _clamp(0.72 + (0.20 * trait_patience) - (0.10 * trait_irritability))
    conversation_drain = 0.025 * max(user_turn_count - 1, 0)
    repetition_drain = 0.115 * max(consecutive_repeats - 1, 0)
    conversation_patience = _clamp(baseline_patience - conversation_drain - repetition_drain)
    intersection_pressure = _clamp(subject_defensiveness * (1.0 - conversation_patience))

    if intersection_pressure >= 0.36:
        posture = "defensive"
    elif intersection_pressure >= 0.16:
        posture = "confused"
    elif semantic_repeat:
        posture = "reclarify"
    else:
        posture = "normal"

    return (
        RepeatDynamics(
            conversation_patience=round(conversation_patience, 4),
            subject_defensiveness=round(subject_defensiveness, 4),
            intersection_pressure=round(intersection_pressure, 4),
            response_posture=posture,
            semantic_repeat=semantic_repeat,
            consecutive_repeats=consecutive_repeats,
            subject_key=subject_key,
            review_confidence=round(float(review["confidence"]), 4),
        ),
        updated_topics,
        changed,
    )


def apply_repeat_posture(speech: str, dynamics: RepeatDynamics) -> str:
    """Make the executive's deterministic posture visible when a small model ignores it.

    The executive still authors the factual reply. This narrow delivery guard only
    adds the already-derived relational boundary, so a valid repeat signal cannot
    silently turn into another identical, endlessly patient response.
    """

    if dynamics.response_posture == "confused":
        return f"{speech.rstrip()} I'm a little confused—we've just covered that."
    if dynamics.response_posture == "defensive":
        return f"{speech.rstrip()} I've already answered that. Please don't keep pressing the same question."
    return speech


async def get_json(client: httpx.AsyncClient, url: str) -> Any:
    try:
        r = await client.get(url)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "A required service did not respond in time. Please retry.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "A required service is unavailable. Please retry shortly.") from exc
    if r.status_code >= 400:
        raise HTTPException(r.status_code, _upstream_detail(r))
    return r.json()


async def post_json(client: httpx.AsyncClient, url: str, payload: Any) -> Any:
    try:
        r = await client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "The character is taking longer than expected. Please retry.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "A required service is unavailable. Please retry shortly.") from exc
    if r.status_code >= 400:
        raise HTTPException(r.status_code, _upstream_detail(r))
    return r.json()


async def put_json(client: httpx.AsyncClient, url: str, payload: Any) -> Any:
    try:
        r = await client.put(url, json=payload)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "The profile update took too long. Please retry.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "A required service is unavailable. Please retry shortly.") from exc
    if r.status_code >= 400:
        raise HTTPException(r.status_code, _upstream_detail(r))
    return r.json()


def _upstream_detail(response: httpx.Response) -> str:
    """Preserve a worker's safe error message instead of leaking JSON as a string."""

    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str) and detail:
            return detail
    except (ValueError, TypeError):
        pass
    return response.text or "The request could not be completed."


async def load_character(client: httpx.AsyncClient, character_id: str) -> tuple[CharacterDocument, dict[str, Any]]:
    state = await get_json(client, f"{MEMORY_URL}/characters/{character_id}")
    char_payload = dict(state["character"])
    # Mutable state and beliefs are supplied separately to models; immutable identity remains in the primer.
    return CharacterDocument.model_validate(char_payload), state


async def infer(
    client: httpx.AsyncClient,
    base_url: str,
    req: CognitiveRequest,
    expected_role: str,
) -> dict[str, Any]:
    data = await post_json(client, f"{base_url}/infer", req.model_dump(mode="json"))
    try:
        response = CognitiveResponse.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(502, f"{expected_role} worker returned an invalid response envelope: {exc}") from exc
    if response.role != expected_role:
        raise HTTPException(502, f"Expected {expected_role} worker response, received {response.role!r}")
    return response.result


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


@app.get("/profiles")
async def profiles() -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/profiles")


@app.get("/profiles/{character_id}")
async def profile(character_id: str) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/profiles/{character_id}")


@app.post("/profiles", status_code=201)
async def create_profile(profile: CharacterDocument) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await post_json(client, f"{MEMORY_URL}/profiles", profile.model_dump(mode="json"))


@app.put("/profiles/{character_id}")
async def update_profile(character_id: str, profile: CharacterDocument) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await put_json(
            client,
            f"{MEMORY_URL}/profiles/{character_id}",
            profile.model_dump(mode="json"),
        )


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

    async with httpx.AsyncClient(timeout=httpx.Timeout(WORKER_REQUEST_TIMEOUT_SECONDS, connect=10)) as client:
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

        left_task = infer(client, LEFT_URL, cognition_req, "left")
        right_task = infer(client, RIGHT_URL, cognition_req, "right")
        left_result, right_result = await asyncio.gather(left_task, right_task)

        # The lobes are intentionally free to arrive at overlapping thoughts. The
        # executive receives a separate, bounded review after both are complete so
        # it can recognize a rephrased repeat without making either lobe suppress
        # its own reasoning.
        transcript = await get_json(client, f"{MEMORY_URL}/sessions/{session_id}/events")
        repeat_review = executive_repeat_review(
            message=req.message,
            topic=topic,
            current_event_id=user_event.id or "",
            session_events=transcript,
            left_result=left_result,
            right_result=right_result,
            prior_times=prior_times,
        )
        user_turn_count = sum(1 for event in transcript if event.get("event_type") == "user_message")
        repeat_dynamics, updated_topic_defensiveness, topic_state_changed = derive_repeat_dynamics(
            character=character,
            mutable_state=state.get("mutable_state", {}),
            review=repeat_review,
            user_turn_count=user_turn_count,
        )

        related_event_ids = list(classification.related_event_ids)
        if repeat_dynamics.semantic_repeat and repeat_review.get("matched_event_id"):
            related_event_ids.append(str(repeat_review["matched_event_id"]))
        interaction_type = classification.interaction_type
        if repeat_dynamics.semantic_repeat:
            interaction_type = (
                "repeated_question"
                if repeat_dynamics.subject_key == topic
                else "paraphrase"
            )
        classification = classification.model_copy(
            update={
                "interaction_type": interaction_type,
                "prior_answer": (
                    repeat_review.get("matched_answer")
                    if repeat_dynamics.semantic_repeat
                    else classification.prior_answer
                ),
                "times_asked": max(classification.times_asked, repeat_dynamics.consecutive_repeats),
                "related_event_ids": list(dict.fromkeys(related_event_ids)),
                "repeat_dynamics": repeat_dynamics,
            }
        )

        dynamics_mutation_results = []
        if topic_state_changed:
            dynamics_proposal = MutationProposal(
                operation=MutationOperation.SET_MUTABLE_STATE,
                target="topic_defensiveness",
                value=updated_topic_defensiveness,
                old_value=state.get("mutable_state", {}).get("topic_defensiveness", {}),
                evidence=[user_event.id or ""],
                confidence=1.0,
                epistemic_type=EpistemicType.SUSPICION,
                reason=(
                    "Persist the deterministic, subject-specific defensiveness signal "
                    "derived from this user turn and repeat review."
                ),
            )
            dynamics_mutation_results = await post_json(
                client,
                f"{MEMORY_URL}/mutations/{character_id}",
                {"proposals": [dynamics_proposal.model_dump(mode="json")]},
            )

        executive_mutable_state = dict(state.get("mutable_state", {}))
        executive_mutable_state["topic_defensiveness"] = updated_topic_defensiveness
        executive_context = {
            **common_context,
            "interaction": classification.model_dump(mode="json"),
            "conversation_dynamics": repeat_dynamics.model_dump(mode="json"),
            "executive_repeat_review": repeat_review,
            "mutable_state": executive_mutable_state,
        }
        exec_req = cognition_req.model_copy(
            update={
                "context": executive_context,
                "left_result": left_result,
                "right_result": right_result,
            }
        )
        executive = await infer(client, EXEC_URL, exec_req, "executive")
        speech = str(executive.get("speech", "")).strip()
        if not speech:
            raise HTTPException(502, "Executive produced no speech")
        speech = apply_repeat_posture(speech, repeat_dynamics)
        executive = {**executive, "speech": speech}

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
                "interaction": classification.model_dump(mode="json"),
                "repeat_review": repeat_review,
            },
        )
        character_event = EventRecord.model_validate(
            await post_json(client, f"{MEMORY_URL}/events", character_event.model_dump(mode="json"))
        )

        # Turn-level self-history is an infrastructure invariant, not a best-effort model suggestion.
        # Reflection handles higher-order consolidation later.
        stored_memories = [
            await post_json(
                client,
                f"{MEMORY_URL}/memories",
                MemoryRecord(
                    character_id=character_id,
                    kind="self_history",
                    topic=str(executive.get("topic") or topic),
                    content=speech,
                    epistemic_type="self_statement",
                    confidence=1.0,
                    salience=0.65,
                    source_event_ids=[user_event.id or "", character_event.id or ""],
                    metadata={"session_id": session_id},
                ).model_dump(mode="json"),
            )
        ]
        memory_writes = executive.get("memory_writes", [])
        for write in memory_writes:
            write_topic = str(write.get("topic") or topic)
            if (
                write.get("kind") == "self_history"
                and write_topic == str(executive.get("topic") or topic)
                and str(write.get("content", "")) == speech
            ):
                continue
            record = MemoryRecord(
                character_id=character_id,
                kind=str(write.get("kind", "self_history")),
                topic=write_topic,
                content=str(write.get("content", speech)),
                epistemic_type=str(write.get("epistemic_type", "self_statement")),
                confidence=float(write.get("confidence", 1.0)),
                salience=float(write.get("salience", 0.5)),
                source_event_ids=[user_event.id or "", character_event.id or ""],
                metadata={"session_id": session_id},
            )
            stored_memories.append(await post_json(client, f"{MEMORY_URL}/memories", record.model_dump(mode="json")))

        immediate = executive.get("mutations", [])
        mutation_results = list(dynamics_mutation_results)
        if immediate:
            proposals = [MutationProposal.model_validate(p).model_dump(mode="json") for p in immediate]
            mutation_results.extend(
                await post_json(
                    client,
                    f"{MEMORY_URL}/mutations/{character_id}",
                    {"proposals": proposals},
                )
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
    async with httpx.AsyncClient(timeout=httpx.Timeout(WORKER_REQUEST_TIMEOUT_SECONDS, connect=10)) as client:
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
        executive = await infer(client, EXEC_URL, reflection_req, "executive")
        summary = str(executive.get("summary", "")).strip()
        if not summary:
            raise HTTPException(502, "Executive produced no reflection summary")

        # A reflection summary is durable, sourced memory even when the model has
        # no additional insight to propose. Route it through the same validator/audit
        # path as any model-authored mutation.
        summary_proposal = MutationProposal(
            operation="add_memory",
            target="interaction_summary",
            value=summary,
            evidence=[str(e["id"]) for e in transcript if e.get("id")],
            confidence=1.0,
            epistemic_type="observation",
            reason="Persist the completed interaction summary with raw-event provenance.",
        ).model_dump(mode="json")
        proposals_raw = [
            p for p in executive.get("mutations", [])
            if not (
                p.get("operation") == "add_memory"
                and p.get("target") == "interaction_summary"
                and str(p.get("value", "")) == summary
            )
        ]
        proposals_raw.insert(0, summary_proposal)
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
            content=summary,
            metadata={"executive": executive, "mutation_results": mutation_results},
        )
        await post_json(client, f"{MEMORY_URL}/events", reflection_event.model_dump(mode="json"))

        return ReflectionResult(
            session_id=session_id,
            summary=summary,
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
