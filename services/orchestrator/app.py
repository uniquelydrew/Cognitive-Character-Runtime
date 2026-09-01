from __future__ import annotations

import asyncio
import hmac
import os
import uuid
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services.common import (
    CharacterDocument,
    CognitiveRequest,
    InteractionClassification,
)
from services.orchestrator.cognitive_policy import (
    CONTENT_TOKEN_RE,
    content_tokens as _content_tokens,
    normalize_topic,
    resolve_cognitive_priorities,
    token_similarity as _token_similarity,
    weighted_arbitration_plan,
)
from services.orchestrator.cognitive_policy import (
    bounded_lobe_transcript as _bounded_lobe_transcript,
    semantic_repeat_evidence as _semantic_repeat_evidence,
)
from services.orchestrator.inference import (
    infer as run_inference,
    infer_timed as run_timed_inference,
    probe_worker as probe_inference_worker,
)
from services.orchestrator.locking import SessionLockRegistry
from services.orchestrator.repeat_dynamics import (
    derive_repeat_dynamics,
    question_signature as _question_signature,
    repeat_intent_fallback,
    response_substantially_repeats_prior_answer,
    response_substantially_repeats_recent_answers,
)
from services.orchestrator.reflection import reflect_session
from services.orchestrator.turns import execute_turn
from services.orchestrator.transport import get_json, get_text, post_json, put_json

MEMORY_URL = os.getenv("MEMORY_URL", "http://memory:8000").rstrip("/")
LEFT_URL = os.getenv("LEFT_URL", "http://left-model:8000").rstrip("/")
RIGHT_URL = os.getenv("RIGHT_URL", "http://right-model:8000").rstrip("/")
EXEC_URL = os.getenv("EXEC_URL", "http://executive-model:8000").rstrip("/")
WORKER_REQUEST_TIMEOUT_SECONDS = float(os.getenv("WORKER_REQUEST_TIMEOUT_SECONDS", "90"))
TURN_TIMEOUT_SECONDS = float(os.getenv("TURN_TIMEOUT_SECONDS", "240"))
SESSION_QUEUE_TIMEOUT_SECONDS = float(os.getenv("SESSION_QUEUE_TIMEOUT_SECONDS", "15"))
MAX_CONCURRENT_TURNS = int(os.getenv("MAX_CONCURRENT_TURNS", "2"))
MAX_LOBE_TRANSCRIPT_EVENTS = int(os.getenv("MAX_LOBE_TRANSCRIPT_EVENTS", "10"))
MAX_LOBE_TRANSCRIPT_CHARS = int(os.getenv("MAX_LOBE_TRANSCRIPT_CHARS", "6000"))
MAX_LOBE_TRANSCRIPT_EVENT_CHARS = int(os.getenv("MAX_LOBE_TRANSCRIPT_EVENT_CHARS", "1200"))
SEMANTIC_EMBEDDING_URL = os.getenv("SEMANTIC_EMBEDDING_URL", "http://ollama:11434/api/embed").rstrip("/")
SEMANTIC_EMBEDDING_MODEL = os.getenv("SEMANTIC_EMBEDDING_MODEL", "all-minilm")
SEMANTIC_EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("SEMANTIC_EMBEDDING_TIMEOUT_SECONDS", "8"))
SEMANTIC_REPEAT_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_REPEAT_SIMILARITY_THRESHOLD", "0.80"))
SEMANTIC_REPEAT_MAX_CANDIDATES = int(os.getenv("SEMANTIC_REPEAT_MAX_CANDIDATES", "12"))
REFLECTION_RETRY_POLL_SECONDS = float(os.getenv("REFLECTION_RETRY_POLL_SECONDS", "30"))
REFLECTION_RETRY_LEASE_SECONDS = int(os.getenv("REFLECTION_RETRY_LEASE_SECONDS", "300"))
DEBUG_API_ENABLED = os.getenv("ENABLE_DEBUG_API", "").lower() in {"1", "true", "yes"}
EXECUTIVE_ONLY_CONTROL = os.getenv("EXECUTIVE_ONLY_CONTROL", "").lower() in {"1", "true", "yes"}
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")
ANALYSIS_STOP_WORDS = {
    "answer", "biography", "character", "consistent", "constraint", "constraints", "established",
    "fact", "facts", "identity", "information", "observation", "observations", "recommend", "response",
    "share", "strategy", "subject", "use",
}

if (
    WORKER_REQUEST_TIMEOUT_SECONDS <= 0
    or TURN_TIMEOUT_SECONDS <= 0
    or SESSION_QUEUE_TIMEOUT_SECONDS <= 0
    or MAX_CONCURRENT_TURNS <= 0
    or MAX_LOBE_TRANSCRIPT_EVENTS <= 0
    or MAX_LOBE_TRANSCRIPT_CHARS < 500
    or MAX_LOBE_TRANSCRIPT_EVENT_CHARS < 200
    or SEMANTIC_EMBEDDING_TIMEOUT_SECONDS <= 0
    or SEMANTIC_REPEAT_MAX_CANDIDATES <= 0
    or REFLECTION_RETRY_POLL_SECONDS <= 0
    or REFLECTION_RETRY_LEASE_SECONDS <= 0
):
    raise RuntimeError("Runtime limits must be positive and transcript bounds must be practical")
if not 0.0 < SEMANTIC_REPEAT_SIMILARITY_THRESHOLD <= 1.0:
    raise RuntimeError("SEMANTIC_REPEAT_SIMILARITY_THRESHOLD must be between 0 and 1")

SESSION_TURN_LOCKS = SessionLockRegistry()
TURN_CAPACITY = asyncio.Semaphore(MAX_CONCURRENT_TURNS)
MODEL_METRICS: dict[str, dict[str, int | float | None]] = {
    role: {"calls": 0, "failures": 0, "last_ms": None, "average_ms": None}
    for role in ("left", "right", "executive")
}


@asynccontextmanager
async def orchestrator_lifespan(_: FastAPI):
    """Retry failed close-time reflections without keeping sessions open."""

    task = asyncio.create_task(_deferred_reflection_retry_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="Cognitive Character Orchestrator",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=orchestrator_lifespan,
)


@app.middleware("http")
async def require_api_key(request: Request, call_next: Any) -> Any:
    """Protect every externally reachable API operation except liveness checks.

    Docker Compose requires the token and nginx injects it into same-origin UI
    requests.  Keeping health uncredentialed permits container orchestration to
    determine readiness without turning operational secrets into health checks.
    """

    if request.url.path == "/health":
        return await call_next(request)
    if not API_AUTH_TOKEN:
        return JSONResponse(status_code=503, content={"detail": "API authentication is not configured."})
    supplied = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(supplied, API_AUTH_TOKEN):
        return JSONResponse(status_code=401, content={"detail": "Valid API credentials are required."})
    return await call_next(request)


class SessionCreate(BaseModel):
    character_id: str = Field(min_length=1, max_length=64)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)


class ProfileImportRequest(BaseModel):
    yaml: str = Field(min_length=1, max_length=5_000_000)


class ProfileDiffRequest(BaseModel):
    yaml: str = Field(min_length=1, max_length=5_000_000)


class KnowledgeCatalogRequest(BaseModel):
    yaml: str = Field(min_length=1, max_length=5_000_000)


class ReflectionResult(BaseModel):
    session_id: str
    summary: str
    mutation_results: list[dict[str, Any]]
    executive: dict[str, Any]


@asynccontextmanager
async def session_operation_lock(session_id: str):
    """Provide one lock for chat, reflection, and close operations on a session."""

    try:
        async with SESSION_TURN_LOCKS.acquire(session_id, SESSION_QUEUE_TIMEOUT_SECONDS):
            yield
    except TimeoutError as exc:
        raise HTTPException(409, "Another turn for this conversation is still being processed.") from exc


@asynccontextmanager
async def turn_capacity_lock():
    """Bound expensive live inference across sessions as well as within one session."""

    try:
        await asyncio.wait_for(TURN_CAPACITY.acquire(), timeout=SESSION_QUEUE_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise HTTPException(429, "The character runtime is busy. Please retry shortly.") from exc
    try:
        yield
    finally:
        TURN_CAPACITY.release()


def serialize_turn(handler: Any) -> Any:
    """Serialise stateful turns per session and bound the full browser-visible wait."""

    @wraps(handler)
    async def wrapped(session_id: str, *args: Any, **kwargs: Any) -> Any:
        async with session_operation_lock(session_id):
            async with turn_capacity_lock():
                try:
                    async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
                        return await handler(session_id, *args, **kwargs)
                except TimeoutError as exc:
                    raise HTTPException(504, "This turn exceeded the response time limit. It was not retried automatically.") from exc

    return wrapped


def bounded_lobe_transcript(session_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bind reusable transcript policy to this runtime's configured limits."""

    return _bounded_lobe_transcript(
        session_events,
        max_events=MAX_LOBE_TRANSCRIPT_EVENTS,
        max_characters=MAX_LOBE_TRANSCRIPT_CHARS,
        max_event_characters=MAX_LOBE_TRANSCRIPT_EVENT_CHARS,
    )


async def semantic_repeat_evidence(
    client: httpx.AsyncClient,
    message: str,
    prior_users: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind reusable embedding evidence to this runtime's configured provider."""

    return await _semantic_repeat_evidence(
        client,
        message,
        prior_users,
        embedding_url=SEMANTIC_EMBEDDING_URL,
        embedding_model=SEMANTIC_EMBEDDING_MODEL,
        timeout_seconds=SEMANTIC_EMBEDDING_TIMEOUT_SECONDS,
        similarity_threshold=SEMANTIC_REPEAT_SIMILARITY_THRESHOLD,
        max_candidates=SEMANTIC_REPEAT_MAX_CANDIDATES,
    )


def _model_topic(value: Any) -> str | None:
    topic = str(value or "").strip()
    if not topic or topic in {"topic.general", "stable topic identifier"}:
        return None
    return topic


def _analysis_anchor_tokens(analysis: dict[str, Any]) -> set[str]:
    """Extract fact-bearing terms from compact or legacy lobe artifacts."""

    values = [
        str(analysis.get("topic", "")),
        str(analysis.get("action", "")),
        # Legacy keys keep stored historical turns useful after the contract change.
        str(analysis.get("recommended_strategy", "")),
    ]
    for key in ("fact_refs", "association_keys", "observations", "associations"):
        values.extend(str(item) for item in analysis.get(key, []) if isinstance(item, str))
    return {
        token
        for value in values
        for token in _content_tokens(value)
        if token not in ANALYSIS_STOP_WORDS
    }


def _analysis_keys(analysis: dict[str, Any], *field_names: str) -> set[str]:
    """Return compact keys exactly, avoiding lexical loss for dotted fact IDs."""

    keys: set[str] = set()
    for field_name in field_names:
        value = analysis.get(field_name, [])
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            keys.update(str(item).strip().lower() for item in value if isinstance(item, str) and item.strip())
    return keys


def immediate_repeat_lobe_reuse(
    *,
    message: str,
    topic: str,
    session_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Reuse lobe artifacts for the immediately preceding answered same question.

    The fast path intentionally requires an exact normalized question in the same
    session. Rephrased questions continue through Left and Right so the executive
    can use their semantic evidence to recognize a non-exact repeat.
    """

    conversation = [
        event
        for event in session_events
        if event.get("event_type") in {"user_message", "character_message"}
    ]
    if not conversation or conversation[-1].get("event_type") != "character_message":
        return None
    reply = conversation[-1]
    reply_metadata = reply.get("metadata", {})
    if not isinstance(reply_metadata, dict):
        return None
    prior_user_id = str(reply_metadata.get("responds_to") or "")
    prior_user = next(
        (event for event in reversed(conversation[:-1]) if str(event.get("id")) == prior_user_id),
        None,
    )
    if not prior_user or str(prior_user.get("topic") or "") != topic:
        return None
    if _question_signature(str(prior_user.get("content", ""))) != _question_signature(message):
        return None
    left = reply_metadata.get("left")
    right = reply_metadata.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    return {
        "left": left,
        "right": right,
        "prior_speech": str(reply.get("content") or "").strip(),
        "source_event_id": reply.get("id"),
        "source_user_event_id": prior_user.get("id"),
        "reason": "immediate_answered_exact_repeat",
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
    embedding_matches: dict[str, float] | None = None,
    embedding_threshold: float = SEMANTIC_REPEAT_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Review repeat evidence after both hemisphere analyses and before executive speech.

    This is deliberately a deterministic *candidate* review, not a replacement for the
    executive model. It supplies recent wording, lobe-topic agreement, and a confidence
    signal so the executive can recognize semantic rephrases without receiving an
    unbounded transcript on every turn.
    """

    prior_events = [event for event in session_events if event.get("id") != current_event_id]
    replies_by_user = {
        str(event.get("metadata", {}).get("responds_to")): event
        for event in prior_events
        if event.get("event_type") == "character_message" and event.get("metadata", {}).get("responds_to")
    }
    # Failed worker calls can leave a user event without a reply. Keep it in the
    # transcript for auditability, but a retried request must not be treated as
    # repeated pressure until the character has actually answered it once.
    prior_users = [
        event
        for event in prior_events
        if event.get("event_type") == "user_message" and str(event.get("id")) in replies_by_user
    ]
    current_left_topic = _model_topic(left_result.get("topic"))
    current_associations = " ".join(
        _analysis_keys(right_result, "association_keys", "associations")
    )
    current_fact_refs = _analysis_keys(left_result, "fact_refs")

    embedding_matches = embedding_matches or {}
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

        embedding_score = float(embedding_matches.get(str(user_event.get("id")), 0.0))
        if embedding_score >= embedding_threshold and embedding_score > score:
            score, reason = embedding_score, "embedding similarity"

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
            prior_left = prior_reply.get("metadata", {}).get("left", {})
            prior_anchors = _analysis_anchor_tokens(prior_left)
            shared_anchors = current_anchors & prior_anchors
            if len(shared_anchors) >= 2 and score < 0.76:
                score, reason = 0.76, "left analyses share subject-specific fact anchors"
            shared_fact_refs = current_fact_refs & _analysis_keys(prior_left, "fact_refs")
            if shared_fact_refs and score < 0.78:
                score, reason = 0.78, "left analyses reference the same established fact"

        if current_associations and prior_reply:
            prior_associations = " ".join(
                _analysis_keys(
                    prior_reply.get("metadata", {}).get("right", {}),
                    "association_keys",
                    "associations",
                )
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
            "embedding_similarity": round(embedding_score, 4) if embedding_score else None,
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
        "embedding_similarity": (best or {}).get("embedding_similarity"),
        "embedding_threshold": embedding_threshold,
        "reason": (best or {}).get("reason", "no prior semantic match"),
        "consecutive_repeats": consecutive_repeats,
        "recent_turns": recent_turns,
    }


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
    """Compatibility façade for validated worker inference."""

    return await run_inference(client, base_url, req, expected_role)


async def infer_timed(
    client: httpx.AsyncClient,
    base_url: str,
    req: CognitiveRequest,
    expected_role: str,
) -> tuple[dict[str, Any], int]:
    """Compatibility façade for instrumented worker inference."""

    return await run_timed_inference(client, base_url, req, expected_role, MODEL_METRICS)


async def _probe_worker(
    client: httpx.AsyncClient,
    role: str,
    url: str,
) -> dict[str, Any]:
    """Compatibility façade for independently-instrumented worker probes."""

    return await probe_inference_worker(client, role, url, MODEL_METRICS)


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
    # Liveness must not disclose provider URLs, model names, or internal errors
    # to an unauthenticated health probe.
    return {"status": "ok" if all(not isinstance(x, Exception) for x in checks) else "degraded"}


@app.get("/status")
async def runtime_status(character_id: str | None = Query(default=None, min_length=1, max_length=64)) -> dict[str, Any]:
    """Expose authenticated, per-role readiness and the effective profile weighting."""

    async with httpx.AsyncClient(timeout=10) as client:
        workers = await asyncio.gather(
            _probe_worker(client, "left", LEFT_URL),
            _probe_worker(client, "right", RIGHT_URL),
            _probe_worker(client, "executive", EXEC_URL),
        )
        try:
            memory_response = await client.get(f"{MEMORY_URL}/health")
            memory_response.raise_for_status()
            memory_status = "ready"
        except httpx.HTTPError:
            memory_status = "unavailable"

        retry_jobs: dict[str, Any] = {"jobs": [], "pending_count": None}
        if memory_status == "ready":
            try:
                retry_response = await client.get(f"{MEMORY_URL}/reflection-jobs?limit=20")
                retry_response.raise_for_status()
                payload = retry_response.json()
                if isinstance(payload, dict):
                    retry_jobs = payload
            except (httpx.HTTPError, ValueError):
                retry_jobs = {"jobs": [], "pending_count": None}

        character: CharacterDocument | None = None
        if character_id:
            try:
                character, _ = await load_character(client, character_id)
            except HTTPException:
                # Model availability must remain visible even if a profile was
                # deleted between the selector refresh and this status request.
                character = None

    priorities = resolve_cognitive_priorities(character)
    ready = memory_status == "ready" and all(worker["status"] == "ready" for worker in workers)
    return {
        "status": "ready" if ready else "degraded",
        "memory": {"status": memory_status},
        "workers": workers,
        "character_id": character.id if character is not None else None,
        "cognitive_priorities": priorities,
        "semantic_repeat": {
            "embedding_model": SEMANTIC_EMBEDDING_MODEL or None,
            "embedding_configured": bool(SEMANTIC_EMBEDDING_URL and SEMANTIC_EMBEDDING_MODEL),
            "similarity_threshold": SEMANTIC_REPEAT_SIMILARITY_THRESHOLD,
        },
        "reflection_retry": {
            "poll_seconds": REFLECTION_RETRY_POLL_SECONDS,
            "lease_seconds": REFLECTION_RETRY_LEASE_SECONDS,
            "pending_count": retry_jobs.get("pending_count"),
            "jobs": retry_jobs.get("jobs", []),
        },
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


@app.get("/profiles/{character_id}/export")
async def export_profile(character_id: str) -> Response:
    async with httpx.AsyncClient(timeout=30) as client:
        upstream = await get_text(client, f"{MEMORY_URL}/profiles/{character_id}/export")
    return Response(
        content=upstream.content,
        media_type=upstream.headers.get("content-type", "application/yaml"),
        headers={
            "Content-Disposition": upstream.headers.get(
                "content-disposition", f'attachment; filename="{character_id}.snapshot.yaml"'
            )
        },
    )


@app.post("/profiles/import")
async def import_profile(request: ProfileImportRequest) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        return await post_json(client, f"{MEMORY_URL}/profiles/import", request.model_dump())


@app.post("/profiles/{character_id}/diff")
async def diff_profile(character_id: str, request: ProfileDiffRequest) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        return await post_json(
            client,
            f"{MEMORY_URL}/profiles/{character_id}/diff",
            request.model_dump(),
        )


@app.get("/knowledge/catalog")
async def knowledge_catalog() -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/knowledge/catalog")


@app.get("/knowledge/export")
async def export_knowledge_catalog() -> Response:
    async with httpx.AsyncClient(timeout=30) as client:
        upstream = await get_text(client, f"{MEMORY_URL}/knowledge/export")
    return Response(
        content=upstream.content,
        media_type=upstream.headers.get("content-type", "application/yaml"),
        headers={
            "Content-Disposition": upstream.headers.get(
                "content-disposition", 'attachment; filename="knowledge-catalog.yaml"'
            )
        },
    )


@app.get("/knowledge/schema-sample")
async def export_knowledge_schema_sample() -> Response:
    async with httpx.AsyncClient(timeout=20) as client:
        upstream = await get_text(client, f"{MEMORY_URL}/knowledge/schema-sample")
    return Response(
        content=upstream.content,
        media_type=upstream.headers.get("content-type", "application/yaml"),
        headers={
            "Content-Disposition": upstream.headers.get(
                "content-disposition", 'attachment; filename="catalog.example.yaml"'
            )
        },
    )


@app.post("/knowledge/validate")
async def validate_knowledge_catalog(request: KnowledgeCatalogRequest) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        return await post_json(client, f"{MEMORY_URL}/knowledge/validate", request.model_dump())


@app.put("/knowledge/catalog")
async def update_knowledge_catalog(request: KnowledgeCatalogRequest) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        return await put_json(client, f"{MEMORY_URL}/knowledge/catalog", request.model_dump())


@app.post("/knowledge/import")
async def import_knowledge_catalog(request: KnowledgeCatalogRequest) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        return await post_json(client, f"{MEMORY_URL}/knowledge/import", request.model_dump())


@app.post("/sessions")
async def create_session(req: SessionCreate) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await post_json(client, f"{MEMORY_URL}/sessions", req.model_dump())


@app.get("/sessions/{session_id}/events")
async def session_events(session_id: str) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/sessions/{session_id}/events")


@app.post("/sessions/{session_id}/chat")
@serialize_turn
async def chat(session_id: str, req: ChatRequest) -> dict[str, Any]:
    return await execute_turn(
        session_id,
        req,
        memory_url=MEMORY_URL,
        left_url=LEFT_URL,
        right_url=RIGHT_URL,
        executive_url=EXEC_URL,
        worker_timeout_seconds=WORKER_REQUEST_TIMEOUT_SECONDS,
        executive_only_control=EXECUTIVE_ONLY_CONTROL,
        semantic_embedding_model=SEMANTIC_EMBEDDING_MODEL,
        semantic_repeat_similarity_threshold=SEMANTIC_REPEAT_SIMILARITY_THRESHOLD,
        model_metrics=MODEL_METRICS,
        get_session=_session_meta,
        load_character=load_character,
        get_json=get_json,
        post_json=post_json,
        infer_timed=infer_timed,
        bounded_lobe_transcript=bounded_lobe_transcript,
        semantic_repeat_evidence=semantic_repeat_evidence,
        immediate_repeat_lobe_reuse=immediate_repeat_lobe_reuse,
        executive_repeat_review=executive_repeat_review,
        interaction_classification=InteractionClassification,
    )



async def _session_meta(client: httpx.AsyncClient, session_id: str) -> dict[str, Any]:
    # Endpoint deliberately kept on memory service to avoid orchestrator-side session state.
    return await get_json(client, f"{MEMORY_URL}/sessions/{session_id}")


async def _reflect(session_id: str) -> ReflectionResult:
    result = await reflect_session(
        session_id,
        memory_url=MEMORY_URL,
        executive_url=EXEC_URL,
        worker_timeout_seconds=WORKER_REQUEST_TIMEOUT_SECONDS,
        get_session=_session_meta,
        load_character=load_character,
        get_json=get_json,
        post_json=post_json,
        infer=infer,
    )
    return ReflectionResult(
        session_id=result.session_id,
        summary=result.summary,
        mutation_results=result.mutation_results,
        executive=result.executive,
    )


async def process_deferred_reflections_once() -> dict[str, Any]:
    """Lease and retry one failed close-time reflection, preserving close semantics."""

    async with httpx.AsyncClient(timeout=20) as client:
        claimed = await post_json(
            client,
            f"{MEMORY_URL}/reflection-jobs/claim",
            {"lease_seconds": REFLECTION_RETRY_LEASE_SECONDS},
        )
    job = claimed.get("job") if isinstance(claimed, dict) else None
    if not isinstance(job, dict):
        return {"processed": False, "reason": "no_due_reflection_job"}
    session_id = str(job.get("session_id") or "")
    if not session_id:
        return {"processed": False, "reason": "invalid_reflection_job"}

    try:
        async with session_operation_lock(session_id):
            async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
                reflection = await _reflect(session_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        async with httpx.AsyncClient(timeout=20) as client:
            released = await post_json(
                client,
                f"{MEMORY_URL}/reflection-jobs/{session_id}/reschedule",
                {"error": "Deferred reflection generation failed; retry scheduled."},
            )
        return {
            "processed": True,
            "status": "rescheduled",
            "session_id": session_id,
            "backoff_seconds": released.get("backoff_seconds"),
        }

    async with httpx.AsyncClient(timeout=20) as client:
        await post_json(client, f"{MEMORY_URL}/reflection-jobs/{session_id}/complete", {})
    return {
        "processed": True,
        "status": "completed",
        "session_id": session_id,
        "summary": reflection.summary,
    }


async def _deferred_reflection_retry_loop() -> None:
    """Run one durable retry at a time; a failed poll never kills the runtime."""

    while True:
        await asyncio.sleep(REFLECTION_RETRY_POLL_SECONDS)
        try:
            await process_deferred_reflections_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            # The job remains pending or lease-expiry reclaimable in memory. Do
            # not let an unavailable memory service stop later retries.
            continue


@app.post("/sessions/{session_id}/reflect", response_model=ReflectionResult)
async def reflect(session_id: str) -> ReflectionResult:
    async with session_operation_lock(session_id):
        try:
            async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
                result = await _reflect(session_id)
        except TimeoutError as exc:
            raise HTTPException(504, "Reflection exceeded the response time limit.") from exc
    async with httpx.AsyncClient(timeout=20) as client:
        await post_json(client, f"{MEMORY_URL}/reflection-jobs/{session_id}/complete", {})
    return result


@app.post("/reflection-retries/run")
async def run_deferred_reflection_retry() -> dict[str, Any]:
    """Run one queued reflection immediately from the authenticated status UI."""

    return await process_deferred_reflections_once()


@app.post("/sessions/{session_id}/close")
async def close(session_id: str) -> dict[str, Any]:
    """Close authoritatively; reflection enriches a conversation but never traps it open."""

    async with session_operation_lock(session_id):
        reflection: ReflectionResult | None = None
        reflection_warning: str | None = None
        try:
            async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
                reflection = await _reflect(session_id)
        except (HTTPException, TimeoutError):
            # A local model can be temporarily unable to produce a valid optional
            # reflection. The user's explicit close action must still release the
            # session lock and prevent more messages from being accepted.
            reflection_warning = "The conversation closed, and its reflection was queued for retry."
        async with httpx.AsyncClient(timeout=20) as client:
            if reflection is None:
                try:
                    await post_json(
                        client,
                        f"{MEMORY_URL}/reflection-jobs/{session_id}/schedule",
                        {"error": "Close-time reflection generation failed."},
                    )
                except HTTPException:
                    reflection_warning = (
                        "The conversation closed, but its reflection could not be generated or queued for retry."
                    )
            closed = await post_json(client, f"{MEMORY_URL}/sessions/{session_id}/close", {})
        return {
            "session": closed,
            "reflection": reflection.model_dump(mode="json") if reflection else None,
            "reflection_warning": reflection_warning,
        }


@app.get("/debug/{character_id}")
async def debug(character_id: str) -> Any:
    if not DEBUG_API_ENABLED:
        raise HTTPException(404, "Debug routes are disabled.")
    async with httpx.AsyncClient(timeout=20) as client:
        return await get_json(client, f"{MEMORY_URL}/debug/{character_id}")
