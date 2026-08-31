from __future__ import annotations

import asyncio
import hmac
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from functools import wraps
from math import isfinite, sqrt
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError

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
from services.orchestrator.claims import claim_evidence_catalog, verify_factual_claims
from services.orchestrator.locking import SessionLockRegistry
from services.orchestrator.relationships import historical_relationships, merge_historical_relationships

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
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")
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


def resolve_cognitive_priorities(character: CharacterDocument | None) -> dict[str, Any]:
    """Normalize an author's Left/Right priorities into an enforceable turn policy.

    The values are source-controlled primer inputs rather than mutable mood. A
    malformed or absent value falls back to a balanced system, avoiding a bad
    profile edit silently starving either cognitive role.
    """

    cognition = character.cognition if character is not None else {}

    def weight(name: str) -> float:
        value = cognition.get(name, 0.5) if isinstance(cognition, dict) else 0.5
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.5
        return numeric if isfinite(numeric) and numeric >= 0 else 0.5

    left_raw = weight("left_weight")
    right_raw = weight("right_weight")
    total = left_raw + right_raw
    left = 0.5 if total <= 0 else left_raw / total
    right = 0.5 if total <= 0 else right_raw / total
    difference = left - right
    primary_role = "balanced" if abs(difference) < 0.08 else ("left" if difference > 0 else "right")
    return {
        "left_weight": round(left, 4),
        "right_weight": round(right, 4),
        "primary_role": primary_role,
        "weight_gap": round(abs(difference), 4),
        # Facts and declared constraints never become optional just because the
        # associative hemisphere is configured as primary for delivery choices.
        "invariant": "left_constraints_bind",
        "enforcement": "weighted_arbitration_plan_and_role_attention_budget",
    }


def bounded_lobe_transcript(session_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a recent raw conversation window with hard event and character caps.

    It deliberately carries raw event content rather than a model summary, while
    keeping an old session from consuming lobe context or response time.
    """

    selected: list[dict[str, Any]] = []
    remaining = MAX_LOBE_TRANSCRIPT_CHARS
    conversation = [
        event
        for event in session_events
        if event.get("event_type") in {"user_message", "character_message"}
    ]
    for event in reversed(conversation[-MAX_LOBE_TRANSCRIPT_EVENTS:]):
        content = str(event.get("content") or "").strip()
        if not content or remaining <= 0:
            continue
        permitted = min(MAX_LOBE_TRANSCRIPT_EVENT_CHARS, remaining)
        clipped = content[:permitted]
        selected.append(
            {
                "event_id": str(event.get("id") or ""),
                "event_type": str(event.get("event_type") or ""),
                "actor": event.get("actor"),
                "content": clipped,
                "topic": event.get("topic"),
                "content_truncated": len(clipped) < len(content),
            }
        )
        remaining -= len(clipped)
    return list(reversed(selected))


def weighted_arbitration_plan(
    priorities: dict[str, Any],
    left_result: dict[str, Any],
    right_result: dict[str, Any],
) -> dict[str, Any]:
    """Materialize the authored weighting as bounded, auditable executive input."""

    primary = str(priorities["primary_role"])
    left_packet = {
        "weight": priorities["left_weight"],
        "action": str(left_result.get("action") or "answer"),
        "fact_refs": [str(item) for item in left_result.get("fact_refs", [])[:4]],
        "constraints": [str(item) for item in left_result.get("constraints", [])[:3]],
    }
    right_packet = {
        "weight": priorities["right_weight"],
        "action": str(right_result.get("action") or "inform"),
        "tone": str(right_result.get("tone") or "neutral"),
        "risk": str(right_result.get("risk") or "low"),
        "association_keys": [str(item) for item in right_result.get("association_keys", [])[:4]],
    }
    primary_packet = left_packet if primary == "left" else right_packet if primary == "right" else None
    return {
        "priorities": priorities,
        "primary_role": primary,
        "primary_packet": primary_packet,
        "left": left_packet,
        "right": right_packet,
        "binding_rules": [
            "left.constraints are binding; right affect or tone cannot override them",
            (
                f"when non-factual response choices conflict, favor the {primary} packet"
                if primary in {"left", "right"}
                else "when non-factual response choices conflict, balance both packets"
            ),
        ],
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    try:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
    except (TypeError, ValueError):
        return None
    if not isfinite(dot) or left_norm <= 0 or right_norm <= 0:
        return None
    score = dot / (left_norm * right_norm)
    return score if isfinite(score) else None


async def semantic_repeat_evidence(
    client: httpx.AsyncClient,
    message: str,
    prior_users: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use a small embedding model after lobe work to find non-lexical repeats.

    The embedding call is bounded to recent answered questions. It fails closed
    to existing deterministic/lobe evidence so an optional semantic accelerator
    can never make a normal character turn unavailable.
    """

    candidates = [
        event for event in prior_users[-SEMANTIC_REPEAT_MAX_CANDIDATES:]
        if str(event.get("content") or "").strip() and str(event.get("id") or "")
    ]
    if not SEMANTIC_EMBEDDING_URL or not SEMANTIC_EMBEDDING_MODEL or not candidates:
        return {
            "available": False,
            "model": SEMANTIC_EMBEDDING_MODEL or None,
            "threshold": SEMANTIC_REPEAT_SIMILARITY_THRESHOLD,
            "matches": {},
            "reason": "no_embedding_candidates_or_configuration",
        }
    inputs = [message, *[str(event["content"]) for event in candidates]]
    try:
        response = await client.post(
            SEMANTIC_EMBEDDING_URL,
            json={"model": SEMANTIC_EMBEDDING_MODEL, "input": inputs, "truncate": True},
            timeout=SEMANTIC_EMBEDDING_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = payload.get("embeddings", []) if isinstance(payload, dict) else []
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise ValueError("embedding provider returned an unexpected vector count")
        parsed = [
            [float(value) for value in vector]
            for vector in vectors
            if isinstance(vector, list)
        ]
        if len(parsed) != len(inputs):
            raise ValueError("embedding provider returned a non-vector value")
    except (httpx.HTTPError, TypeError, ValueError):
        return {
            "available": False,
            "model": SEMANTIC_EMBEDDING_MODEL,
            "threshold": SEMANTIC_REPEAT_SIMILARITY_THRESHOLD,
            "matches": {},
            "reason": "embedding_provider_unavailable",
        }

    matches: dict[str, float] = {}
    for event, vector in zip(candidates, parsed[1:]):
        score = _cosine_similarity(parsed[0], vector)
        if score is not None:
            matches[str(event["id"])] = round(score, 4)
    return {
        "available": True,
        "model": str(payload.get("model") or SEMANTIC_EMBEDDING_MODEL),
        "threshold": SEMANTIC_REPEAT_SIMILARITY_THRESHOLD,
        "matches": matches,
        "reason": "embedding_similarity",
    }


def _response_terms(text: str) -> set[str]:
    """Normalize a few common inflections before checking a repeated answer.

    This is deliberately not a semantic similarity system.  It is a narrow guard
    for the exact-repeat fast path, where the executive has already been told
    which previous answer it must reframe.  A small amount of stemming catches
    outputs such as ``protect`` / ``protecting`` without making a new answer on
    the same subject look automatically invalid.
    """

    terms: set[str] = set()
    for token in _content_tokens(text):
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ied"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if len(token) > 1:
            terms.add(token)
    return terms


def response_substantially_repeats_prior_answer(speech: str, prior_speech: str) -> bool:
    """Return whether the Executive merely echoed the immediate prior answer.

    The check is only applied after exact-question lobe reuse.  It permits the
    Executive to stay on the same topic, while blocking answers whose fact-bearing
    wording is substantially the same as the answer the user has already seen.
    """

    if _question_signature(speech) == _question_signature(prior_speech):
        return True
    speech_terms = _response_terms(speech)
    prior_terms = _response_terms(prior_speech)
    if len(speech_terms) < 3 or len(prior_terms) < 3:
        return False
    shared = len(speech_terms & prior_terms)
    return shared >= 3 and (shared / min(len(speech_terms), len(prior_terms))) >= 0.50


def response_substantially_repeats_recent_answers(speech: str, prior_speeches: list[str]) -> bool:
    """Keep repeat deliberation from cycling back to an earlier answer."""

    return any(
        response_substantially_repeats_prior_answer(speech, prior_speech)
        for prior_speech in prior_speeches
        if prior_speech.strip()
    )


def repeat_intent_fallback(assessment: dict[str, Any], consecutive_repeats: int) -> str:
    """Last-resort response tied to the Executive's chosen repeat hypothesis.

    A tiny local model can still echo after a planned revision.  This guard keeps
    the visible response distinct without collapsing every failed retry into the
    same generic clarification phrase.  It is only used after the assessment and
    two Executive speech attempts have both completed.
    """

    response_mode = str(assessment.get("response_mode") or "")
    primary = str(assessment.get("primary_hypothesis") or "")
    # The stage deliberately advances with the repeat streak. That gives a
    # character several reasonable interpretations to try before treating the
    # user as adversarial, instead of restating one static question forever.
    stage = max(consecutive_repeats - 2, 0) % 4

    if response_mode == "test_consistency" or "consisten" in primary:
        variants = [
            "If you are checking whether my answer changes when you ask again, it does not. Is there a circumstance you want to test?",
            "My answer is consistent. Are you trying to see whether a different situation would change it?",
            "I cannot tell whether you are testing my consistency or looking for more detail. Which is it?",
            "If consistency is the point, I have been clear. Tell me what condition you think might alter the answer.",
        ]
        return variants[stage]

    if response_mode == "check_understanding" or any(key in primary for key in ("understand", "clear")):
        variants = [
            "I may not have explained the point clearly. Do you want the principle, or an example of it in practice?",
            "Perhaps I have answered too broadly. Which word or part of the answer is unclear?",
            "We may be using the same words for different questions. What do you mean by it here?",
            "I do not want to keep guessing at the gap. Name the part you want me to unpack.",
        ]
        return variants[stage]

    if response_mode == "set_boundary":
        variants = [
            "I've answered the question directly. If you mean something more specific, say which part you want to examine.",
            "I can approach the subject another way, but repeating the same words does not tell me what is missing.",
            "If you are looking for a different answer, be direct about the distinction you want to explore.",
            "I have tried to meet the question as asked. Tell me the actual point you want to press.",
        ]
        return variants[stage]

    # New angle and invitation modes share a progression from offering a useful
    # distinction, through checking the user's frame, to explicitly requesting
    # the missing distinction. This is not an emotional escalation on its own.
    variants = [
        "Perhaps the broad answer is not the useful one. Are you asking what I value, or how that value guides a decision?",
        "It sounds as though the principle alone is not enough. Do you want to know how it guides a decision, or whether another concern can outrank it?",
        "We may be talking past each other. Are you asking about the value itself, the work it leads me to do, or a situation where it is tested?",
        "If the answer is still not reaching you, tell me what distinction you need me to make.",
    ]
    return variants[stage]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


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


def _question_signature(message: str) -> str:
    """Canonicalize only exact repeats; rephrases still receive fresh lobe work."""

    return " ".join(CONTENT_TOKEN_RE.findall(message.lower()))


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


def derive_repeat_dynamics(
    *,
    character: CharacterDocument,
    mutable_state: dict[str, Any],
    review: dict[str, Any],
    user_turn_count: int,
    escalation_decision: str = "hold",
) -> tuple[RepeatDynamics, dict[str, float], bool]:
    """Measure repeat pressure; only the executive may raise durable defensiveness."""

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
        projected_defensiveness = _clamp((prior_defensiveness * 0.96) + added_pressure)
    elif prior_defensiveness:
        # Returning to a charged subject can cool it slowly, but never erase it merely
        # because the user changed wording or briefly moved to another subject.
        projected_defensiveness = _clamp(prior_defensiveness * 0.97)
    else:
        projected_defensiveness = 0.0

    if semantic_repeat and escalation_decision == "increase":
        subject_defensiveness = projected_defensiveness
    elif escalation_decision == "deescalate":
        subject_defensiveness = _clamp(prior_defensiveness * 0.70)
    elif semantic_repeat:
        # A repeat alone is evidence, not a license to make the character more
        # suspicious. The executive's default hold preserves the current gauge.
        subject_defensiveness = prior_defensiveness
    else:
        subject_defensiveness = projected_defensiveness

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
    suggested_pressure = _clamp(projected_defensiveness * (1.0 - conversation_patience))

    def posture_for(pressure: float) -> str:
        if pressure >= 0.36:
            return "defensive"
        if pressure >= 0.16:
            return "confused"
        return "reclarify" if semantic_repeat else "normal"

    posture = posture_for(intersection_pressure)
    suggested_posture = posture_for(suggested_pressure)
    escalation_recommendation = "increase" if semantic_repeat and consecutive_repeats >= 2 else "hold"

    return (
        RepeatDynamics(
            conversation_patience=round(conversation_patience, 4),
            subject_defensiveness=round(subject_defensiveness, 4),
            intersection_pressure=round(intersection_pressure, 4),
            response_posture=posture,
            suggested_posture=suggested_posture,
            escalation_recommendation=escalation_recommendation,
            semantic_repeat=semantic_repeat,
            consecutive_repeats=consecutive_repeats,
            subject_key=subject_key,
            review_confidence=round(float(review["confidence"]), 4),
        ),
        updated_topics,
        changed,
    )


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
        raise HTTPException(504, "The update took too long. Please retry.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "A required service is unavailable. Please retry shortly.") from exc
    if r.status_code >= 400:
        raise HTTPException(r.status_code, _upstream_detail(r))
    return r.json()


async def get_text(client: httpx.AsyncClient, url: str) -> httpx.Response:
    try:
        response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "The export took too long. Please retry.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "The export service is unavailable. Please retry shortly.") from exc
    if response.status_code >= 400:
        raise HTTPException(response.status_code, _upstream_detail(response))
    return response


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


async def infer_timed(
    client: httpx.AsyncClient,
    base_url: str,
    req: CognitiveRequest,
    expected_role: str,
) -> tuple[dict[str, Any], int]:
    """Return a worker result with monotonic timing for live topology comparisons."""

    started = time.perf_counter()
    try:
        result = await infer(client, base_url, req, expected_role)
    except Exception:
        elapsed = round((time.perf_counter() - started) * 1000)
        metric = MODEL_METRICS[expected_role]
        metric["calls"] = int(metric["calls"] or 0) + 1
        metric["failures"] = int(metric["failures"] or 0) + 1
        metric["last_ms"] = elapsed
        raise
    elapsed = round((time.perf_counter() - started) * 1000)
    metric = MODEL_METRICS[expected_role]
    calls = int(metric["calls"] or 0) + 1
    previous_average = metric["average_ms"]
    metric["calls"] = calls
    metric["last_ms"] = elapsed
    metric["average_ms"] = round(
        elapsed if previous_average is None else ((float(previous_average) * (calls - 1)) + elapsed) / calls,
        1,
    )
    return result, elapsed


async def _probe_worker(
    client: httpx.AsyncClient,
    role: str,
    url: str,
) -> dict[str, Any]:
    """Probe a role independently so one degraded worker never hides the others."""

    started = time.perf_counter()
    try:
        response = await client.get(f"{url}/health")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("worker health response was not an object")
        readiness = "ready"
        model = str(payload.get("model") or "configured")
    except (httpx.HTTPError, ValueError, TypeError):
        readiness = "unavailable"
        model = None
    latency = round((time.perf_counter() - started) * 1000)
    metric = MODEL_METRICS[role]
    return {
        "role": role,
        "status": readiness,
        "model": model,
        "health_probe_ms": latency,
        "calls": int(metric["calls"] or 0),
        "failures": int(metric["failures"] or 0),
        "last_inference_ms": metric["last_ms"],
        "average_inference_ms": metric["average_ms"],
    }


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
        prior_transcript = await get_json(client, f"{MEMORY_URL}/sessions/{session_id}/events?limit=200")
        replies_by_user = {
            str(event.get("metadata", {}).get("responds_to")): event
            for event in prior_transcript
            if event.get("event_type") == "character_message"
            and isinstance(event.get("metadata"), dict)
            and event["metadata"].get("responds_to")
        }
        completed_session_questions = [
            event
            for event in prior_transcript
            if event.get("event_type") == "user_message"
            and event.get("topic") == topic
            and str(event.get("id")) in replies_by_user
        ]
        prior_times = len(completed_session_questions)
        prior_answer = None
        for event in reversed(prior_transcript):
            if event.get("event_type") == "character_message" and event.get("topic") == topic:
                prior_answer = event.get("content")
                break
        classification = InteractionClassification(
            interaction_type="repeated_question" if prior_times > 0 else "new_subject",
            topic=topic,
            prior_answer=prior_answer,
            times_asked=prior_times + 1,
            related_event_ids=[str(event["id"]) for event in completed_session_questions],
        )
        existing_idempotent_user: dict[str, Any] | None = None
        if req.idempotency_key:
            for event in prior_transcript:
                metadata = event.get("metadata", {})
                if (
                    event.get("event_type") == "user_message"
                    and isinstance(metadata, dict)
                    and metadata.get("idempotency_key") == req.idempotency_key
                ):
                    if event.get("content") != req.message:
                        raise HTTPException(409, "This idempotency key was already used for a different message.")
                    existing_idempotent_user = event
                    break
            if existing_idempotent_user:
                matching_reply = next(
                    (
                        event
                        for event in prior_transcript
                        if event.get("event_type") == "character_message"
                        and isinstance(event.get("metadata"), dict)
                        and event["metadata"].get("responds_to") == existing_idempotent_user.get("id")
                    ),
                    None,
                )
                if matching_reply:
                    metadata = matching_reply.get("metadata", {})
                    return {
                        "session_id": session_id,
                        "character_id": character_id,
                        "message": matching_reply.get("content", ""),
                        "interaction": metadata.get("interaction", {}),
                        "cognition": {
                            "left": metadata.get("left", {}),
                            "right": metadata.get("right", {}),
                            "executive": metadata.get("executive", {}),
                            "lobe_execution": metadata.get("lobe_execution", {}),
                            "repeat_review": metadata.get("repeat_review", {}),
                            "cognitive_priorities": metadata.get("cognitive_priorities", {}),
                            "weighted_arbitration": metadata.get("weighted_arbitration", {}),
                            "timing_ms": {"replayed": True},
                        },
                        "memory_writes": [],
                        "mutation_results": [],
                        "idempotent_replay": True,
                    }
        lobe_reuse = immediate_repeat_lobe_reuse(
            message=req.message,
            topic=topic,
            session_events=prior_transcript,
        )

        if existing_idempotent_user:
            user_event = EventRecord.model_validate(existing_idempotent_user)
        else:
            user_event = EventRecord(
                character_id=character_id,
                session_id=session_id,
                event_type="user_message",
                actor="user",
                content=req.message,
                topic=topic,
                metadata={
                    "interaction": classification.model_dump(mode="json"),
                    "idempotency_key": req.idempotency_key,
                },
            )
            user_event = EventRecord.model_validate(
                await post_json(client, f"{MEMORY_URL}/events", user_event.model_dump(mode="json"))
            )

        memories = await get_json(client, f"{MEMORY_URL}/memories/{character_id}?topic={topic}&limit=20")
        knowledge_query = urlencode({"query": req.message, "limit": 12})
        knowledge_context = await get_json(
            client, f"{MEMORY_URL}/knowledge/for-character/{character_id}?{knowledge_query}"
        )
        cognitive_priorities = resolve_cognitive_priorities(character)
        lobe_transcript = bounded_lobe_transcript(prior_transcript)
        common_context = {
            "interaction": classification.model_dump(mode="json"),
            "memories": memories,
            "mutable_state": state.get("mutable_state", {}),
            "beliefs": state.get("beliefs", {}),
            "goals": state.get("goals", []),
            "cognitive_priorities": cognitive_priorities,
            # This is a label-authorized view of the static general corpus. Raw
            # corpus records and denied classifications never reach a model.
            "general_knowledge": knowledge_context.get("items", []),
        }
        # This is the sole citation namespace accepted from Executive factual
        # claims. The values are visible for grounded writing; the keys are
        # persisted as audit-friendly provenance.
        common_context["claim_evidence"] = claim_evidence_catalog(
            character, state, knowledge_context.get("items", [])
        )
        cognition_req = CognitiveRequest(
            character=character,
            user_input=req.message,
            context=common_context,
            transcript=lobe_transcript,
        )

        if lobe_reuse:
            left_result = dict(lobe_reuse["left"])
            right_result = dict(lobe_reuse["right"])
            left_ms = right_ms = 0
            lobe_execution = {
                "mode": "reused",
                "source_event_id": lobe_reuse["source_event_id"],
                "source_user_event_id": lobe_reuse["source_user_event_id"],
                "reason": lobe_reuse["reason"],
            }
        else:
            left_req = cognition_req.model_copy(
                update={
                    "context": {
                        **common_context,
                        "role_attention": {
                            "role": "left",
                            "weight": cognitive_priorities["left_weight"],
                            "attention_budget": round(0.5 + cognitive_priorities["left_weight"], 4),
                        },
                    }
                }
            )
            right_req = cognition_req.model_copy(
                update={
                    "context": {
                        **common_context,
                        "role_attention": {
                            "role": "right",
                            "weight": cognitive_priorities["right_weight"],
                            "attention_budget": round(0.5 + cognitive_priorities["right_weight"], 4),
                        },
                    }
                }
            )
            left_task = infer_timed(client, LEFT_URL, left_req, "left")
            right_task = infer_timed(client, RIGHT_URL, right_req, "right")
            (left_result, left_ms), (right_result, right_ms) = await asyncio.gather(left_task, right_task)
            lobe_execution = {
                "mode": "fresh",
                "transcript_events": len(lobe_transcript),
                "transcript_characters": sum(len(str(event["content"])) for event in lobe_transcript),
            }

        # The lobes are intentionally free to arrive at overlapping thoughts. The
        # executive receives a separate, bounded review after both are complete so
        # it can recognize a rephrased repeat without making either lobe suppress
        # its own reasoning.
        transcript = await get_json(client, f"{MEMORY_URL}/sessions/{session_id}/events?limit=200")
        prior_answered_users = [
            event
            for event in prior_transcript
            if event.get("event_type") == "user_message" and str(event.get("id")) in replies_by_user
        ]
        embedding_evidence = (
            await semantic_repeat_evidence(client, req.message, prior_answered_users)
            if not lobe_reuse
            else {
                "available": False,
                "model": SEMANTIC_EMBEDDING_MODEL,
                "threshold": SEMANTIC_REPEAT_SIMILARITY_THRESHOLD,
                "matches": {},
                "reason": "exact_repeat_lobe_reuse",
            }
        )
        repeat_review = executive_repeat_review(
            message=req.message,
            topic=topic,
            current_event_id=user_event.id or "",
            session_events=transcript,
            left_result=left_result,
            right_result=right_result,
            prior_times=prior_times,
            embedding_matches=embedding_evidence["matches"],
            embedding_threshold=float(embedding_evidence["threshold"]),
        )
        repeat_review["embedding"] = {
            key: embedding_evidence[key]
            for key in ("available", "model", "threshold", "reason")
        }
        arbitration_plan = weighted_arbitration_plan(cognitive_priorities, left_result, right_result)
        user_turn_count = sum(1 for event in transcript if event.get("event_type") == "user_message")
        repeat_dynamics, _, _ = derive_repeat_dynamics(
            character=character,
            mutable_state=state.get("mutable_state", {}),
            review=repeat_review,
            user_turn_count=user_turn_count,
            escalation_decision="hold",
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

        prior_repeated_speech = str(
            lobe_reuse.get("prior_speech", "") if lobe_reuse else repeat_review.get("matched_answer", "")
        ).strip()
        repeat_deliberation = {
            "enabled": bool(repeat_dynamics.semantic_repeat and prior_repeated_speech),
            # A bounded copy lets the Executive compare its answer without adding
            # unbounded transcript history to the repeat path.
            "previous_speech": prior_repeated_speech[:1600],
        }
        prior_repeat_speeches = [
            str(turn.get("content") or "")
            for turn in repeat_review.get("recent_turns", [])
            if turn.get("event_type") == "character_message"
        ]
        if repeat_deliberation["enabled"]:
            lobe_execution["repeat_deliberation_required"] = True

        executive_mutable_state = dict(state.get("mutable_state", {}))
        executive_mutable_state["topic_defensiveness"] = state.get("mutable_state", {}).get(
            "topic_defensiveness", {}
        )
        executive_context = {
            **common_context,
            "interaction": classification.model_dump(mode="json"),
            "conversation_dynamics": repeat_dynamics.model_dump(mode="json"),
            "cognitive_priorities": cognitive_priorities,
            "weighted_arbitration": arbitration_plan,
            "executive_repeat_review": repeat_review,
            "lobe_execution": lobe_execution,
            "repeat_deliberation": repeat_deliberation,
            "mutable_state": executive_mutable_state,
        }
        repeat_assessment: dict[str, Any] | None = None
        repeat_assessment_ms = 0
        if repeat_deliberation["enabled"]:
            # Repeated questions deserve a separate Executive-only assessment of
            # plausible intent. This replaces the old canned reframe fallback and
            # leaves the two lobe calls skipped for immediate exact repeats.
            assessment_context = {
                **executive_context,
                "repeat_deliberation": {**repeat_deliberation, "phase": "assessment"},
            }
            assessment_req = cognition_req.model_copy(
                update={
                    "context": assessment_context,
                    "left_result": left_result,
                    "right_result": right_result,
                    "mode": "repeat_assessment",
                }
            )
            repeat_assessment, repeat_assessment_ms = await infer_timed(
                client, EXEC_URL, assessment_req, "executive"
            )
            repeat_deliberation["assessment"] = repeat_assessment

        executive_context["repeat_deliberation"] = repeat_deliberation
        exec_req = cognition_req.model_copy(
            update={
                "context": executive_context,
                "left_result": left_result,
                "right_result": right_result,
            }
        )
        executive, executive_ms = await infer_timed(client, EXEC_URL, exec_req, "executive")
        speech = str(executive.get("speech", "")).strip()
        if not speech:
            raise HTTPException(502, "Executive produced no speech")
        executive = {**executive, "speech": speech}
        claim_audit = verify_factual_claims(executive, common_context["claim_evidence"])

        repeat_revision_ms = 0
        repeat_revision_used = False
        intent_fallback_used = False
        if repeat_deliberation["enabled"] and response_substantially_repeats_recent_answers(
            speech, prior_repeat_speeches
        ):
            # The assessment is preserved for a second, higher-budget Executive
            # response. It should revise the chosen hypothesis-driven action,
            # rather than falling through to a generic clarification line.
            repeat_revision_used = True
            retry_deliberation = {
                **repeat_deliberation,
                "rejected_speech": speech[:1600],
            }
            retry_context = {**executive_context, "repeat_deliberation": retry_deliberation}
            retry_req = exec_req.model_copy(update={"context": retry_context})
            executive, repeat_revision_ms = await infer_timed(client, EXEC_URL, retry_req, "executive")
            executive_ms += repeat_revision_ms
            speech = str(executive.get("speech", "")).strip()
            if not speech:
                raise HTTPException(502, "Executive produced no speech on repeat deliberation revision")
            executive = {**executive, "speech": speech}
            claim_audit = verify_factual_claims(executive, common_context["claim_evidence"])
            if response_substantially_repeats_recent_answers(speech, prior_repeat_speeches):
                # The emergency response follows the Executive's response mode,
                # so it cannot collapse the entire conversation into one stock
                # phrase. The Executive still owns emotional escalation below.
                speech = repeat_intent_fallback(
                    repeat_assessment or {}, repeat_dynamics.consecutive_repeats
                )
                # The deterministic fallback deliberately makes no factual
                # assertion, so inherited claims from the rejected model text
                # must not be attached to it.
                executive = {**executive, "speech": speech, "factual_claims": []}
                claim_audit = []
                intent_fallback_used = True
        lobe_execution["repeat_deliberation"] = {
            "enabled": repeat_deliberation["enabled"],
            "assessment": repeat_assessment,
            "revision_used": repeat_revision_used,
            "intent_fallback_used": intent_fallback_used,
        }

        executive_escalation = str(executive.get("repeat_escalation", "hold"))
        repeat_dynamics, updated_topic_defensiveness, topic_state_changed = derive_repeat_dynamics(
            character=character,
            mutable_state=state.get("mutable_state", {}),
            review=repeat_review,
            user_turn_count=user_turn_count,
            escalation_decision=executive_escalation,
        )
        classification = classification.model_copy(update={"repeat_dynamics": repeat_dynamics})
        relationships = merge_historical_relationships(
            heuristic=historical_relationships(message=req.message, topic=topic, review=repeat_review),
            proposed=executive.get("historical_relationships", []),
            allowed_event_ids={str(event.get("id")) for event in transcript if event.get("id")},
        )

        turn_proposals: list[dict[str, Any]] = []
        # Persist the historical relationship separately from repeat posture.
        # A repeat is therefore one auditable edge type, not the only kind of
        # relationship the Executive can reason over in later turns.
        for relationship in relationships:
            target_id = relationship["target_event_id"]
            turn_proposals.append(MutationProposal(
                operation=MutationOperation.LINK_EVENTS,
                target=str(relationship["subject_key"]),
                value={
                    "from": target_id,
                    "to": user_event.id,
                    "relationship": relationship["relationship"],
                },
                evidence=[target_id, user_event.id or ""],
                confidence=float(relationship["confidence"]),
                epistemic_type=EpistemicType.OBSERVATION,
                reason="Record an Executive-reviewed historical relationship.",
            ).model_dump(mode="json"))
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
                    "Persist subject-specific defensiveness after the executive selected "
                    f"repeat_escalation={executive_escalation}."
                ),
            )
            turn_proposals.append(dynamics_proposal.model_dump(mode="json"))

        character_event = EventRecord(
            id=f"evt_{uuid.uuid4().hex}",
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
                "claim_verification": claim_audit,
                "historical_relationships": relationships,
                "responds_to": user_event.id,
                "interaction": classification.model_dump(mode="json"),
                "repeat_review": repeat_review,
                "lobe_execution": lobe_execution,
                "cognitive_priorities": cognitive_priorities,
                "weighted_arbitration": arbitration_plan,
            },
        )
        # Turn-level self-history is an infrastructure invariant, not a best-effort model suggestion.
        # Reflection handles higher-order consolidation later.
        pending_memories = [
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
            pending_memories.append(record)

        immediate = executive.get("mutations", [])
        if immediate:
            turn_proposals.extend(MutationProposal.model_validate(p).model_dump(mode="json") for p in immediate)
        committed_turn = await post_json(
            client,
            f"{MEMORY_URL}/sessions/{session_id}/turn",
            {
                "character_event": character_event.model_dump(mode="json"),
                "memories": [memory.model_dump(mode="json") for memory in pending_memories],
                "proposals": turn_proposals,
            },
        )
        stored_memories = committed_turn["memories"]
        mutation_results = committed_turn["mutation_results"]

        return {
            "session_id": session_id,
            "character_id": character_id,
            "message": speech,
            "interaction": classification.model_dump(mode="json"),
            "cognition": {
                "left": left_result,
                "right": right_result,
                "executive": executive,
                "claim_verification": claim_audit,
                "historical_relationships": relationships,
                "lobe_execution": lobe_execution,
                "repeat_review": repeat_review,
                "cognitive_priorities": cognitive_priorities,
                "weighted_arbitration": arbitration_plan,
                "timing_ms": {
                    "left": left_ms,
                    "right": right_ms,
                    "lobes_critical_path": max(left_ms, right_ms),
                    "executive": repeat_assessment_ms + executive_ms,
                    "executive_speech": executive_ms,
                    "executive_repeat_assessment": repeat_assessment_ms,
                    "executive_repeat_revision": repeat_revision_ms,
                    "model_critical_path": max(left_ms, right_ms) + repeat_assessment_ms + executive_ms,
                },
            },
            "memory_writes": stored_memories,
            "mutation_results": mutation_results,
        }


async def _session_meta(client: httpx.AsyncClient, session_id: str) -> dict[str, Any]:
    # Endpoint deliberately kept on memory service to avoid orchestrator-side session state.
    return await get_json(client, f"{MEMORY_URL}/sessions/{session_id}")


async def _reflect(session_id: str) -> ReflectionResult:
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
