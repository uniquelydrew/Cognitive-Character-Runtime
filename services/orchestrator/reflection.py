"""Reflection generation independent of the FastAPI route lifecycle.

The orchestrator owns locks, time limits, and retry scheduling.  This module
owns the idempotent cognitive work performed once a reflection is requested.
Keeping that distinction explicit lets retries and manual reflection use the
same durable generation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from fastapi import HTTPException

from services.common import CognitiveRequest, EventRecord, MutationProposal


JsonReader = Callable[[httpx.AsyncClient, str], Awaitable[Any]]
JsonWriter = Callable[[httpx.AsyncClient, str, dict[str, Any]], Awaitable[Any]]
SessionReader = Callable[[httpx.AsyncClient, str], Awaitable[dict[str, Any]]]
CharacterLoader = Callable[[httpx.AsyncClient, str], Awaitable[tuple[Any, dict[str, Any]]]]
InferenceRunner = Callable[[httpx.AsyncClient, str, CognitiveRequest, str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ReflectionOutput:
    """The durable result of one reflection pass."""

    session_id: str
    summary: str
    mutation_results: list[dict[str, Any]]
    executive: dict[str, Any]


async def reflect_session(
    session_id: str,
    *,
    memory_url: str,
    executive_url: str,
    worker_timeout_seconds: float,
    get_session: SessionReader,
    load_character: CharacterLoader,
    get_json: JsonReader,
    post_json: JsonWriter,
    infer: InferenceRunner,
) -> ReflectionOutput:
    """Generate and persist one idempotent executive reflection."""

    timeout = httpx.Timeout(worker_timeout_seconds, connect=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        session_meta = await get_session(client, session_id)
        character_id = session_meta["character_id"]
        character, state = await load_character(client, character_id)
        transcript = await get_json(client, f"{memory_url}/sessions/{session_id}/events")

        # Once a reflection follows every conversation event, it is authoritative
        # until a new conversational event makes a fresh pass necessary.
        last_reflection_index = max(
            (i for i, event in enumerate(transcript) if event["event_type"] == "reflection"),
            default=-1,
        )
        last_conversation_index = max(
            (
                i
                for i, event in enumerate(transcript)
                if event["event_type"] in {"user_message", "character_message"}
            ),
            default=-1,
        )
        if last_reflection_index > last_conversation_index:
            prior = transcript[last_reflection_index]
            metadata = prior.get("metadata", {})
            return ReflectionOutput(
                session_id=session_id,
                summary=prior.get("content", ""),
                mutation_results=metadata.get("mutation_results", []),
                executive=metadata.get("executive", {}),
            )

        memories = await get_json(client, f"{memory_url}/memories/{character_id}?limit=50")
        current_event_ids = {event["id"] for event in transcript}
        topics = sorted({event.get("topic") for event in transcript if event.get("topic")})
        related_history: dict[str, list[dict[str, Any]]] = {}
        for topic in topics:
            history = await get_json(
                client,
                f"{memory_url}/interaction-history/{character_id}?topic={topic}&limit=50",
            )
            prior_events = [event for event in history.get("events", []) if event["id"] not in current_event_ids]
            if prior_events:
                related_history[str(topic)] = prior_events

        reflection_request = CognitiveRequest(
            character=character,
            mode="reflection",
            transcript=[
                {
                    "event_id": event["id"],
                    "event_type": event["event_type"],
                    "actor": event["actor"],
                    "content": event["content"],
                    "topic": event.get("topic"),
                }
                for event in transcript
            ],
            context={
                "memories": memories,
                "mutable_state": state.get("mutable_state", {}),
                "beliefs": state.get("beliefs", {}),
                "goals": state.get("goals", []),
                "related_history": related_history,
            },
        )
        executive = await infer(client, executive_url, reflection_request, "executive")
        summary = str(executive.get("summary", "")).strip()
        if not summary:
            raise HTTPException(502, "Executive produced no reflection summary")

        # The summary is sourced durable memory even if no additional insight was
        # proposed, so it passes through the same mutation validator/audit path.
        summary_proposal = MutationProposal(
            operation="add_memory",
            target="interaction_summary",
            value=summary,
            evidence=[str(event["id"]) for event in transcript if event.get("id")],
            confidence=1.0,
            epistemic_type="observation",
            reason="Persist the completed interaction summary with raw-event provenance.",
        ).model_dump(mode="json")
        proposals_raw = [
            proposal
            for proposal in executive.get("mutations", [])
            if not (
                proposal.get("operation") == "add_memory"
                and proposal.get("target") == "interaction_summary"
                and str(proposal.get("value", "")) == summary
            )
        ]
        proposals_raw.insert(0, summary_proposal)
        proposals = [MutationProposal.model_validate(proposal).model_dump(mode="json") for proposal in proposals_raw]
        mutation_results: list[dict[str, Any]] = []
        if proposals:
            mutation_results = await post_json(
                client,
                f"{memory_url}/mutations/{character_id}",
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
        await post_json(client, f"{memory_url}/events", reflection_event.model_dump(mode="json"))

    return ReflectionOutput(
        session_id=session_id,
        summary=summary,
        mutation_results=mutation_results,
        executive=executive,
    )
