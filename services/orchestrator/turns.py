"""The full live-turn cognitive pipeline, separate from HTTP route concerns."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException

from services.common import (
    CognitiveRequest,
    EpistemicType,
    EventRecord,
    MemoryRecord,
    MutationOperation,
    MutationProposal,
)
from services.orchestrator.claims import claim_evidence_catalog, verify_factual_claims
from services.orchestrator.cognitive_policy import normalize_topic, resolve_cognitive_priorities, weighted_arbitration_plan
from services.orchestrator.executive import infer_verified_turn
from services.orchestrator.relationships import historical_relationships, merge_historical_relationships
from services.orchestrator.repeat_dynamics import (
    derive_repeat_dynamics,
    repeat_intent_fallback,
    response_substantially_repeats_recent_answers,
)


async def execute_turn(
    session_id: str,
    request: Any,
    *,
    memory_url: str,
    left_url: str,
    right_url: str,
    executive_url: str,
    worker_timeout_seconds: float,
    executive_only_control: bool,
    semantic_embedding_model: str,
    semantic_repeat_similarity_threshold: float,
    model_metrics: dict[str, dict[str, int | float | None]],
    get_session: Callable[[httpx.AsyncClient, str], Awaitable[dict[str, Any]]],
    load_character: Callable[[httpx.AsyncClient, str], Awaitable[tuple[Any, dict[str, Any]]]],
    get_json: Callable[[httpx.AsyncClient, str], Awaitable[Any]],
    post_json: Callable[[httpx.AsyncClient, str, dict[str, Any]], Awaitable[Any]],
    infer_timed: Callable[[httpx.AsyncClient, str, CognitiveRequest, str], Awaitable[tuple[dict[str, Any], int]]],
    bounded_lobe_transcript: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    semantic_repeat_evidence: Callable[[httpx.AsyncClient, str, list[dict[str, Any]]], Awaitable[dict[str, Any]]],
    immediate_repeat_lobe_reuse: Callable[..., dict[str, Any] | None],
    executive_repeat_review: Callable[..., dict[str, Any]],
    interaction_classification: Callable[..., Any],
) -> dict[str, Any]:
    """Execute and durably commit one conversational turn."""

    req = request
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    async with httpx.AsyncClient(timeout=httpx.Timeout(worker_timeout_seconds, connect=10)) as client:
        # Resolve the session from its event stream if it already has events. For a new session,
        # the memory service doesn't expose session metadata yet, so resolve through a small helper call.
        # We store character_id in each event and also accept a private metadata route via SQLite service.
        session_meta = await get_session(client, session_id)
        if session_meta.get("status") != "open":
            raise HTTPException(409, "Interaction is already closed")
        character_id = session_meta["character_id"]
        character, state = await load_character(client, character_id)

        topic = normalize_topic(req.message)
        prior_transcript = await get_json(client, f"{memory_url}/sessions/{session_id}/events?limit=200")
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
        classification = interaction_classification(
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
                await post_json(client, f"{memory_url}/events", user_event.model_dump(mode="json"))
            )

        memories = await get_json(client, f"{memory_url}/memories/{character_id}?topic={topic}&limit=20")
        knowledge_query = urlencode({"query": req.message, "limit": 12})
        knowledge_context = await get_json(
            client, f"{memory_url}/knowledge/for-character/{character_id}?{knowledge_query}"
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

        if executive_only_control:
            # Benchmark control: preserve the Executive prompt and all
            # retrieval/state inputs while withholding independent lobe
            # analyses. This makes a paired comparison attributable to the
            # multi-perspective pipeline rather than different knowledge.
            left_result = {"topic": topic, "fact_refs": [], "constraints": [], "action": "control"}
            right_result = {"action": "control", "affect": {}, "tone": "neutral", "risk": "unknown", "association_keys": []}
            left_ms = right_ms = 0
            lobe_execution = {"mode": "executive_only_control", "transcript_events": len(lobe_transcript)}
        elif lobe_reuse:
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
            left_task = infer_timed(client, left_url, left_req, "left")
            right_task = infer_timed(client, right_url, right_req, "right")
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
        transcript = await get_json(client, f"{memory_url}/sessions/{session_id}/events?limit=200")
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
                "model": semantic_embedding_model,
                "threshold": semantic_repeat_similarity_threshold,
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
                client, executive_url, assessment_req, "executive"
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
        executive, claim_audit, executive_ms, claim_revision_used = await infer_verified_turn(
            client, executive_url, exec_req, common_context["claim_evidence"], model_metrics
        )
        speech = executive["speech"]

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
            executive, repeat_revision_ms = await infer_timed(client, executive_url, retry_req, "executive")
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
        lobe_execution["claim_coverage_revision_used"] = claim_revision_used

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
            f"{memory_url}/sessions/{session_id}/turn",
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
