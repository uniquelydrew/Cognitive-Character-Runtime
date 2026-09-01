"""Conversation-session lifecycle and durable reflection-retry persistence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from fastapi import HTTPException
from pydantic import BaseModel

from services.memory.models import ReflectionRetryClaim, ReflectionRetrySchedule
from services.memory.storage import now_iso


class SessionCreate(BaseModel):
    character_id: str


class SessionStore:
    def __init__(self, *, db: Callable[..., Any]) -> None:
        self.db = db

    def create(self, request: SessionCreate) -> dict[str, Any]:
        session_id = f"sess_{uuid.uuid4().hex}"
        timestamp = now_iso()
        with self.db() as conn:
            exists = conn.execute("SELECT 1 FROM characters WHERE id=?", (request.character_id,)).fetchone()
            if not exists:
                raise HTTPException(404, "Character not found")
            conn.execute(
                "INSERT INTO sessions(id, character_id, status, created_at) VALUES (?, ?, 'open', ?)",
                (session_id, request.character_id, timestamp),
            )
        return {"id": session_id, "character_id": request.character_id, "status": "open", "created_at": timestamp}

    def get(self, session_id: str) -> dict[str, Any]:
        with self.db() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Session not found")
        return dict(row)

    def events(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        with self.db() as conn:
            session = conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not session:
                raise HTTPException(404, "Session not found")
            rows = list(reversed(conn.execute(
                "SELECT * FROM events WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()))
        return [
            {**{key: row[key] for key in row.keys() if key != "metadata_json"}, "metadata": json.loads(row["metadata_json"])}
            for row in rows
        ]

    def close(self, session_id: str) -> dict[str, str]:
        timestamp = now_iso()
        with self.db() as conn:
            row = conn.execute("SELECT status, closed_at FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not row:
                raise HTTPException(404, "Session not found")
            if row["status"] == "closed":
                return {"id": session_id, "status": "closed", "closed_at": row["closed_at"]}
            conn.execute("UPDATE sessions SET status='closed', closed_at=? WHERE id=?", (timestamp, session_id))
        return {"id": session_id, "status": "closed", "closed_at": timestamp}

    @staticmethod
    def job_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "character_id": row["character_id"],
            "status": row["status"],
            "attempts": int(row["attempts"]),
            "last_error": row["last_error"],
            "next_attempt_at": row["next_attempt_at"],
            "lease_expires_at": row["lease_expires_at"],
            "completed_at": row["completed_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def schedule_reflection_retry(self, session_id: str, request: ReflectionRetrySchedule) -> dict[str, Any]:
        timestamp = now_iso()
        with self.db() as conn:
            session = conn.execute("SELECT character_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not session:
                raise HTTPException(404, "Session not found")
            existing = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
            if existing and existing["status"] == "completed":
                return {**self.job_payload(existing), "already_completed": True}
            if existing:
                conn.execute(
                    """
                    UPDATE reflection_jobs
                    SET status='pending', last_error=?, next_attempt_at=?, lease_expires_at=NULL, updated_at=?
                    WHERE session_id=?
                    """,
                    (request.error, timestamp, timestamp, session_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO reflection_jobs(
                        session_id, character_id, status, attempts, last_error, next_attempt_at,
                        lease_expires_at, completed_at, created_at, updated_at
                    ) VALUES (?, ?, 'pending', 0, ?, ?, NULL, NULL, ?, ?)
                    """,
                    (session_id, session["character_id"], request.error, timestamp, timestamp, timestamp),
                )
            row = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
        return self.job_payload(row)

    def claim_reflection_retry(self, request: ReflectionRetryClaim) -> dict[str, Any]:
        timestamp = now_iso()
        lease_until = (datetime.now(timezone.utc) + timedelta(seconds=request.lease_seconds)).isoformat()
        with self.db() as conn:
            row = conn.execute(
                """
                SELECT * FROM reflection_jobs
                WHERE (status='pending' AND next_attempt_at <= ?)
                   OR (status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                ORDER BY next_attempt_at, created_at
                LIMIT 1
                """,
                (timestamp, timestamp),
            ).fetchone()
            if not row:
                return {"job": None}
            conn.execute(
                """
                UPDATE reflection_jobs
                SET status='running', attempts=attempts + 1, lease_expires_at=?, updated_at=?
                WHERE session_id=?
                """,
                (lease_until, timestamp, row["session_id"]),
            )
            claimed = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (row["session_id"],)).fetchone()
        return {"job": self.job_payload(claimed)}

    def complete_reflection_retry(self, session_id: str) -> dict[str, Any]:
        timestamp = now_iso()
        with self.db() as conn:
            row = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                return {"found": False, "session_id": session_id}
            conn.execute(
                """
                UPDATE reflection_jobs
                SET status='completed', last_error=NULL, next_attempt_at=?, lease_expires_at=NULL,
                    completed_at=?, updated_at=?
                WHERE session_id=?
                """,
                (timestamp, timestamp, timestamp, session_id),
            )
            completed = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
        return {"found": True, **self.job_payload(completed)}

    def reschedule_reflection_retry(self, session_id: str, request: ReflectionRetrySchedule) -> dict[str, Any]:
        timestamp = now_iso()
        with self.db() as conn:
            row = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                raise HTTPException(404, "Reflection retry job not found")
            attempts = max(int(row["attempts"]), 1)
            delay_seconds = min(30 * (2 ** min(attempts - 1, 7)), 3_600)
            next_attempt = (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()
            conn.execute(
                """
                UPDATE reflection_jobs
                SET status='pending', last_error=?, next_attempt_at=?, lease_expires_at=NULL, updated_at=?
                WHERE session_id=?
                """,
                (request.error, next_attempt, timestamp, session_id),
            )
            scheduled = conn.execute("SELECT * FROM reflection_jobs WHERE session_id=?", (session_id,)).fetchone()
        return {"backoff_seconds": delay_seconds, **self.job_payload(scheduled)}

    def reflection_jobs(self, limit: int) -> dict[str, Any]:
        with self.db() as conn:
            rows = conn.execute("SELECT * FROM reflection_jobs ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        jobs = [self.job_payload(row) for row in rows]
        return {"jobs": jobs, "pending_count": sum(job["status"] in {"pending", "running"} for job in jobs)}
