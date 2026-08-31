"""Per-session locks with reference-counted lifecycle cleanup.

This is process-local. Deploy exactly one orchestrator replica per memory
database; cross-process serialization belongs in a database lease if scaling is
introduced.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class SessionLockRegistry:
    def __init__(self) -> None:
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, session_id: str, timeout: float):
        async with self._guard:
            lock, users = self._locks.get(session_id, (asyncio.Lock(), 0))
            self._locks[session_id] = (lock, users + 1)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            try:
                yield
            finally:
                lock.release()
        finally:
            async with self._guard:
                lock, users = self._locks[session_id]
                if users <= 1 and not lock.locked():
                    self._locks.pop(session_id, None)
                else:
                    self._locks[session_id] = (lock, users - 1)
