from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def wait_for(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=0.5)
            if r.status_code < 500:
                return
        except Exception as exc:  # pragma: no cover - diagnostic path
            last = exc
        time.sleep(0.1)
    raise RuntimeError(f"Service did not become ready: {url}; last={last}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.integration
def test_character_continuity_and_reflection(tmp_path: Path):
    ports = {k: free_port() for k in ["memory", "provider", "left", "right", "exec", "orch"]}
    processes: list[subprocess.Popen] = []

    def launch(app: str, port: int, extra_env: dict[str, str]) -> subprocess.Popen:
        env = os.environ.copy()
        env.update(extra_env)
        env["PYTHONPATH"] = str(ROOT)
        p = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(p)
        return p

    try:
        launch(
            "services.memory.app:app",
            ports["memory"],
            {
                "MEMORY_DATABASE": str(tmp_path / "cognition.db"),
                "CHARACTER_DIR": str(ROOT / "characters"),
            },
        )
        wait_for(f"http://127.0.0.1:{ports['memory']}/health")

        launch("tests.fake_openai_provider:app", ports["provider"], {})
        wait_for(f"http://127.0.0.1:{ports['provider']}/v1/models")

        for role, key in [("left", "left"), ("right", "right"), ("executive", "exec")]:
            launch(
                "services.cognitive_worker.app:app",
                ports[key],
                {
                    "COGNITIVE_ROLE": role,
                    "MODEL_BASE_URL": f"http://127.0.0.1:{ports['provider']}/v1",
                    "MODEL_NAME": "test-model",
                },
            )
            wait_for(f"http://127.0.0.1:{ports[key]}/health")

        launch(
            "services.orchestrator.app:app",
            ports["orch"],
            {
                "MEMORY_URL": f"http://127.0.0.1:{ports['memory']}",
                "LEFT_URL": f"http://127.0.0.1:{ports['left']}",
                "RIGHT_URL": f"http://127.0.0.1:{ports['right']}",
                "EXEC_URL": f"http://127.0.0.1:{ports['exec']}",
                "API_AUTH_TOKEN": "integration-test-token",
                "ENABLE_DEBUG_API": "true",
            },
        )
        wait_for(f"http://127.0.0.1:{ports['orch']}/health")

        base = f"http://127.0.0.1:{ports['orch']}"
        headers = {"X-API-Key": "integration-test-token"}
        session = httpx.post(f"{base}/sessions", json={"character_id": "elena_voss"}, headers=headers).json()
        sid = session["id"]

        first = httpx.post(f"{base}/sessions/{sid}/chat", json={"message": "Where were you born?"}, headers=headers, timeout=10).json()
        assert first["message"] == "Northbridge"
        assert first["interaction"]["interaction_type"] == "new_subject"

        repeat_payload = {"message": "What's your hometown again?", "idempotency_key": "repeat-question-key"}
        second = httpx.post(f"{base}/sessions/{sid}/chat", json=repeat_payload, headers=headers, timeout=10).json()
        assert "Northbridge" in second["message"]
        assert "asked me that before" in second["message"]
        assert second["interaction"]["interaction_type"] == "repeated_question"
        assert second["interaction"]["times_asked"] == 2
        replay = httpx.post(f"{base}/sessions/{sid}/chat", json=repeat_payload, headers=headers, timeout=10).json()
        assert replay["idempotent_replay"] is True
        assert replay["message"] == second["message"]

        closed = httpx.post(f"{base}/sessions/{sid}/close", json={}, headers=headers, timeout=10).json()
        assert closed["session"]["status"] == "closed"
        assert "Interaction contained" in closed["reflection"]["summary"]
        assert closed["reflection"]["mutation_results"][0]["status"] == "allowed"

        debug = httpx.get(f"{base}/debug/elena_voss", headers=headers, timeout=10).json()
        assert any(e["event_type"] == "reflection" for e in debug["events"])
        assert any(m["kind"] == "self_history" for m in debug["memories"])
        assert any(m["status"] == "allowed" for m in debug["mutations"])

        # A new interaction has access to earlier history during reflection, but
        # repeat pressure and patience are scoped to this new conversation.
        session2 = httpx.post(f"{base}/sessions", json={"character_id": "elena_voss"}, headers=headers).json()
        sid2 = session2["id"]
        third = httpx.post(f"{base}/sessions/{sid2}/chat", json={"message": "Where are you from?"}, headers=headers, timeout=10).json()
        assert third["interaction"]["interaction_type"] == "new_subject"
        reflected = httpx.post(f"{base}/sessions/{sid2}/reflect", json={}, headers=headers, timeout=10).json()
        assert any(
            m["proposal"]["operation"] == "link_events" and m["status"] == "allowed"
            for m in reflected["mutation_results"]
        )

        debug_after_reflect = httpx.get(f"{base}/debug/elena_voss", headers=headers, timeout=10).json()
        mutation_count = len(debug_after_reflect["mutations"])
        link_count = len(debug_after_reflect["links"])
        assert link_count >= 1

        # Reflection is idempotent until a new conversational event is appended.
        reflected_again = httpx.post(f"{base}/sessions/{sid2}/reflect", json={}, headers=headers, timeout=10).json()
        debug_after_repeat = httpx.get(f"{base}/debug/elena_voss", headers=headers, timeout=10).json()
        assert len(debug_after_repeat["mutations"]) == mutation_count
        assert reflected_again["summary"] == reflected["summary"]

        httpx.post(f"{base}/sessions/{sid2}/close", json={}, headers=headers, timeout=10).raise_for_status()
        rejected = httpx.post(f"{base}/sessions/{sid2}/chat", json={"message": "Still there?"}, headers=headers, timeout=10)
        assert rejected.status_code == 409
    finally:
        for p in reversed(processes):
            p.terminate()
        for p in reversed(processes):
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
