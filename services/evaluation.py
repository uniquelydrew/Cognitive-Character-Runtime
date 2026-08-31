"""Paired correctness evaluation for multi-perspective and control runs.

Run the same scenario set against two deployments, save each response list as
JSON, then compare them with ``python -m services.evaluation --multi ...
--control ...``. The evaluator deliberately measures observable answers and
claim verification, not model chain-of-thought.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import httpx


def _terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def score_response(response: dict[str, Any], expected: list[str]) -> dict[str, float]:
    """Score answer correctness and evidence discipline for one benchmark row."""
    answer = _terms(str(response.get("message", "")))
    expected_sets = [_terms(value) for value in expected if _terms(value)]
    correctness = mean([float(items.issubset(answer)) for items in expected_sets]) if expected_sets else 1.0
    claims = response.get("cognition", {}).get("claim_verification", [])
    verified = sum(1 for claim in claims if claim.get("status") == "verified") if isinstance(claims, list) else 0
    rejected = sum(1 for claim in claims if claim.get("status") == "rejected") if isinstance(claims, list) else 0
    return {"correctness": correctness, "verified_claims": float(verified), "rejected_claims": float(rejected)}


def compare_runs(multi: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare paired scenario results keyed by a stable scenario id."""
    multi_by_id = {str(row["id"]): row for row in multi}
    control_by_id = {str(row["id"]): row for row in control}
    ids = sorted(set(multi_by_id) & set(control_by_id))
    if not ids:
        raise ValueError("No shared scenario IDs between multi and control results.")
    deltas: list[float] = []
    rows: list[dict[str, Any]] = []
    for scenario_id in ids:
        left, right = multi_by_id[scenario_id], control_by_id[scenario_id]
        expected = list(left.get("expected", right.get("expected", [])))
        multi_score, control_score = score_response(left, expected), score_response(right, expected)
        delta = multi_score["correctness"] - control_score["correctness"]
        deltas.append(delta)
        rows.append({"id": scenario_id, "multi": multi_score, "control": control_score, "correctness_delta": delta})
    return {
        "scenarios": len(ids),
        "mean_correctness_delta": mean(deltas),
        "multi_wins": sum(delta > 0 for delta in deltas),
        "control_wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "multi_failures": sum(not bool(multi_by_id[scenario_id].get("successful", True)) for scenario_id in ids),
        "control_failures": sum(not bool(control_by_id[scenario_id].get("successful", True)) for scenario_id in ids),
        "rows": rows,
    }


def run_benchmark(
    base_url: str, token: str, scenarios: list[dict[str, Any]], *, timeout_seconds: float = 300,
) -> list[dict[str, Any]]:
    """Execute a manifest against one deployed topology and retain raw output."""
    headers = {"X-API-Key": token}
    results: list[dict[str, Any]] = []
    # The client must outlive the orchestrator's default 240-second turn limit;
    # otherwise the evaluator would misclassify a still-valid response as a
    # benchmark failure.
    with httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds) as client:
        for scenario in scenarios:
            try:
                session = client.post("/sessions", json={"character_id": scenario["character_id"]})
            except httpx.HTTPError as exc:
                results.append(_transport_failure(scenario, exc))
                continue
            if session.is_error:
                results.append({
                    "id": scenario["id"], "expected": scenario["expected"], "successful": False,
                    "status_code": session.status_code, "error": _response_error(session),
                })
                continue
            try:
                reply = client.post(f"/sessions/{session.json()['id']}/chat", json={"message": scenario["message"]})
            except httpx.HTTPError as exc:
                results.append(_transport_failure(scenario, exc))
                continue
            if reply.is_error:
                results.append({
                    "id": scenario["id"], "expected": scenario["expected"], "successful": False,
                    "status_code": reply.status_code, "error": _response_error(reply),
                })
                continue
            results.append({"id": scenario["id"], "expected": scenario["expected"], "successful": True, **reply.json()})
    return results


def _response_error(response: httpx.Response) -> Any:
    """Keep a service failure observable without requiring an exception trace."""

    try:
        body = response.json()
    except ValueError:
        return response.text[:1_000]
    return body.get("detail", body) if isinstance(body, dict) else body


def _transport_failure(scenario: dict[str, Any], error: httpx.HTTPError) -> dict[str, Any]:
    return {
        "id": scenario["id"], "expected": scenario["expected"], "successful": False,
        "status_code": None, "error": f"transport_error: {type(error).__name__}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multi-perspective and control benchmark responses.")
    parser.add_argument("--multi", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--benchmark", type=Path, help="Scenario manifest to execute against one deployment.")
    parser.add_argument("--base-url", help="Orchestrator URL for --benchmark mode.")
    parser.add_argument("--token", help="API token for --benchmark mode.")
    parser.add_argument("--output", type=Path, help="Write benchmark responses to this JSON file.")
    args = parser.parse_args()
    if args.benchmark:
        if not all((args.base_url, args.token, args.output)):
            parser.error("--benchmark requires --base-url, --token, and --output")
        results = run_benchmark(args.base_url, args.token, json.loads(args.benchmark.read_text()))
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return
    if not args.multi or not args.control:
        parser.error("provide --multi and --control, or use --benchmark")
    print(json.dumps(compare_runs(json.loads(args.multi.read_text()), json.loads(args.control.read_text())), indent=2))


if __name__ == "__main__":
    main()
