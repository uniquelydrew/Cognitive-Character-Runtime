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
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multi-perspective and control benchmark responses.")
    parser.add_argument("--multi", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_runs(json.loads(args.multi.read_text()), json.loads(args.control.read_text())), indent=2))


if __name__ == "__main__":
    main()
