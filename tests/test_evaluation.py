from services.evaluation import compare_runs


def test_paired_evaluation_reports_observable_correctness_delta() -> None:
    report = compare_runs(
        [{"id": "birthplace", "expected": ["Northbridge"], "message": "Northbridge.", "cognition": {"claim_verification": []}}],
        [{"id": "birthplace", "expected": ["Northbridge"], "message": "Greyhaven.", "cognition": {"claim_verification": []}}],
    )
    assert report["mean_correctness_delta"] == 1.0
    assert report["multi_wins"] == 1
