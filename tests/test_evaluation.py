from services.evaluation import compare_runs, run_benchmark


def test_paired_evaluation_reports_observable_correctness_delta() -> None:
    report = compare_runs(
        [{"id": "birthplace", "expected": ["Northbridge"], "message": "Northbridge.", "cognition": {"claim_verification": []}}],
        [{"id": "birthplace", "expected": ["Northbridge"], "message": "Greyhaven.", "cognition": {"claim_verification": []}}],
    )
    assert report["mean_correctness_delta"] == 1.0
    assert report["multi_wins"] == 1


def test_benchmark_runner_retains_scenario_identity(monkeypatch) -> None:
    class Response:
        def __init__(self, value): self.value, self.is_error = value, False
        def raise_for_status(self): pass
        def json(self): return self.value
    class Client:
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def post(self, url, json):
            return Response({"id": "sess_test"} if url == "/sessions" else {"message": "Northbridge", "cognition": {}})
    monkeypatch.setattr("services.evaluation.httpx.Client", Client)
    result = run_benchmark("http://example", "token", [{"id": "case", "character_id": "elena_voss", "message": "Where?", "expected": ["Northbridge"]}])
    assert result[0]["id"] == "case"
    assert result[0]["successful"] is True


def test_benchmark_runner_retains_rejected_turns(monkeypatch) -> None:
    class Response:
        def __init__(self, value, *, error=False):
            self.value, self.is_error, self.status_code = value, error, 422 if error else 200
        def json(self): return self.value
    class Client:
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def post(self, url, json):
            return Response({"id": "sess_test"}) if url == "/sessions" else Response({"detail": "citation rejected"}, error=True)
    monkeypatch.setattr("services.evaluation.httpx.Client", Client)

    result = run_benchmark("http://example", "token", [{"id": "case", "character_id": "elena_voss", "message": "Where?", "expected": ["Northbridge"]}])

    assert result == [{"id": "case", "expected": ["Northbridge"], "successful": False, "status_code": 422, "error": "citation rejected"}]
