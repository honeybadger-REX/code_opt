from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import TestResult

OPTIMIZED = """\
def find_common(a: list[int], b: list[int]) -> list[int]:
    from collections import Counter

    counts = Counter(b)
    result = []
    for x in a:
        n = counts.get(x, 0)
        if n:
            result.extend([x] * n)
    return result
"""


class FakeLLM:
    def __init__(self, source: str):
        self.source = source

    def available(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "strategy": "Counter lookup",
                "rationale": "preserves order and duplicates",
                "risk": "low",
                "optimized_function": self.source,
            }
        )


class DisabledLLM:
    def available(self) -> bool:
        return False

    def complete(self, prompt: str) -> str:  # pragma: no cover
        raise AssertionError("disabled")


def fake_tests(repo_dir, timeout=300.0, extra_args=None):
    return TestResult(
        ran=True,
        passed=True,
        passed_count=2,
        failed_count=0,
        error_count=0,
        skipped_count=0,
        returncode=0,
        duration_seconds=0.01,
        output="fake",
    )


def test_health_and_frontend():
    client = TestClient(create_app(llm=DisabledLLM()))
    assert client.get("/health").json()["status"] == "ok"
    response = client.get("/")
    assert response.status_code == 200
    assert "CodeOptimizer AI" in response.text


def test_tools_endpoint():
    client = TestClient(create_app(llm=DisabledLLM()))
    tools = client.get("/tools").json()["tools"]
    names = {tool["name"] for tool in tools}
    assert {
        "read_file",
        "write_file",
        "search_repository",
        "analyze_python",
        "run_tests",
        "benchmark",
        "git_diff",
        "git_revert",
    } <= names


def test_analyze_endpoint():
    client = TestClient(create_app(llm=DisabledLLM()))
    response = client.post("/analyze", json={"repo_path": "examples"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    analysis = data["analysis"]
    kinds = {finding["kind"] for finding in analysis["findings"]}
    assert "nested_loops" in kinds
    assert analysis["metrics"]["files"] >= 2


def test_analyze_rejects_bad_path():
    client = TestClient(create_app(llm=DisabledLLM()))
    response = client.post("/analyze", json={"repo_path": "does/not/exist"})
    assert response.status_code == 400


def test_optimize_flow(tmp_path, monkeypatch, sample_repo):
    repo, _ = sample_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests)
    app = create_app(llm=FakeLLM(OPTIMIZED))
    client = TestClient(app)

    response = client.post(
        "/optimize",
        json={
            "repo_path": str(repo),
            "max_iterations": 1,
            "benchmark_size": 300,
            "benchmark_repeats": 2,
        },
    )
    assert response.status_code == 200
    job = response.json()
    job_id = job["job_id"]

    status = None
    for _ in range(150):
        status = client.get(f"/status/{job_id}").json()
        if status["status"] in {"completed", "failed", "aborted"}:
            break
        time.sleep(0.1)
    assert status is not None
    assert status["status"] == "completed", status

    report_response = client.get(f"/report/{job_id}")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["job_id"] == job_id
    assert report["kept"] == 1
    assert report["final_diff"]
    assert report["iterations"][0]["benchmark"]["available"] is True

    assert client.get("/status").json()["job_id"] == job_id
    assert client.get("/report").status_code == 200
    jobs = client.get("/jobs").json()["jobs"]
    assert any(entry["job_id"] == job_id for entry in jobs)


def test_unknown_job():
    client = TestClient(create_app(llm=DisabledLLM()))
    assert client.get("/status/nope").status_code == 404
    assert client.get("/report/nope").status_code == 404


def test_report_payload_for_failed_job_without_report():
    from app.main import _report_payload

    payload = _report_payload(
        {
            "job_id": "abc123",
            "repo_path": "repo",
            "status": "failed",
            "stage": "ERROR",
            "error": "RuntimeError: boom",
            "failure": {
                "stage": "TEST",
                "type": "RuntimeError",
                "message": "boom",
                "traceback": "Traceback ...",
            },
            "traceback": "Traceback ...",
            "logs": [],
        }
    )
    assert payload["status"] == "failed"
    assert payload["error"] == "RuntimeError: boom"
    assert payload["iterations"] == []
    assert payload["failure"]["type"] == "RuntimeError"
    assert payload["failure"]["stage"] == "TEST"
    assert payload["traceback"] == "Traceback ..."


def test_background_job_crash_is_reported(tmp_path, monkeypatch):
    class CrashingAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, *args, **kwargs):
            raise RuntimeError("background boom")

    monkeypatch.setattr("app.main.Agent", CrashingAgent)
    app = create_app(llm=DisabledLLM())
    client = TestClient(app)

    response = client.post(
        "/optimize",
        json={"repo_path": str(tmp_path), "max_iterations": 1},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = None
    for _ in range(100):
        status = client.get(f"/status/{job_id}").json()
        if status["status"] in {"completed", "failed", "aborted"}:
            break
        time.sleep(0.05)
    assert status is not None
    assert status["status"] == "failed"
    assert status["error"] == "RuntimeError: background boom"
    assert status["failure"]["type"] == "RuntimeError"
    assert status["failure"]["stage"] == "OBSERVE"

    report = client.get(f"/report/{job_id}").json()
    assert report["status"] == "failed"
    assert report["outcome"] == "failed"
    assert report["failure"]["type"] == "RuntimeError"
    assert report["failure"]["stage"] == "OBSERVE"
    assert "RuntimeError: background boom" in report["failure"]["traceback"]
    assert report["logs"]


def test_report_payload_pending_job_returns_202():
    from app.main import _report_payload

    response = _report_payload(
        {"job_id": "abc123", "status": "running", "stage": "TEST", "report": None}
    )
    assert response.status_code == 202


def test_optimize_rejects_bad_path():
    client = TestClient(create_app(llm=DisabledLLM()))
    response = client.post("/optimize", json={"repo_path": "does/not/exist"})
    assert response.status_code == 400
