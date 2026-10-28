from __future__ import annotations

import json
from pathlib import Path
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


def test_job_files_preview_and_safe_import_flow(tmp_path, monkeypatch, sample_repo):
    repo, _ = sample_repo
    orig_content = (repo / "slow_mod.py").read_text(encoding="utf-8")
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
    job_id = response.json()["job_id"]

    for _ in range(150):
        status = client.get(f"/status/{job_id}").json()
        if status["status"] in {"completed", "failed", "aborted"}:
            break
        time.sleep(0.1)
    assert status["status"] == "completed"

    # 1. GET /jobs/{job_id}/files
    files_res = client.get(f"/jobs/{job_id}/files")
    assert files_res.status_code == 200
    files_data = files_res.json()
    assert files_data["job_id"] == job_id
    assert len(files_data["files"]) == 1
    file_info = files_data["files"][0]
    assert file_info["file"] == "slow_mod.py"
    assert file_info["status"] == "modified"
    assert "find_common" in file_info["functions"]

    # 2. GET /jobs/{job_id}/files/{file_path}/diff
    diff_res = client.get(f"/jobs/{job_id}/files/slow_mod.py/diff")
    assert diff_res.status_code == 200
    diff_data = diff_res.json()
    assert diff_data["file"] == "slow_mod.py"
    assert diff_data["status"] == "modified"
    assert diff_data["additions"] > 0
    assert "Counter" in diff_data["optimized_content"]
    assert "for y in b:" in diff_data["original_content"]
    assert "-    for x in a:" in diff_data["diff"] or "+    counts = Counter(b)" in diff_data["diff"]

    # Original file is NOT modified before import
    assert (repo / "slow_mod.py").read_text(encoding="utf-8") == orig_content

    # 3. POST /jobs/{job_id}/files/{file_path}/import
    import_res = client.post(f"/jobs/{job_id}/files/slow_mod.py/import")
    assert import_res.status_code == 200
    import_data = import_res.json()
    assert import_data["status"] == "ok"
    assert "Successfully imported" in import_data["message"]
    assert Path(import_data["backup_path"]).is_file()
    assert Path(import_data["patch_path"]).is_file()

    # Original repo is now safely updated
    imported_content = (repo / "slow_mod.py").read_text(encoding="utf-8")
    assert "Counter" in imported_content

    # Backup contains original content
    backup_content = Path(import_data["backup_path"]).read_text(encoding="utf-8")
    assert backup_content == orig_content


def test_job_files_empty_and_rejects_unaccepted_candidate(tmp_path):
    app = create_app(llm=DisabledLLM())
    client = TestClient(app)

    # Fake a job with only a rejected candidate
    ws_dir = tmp_path / "fake_workspace"
    ws_dir.mkdir()
    (ws_dir / "bad.py").write_text("def f(): pass", encoding="utf-8")
    repo_dir = tmp_path / "fake_repo"
    repo_dir.mkdir()
    (repo_dir / "bad.py").write_text("def f(): return 1", encoding="utf-8")

    store = app.state.store
    job = store.create(str(repo_dir), {"max_iterations": 1})
    job_id = job["job_id"]
    store.start(job_id)
    store.finish(
        job_id,
        {
            "status": "completed",
            "workspace": str(ws_dir),
            "iterations": [
                {
                    "index": 1,
                    "finding": {"file": "bad.py", "function": "f"},
                    "verification": {"status": "rejected", "action": "REJECT"},
                }
            ],
            "kept": 0,
            "reverted": 0,
            "rejected": 1,
        },
    )

    # 1. /jobs/{job_id}/files should be empty
    res = client.get(f"/jobs/{job_id}/files")
    assert res.status_code == 200
    assert res.json()["files"] == []

    # 2. Previewing rejected candidate file should be forbidden (403)
    diff_res = client.get(f"/jobs/{job_id}/files/bad.py/diff")
    assert diff_res.status_code == 403
    assert "not modified by an accepted" in diff_res.json()["detail"]

    # 3. Importing rejected candidate file should be forbidden (403)
    import_res = client.post(f"/jobs/{job_id}/files/bad.py/import")
    assert import_res.status_code == 403
    assert "Only accepted and verified candidates" in import_res.json()["detail"]

    # 4. Path traversal should be rejected (400 or 403 or 404)
    trav_res = client.get(f"/jobs/{job_id}/files/..%2F..%2Fetc%2Fpasswd/diff")
    assert trav_res.status_code in {400, 403}

    # 5. Non-existent job (404)
    missing_res = client.get("/jobs/nonexistent123/files")
    assert missing_res.status_code == 404
