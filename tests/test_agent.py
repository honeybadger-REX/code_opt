from __future__ import annotations

import json
from pathlib import Path

from app.agent import Agent
from app.models import TestResult


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


class DisabledLLM:
    def available(self) -> bool:
        return False

    def complete(self, prompt: str) -> str:  # pragma: no cover
        raise AssertionError("disabled LLM should not be called")


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

WRONG = """\
def find_common(a: list[int], b: list[int]) -> list[int]:
    return []
"""

SLOWER = """\
def find_common(a: list[int], b: list[int]) -> list[int]:
    from collections import Counter

    counts = Counter(b)
    result = []
    for x in a:
        for _ in range(20000):
            pass
        n = counts.get(x, 0)
        if n:
            result.extend([x] * n)
    return result
"""

WRONG_HAS_DUPES = """\
def has_duplicates(items: list[int]) -> bool:
    return False
"""

SLOWER_HAS_DUPES = """\
def has_duplicates(items: list[int]) -> bool:
    seen = set()
    for item in items:
        for _ in range(200000):
            pass
        if item in seen:
            return True
        seen.add(item)
    return False
"""


def payload(source: str) -> dict:
    return {
        "strategy": "Counter lookup",
        "rationale": "counts once, preserves order and duplicates",
        "risk": "low",
        "optimized_function": source,
    }


def fake_tests(passed: bool = True, ran: bool = True):
    def runner(repo_dir, timeout=300.0, extra_args=None):
        return TestResult(
            ran=ran,
            passed=passed and ran,
            passed_count=2 if passed and ran else 0,
            failed_count=0 if passed else 1,
            error_count=0,
            skipped_count=0,
            returncode=0 if passed else 1,
            duration_seconds=0.01,
            output="fake test output",
            reason=None if passed and ran else "fake failure",
        )

    return runner


def test_agent_keeps_verified_optimization(tmp_path, monkeypatch, sample_repo):
    repo, original = sample_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))
    agent = Agent(llm=FakeLLM(payload(OPTIMIZED)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["status"] == "completed"
    assert report["kept"] == 1
    assert report["reverted"] == 0
    iteration = report["iterations"][0]
    assert iteration["verification"]["status"] == "kept"
    assert iteration["proposal"]["planner"] == "llm"
    assert iteration["tests"]["passed"] is True
    assert iteration["benchmark"]["available"] is True
    assert iteration["benchmark"]["outputs_match"] is True
    assert iteration["benchmark"]["improvement_pct"] > 5
    assert "counts" in report["final_diff"]
    # The original repository must not be modified.
    assert (repo / "slow_mod.py").read_text(encoding="utf-8") == original


def test_agent_reverts_when_tests_fail(tmp_path, monkeypatch, membership_repo):
    repo, original = membership_repo
    calls = {"n": 0}

    def runner(repo_dir, timeout=300.0, extra_args=None):
        calls["n"] += 1
        passed = calls["n"] == 1  # baseline passes, modified version fails
        return TestResult(
            ran=True,
            passed=passed,
            passed_count=2 if passed else 0,
            failed_count=0 if passed else 1,
            error_count=0,
            skipped_count=0,
            returncode=0 if passed else 1,
            duration_seconds=0.01,
            output="fake",
            reason=None if passed else "tests failed",
        )

    monkeypatch.setattr("app.tools.run_pytest", runner)
    agent = Agent(llm=FakeLLM(payload(WRONG_HAS_DUPES)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["status"] == "completed"
    assert report["kept"] == 0
    assert report["reverted"] == 1
    iteration = report["iterations"][0]
    assert iteration["verification"]["status"] == "reverted"
    assert report["final_diff"].strip() == ""
    assert (repo / "membership_mod.py").read_text(encoding="utf-8") == original


def test_agent_tries_fallback_after_llm_test_failure(tmp_path, monkeypatch, sample_repo):
    repo, original = sample_repo
    calls = {"n": 0}

    def runner(repo_dir, timeout=300.0, extra_args=None):
        calls["n"] += 1
        passed = calls["n"] != 2  # baseline passes, LLM proposal fails, fallback passes
        return TestResult(
            ran=True,
            passed=passed,
            passed_count=2 if passed else 0,
            failed_count=0 if passed else 1,
            error_count=0,
            skipped_count=0,
            returncode=0 if passed else 1,
            duration_seconds=0.01,
            output="fake",
            reason=None if passed else "tests failed",
        )

    monkeypatch.setattr("app.tools.run_pytest", runner)
    agent = Agent(llm=FakeLLM(payload(WRONG)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["kept"] == 1
    iteration = report["iterations"][0]
    assert len(iteration["attempts"]) == 2
    assert iteration["attempts"][0]["verification"]["status"] == "reverted"
    assert iteration["attempts"][1]["verification"]["status"] == "kept"
    assert iteration["proposal"]["planner"] == "deterministic"
    assert (repo / "slow_mod.py").read_text(encoding="utf-8") == original


def test_agent_reverts_when_not_faster(tmp_path, monkeypatch, membership_repo):
    repo, original = membership_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))
    agent = Agent(llm=FakeLLM(payload(SLOWER_HAS_DUPES)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["kept"] == 0
    assert report["reverted"] == 1
    iteration = report["iterations"][0]
    assert iteration["benchmark"]["available"] is True
    assert iteration["verification"]["status"] == "reverted"
    assert any("below the required" in reason for reason in iteration["verification"]["reasons"])
    assert (repo / "membership_mod.py").read_text(encoding="utf-8") == original


def test_agent_uses_deterministic_fallback(tmp_path, monkeypatch, sample_repo):
    repo, _ = sample_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))
    agent = Agent(llm=DisabledLLM(), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["kept"] == 1
    iteration = report["iterations"][0]
    assert iteration["proposal"]["planner"] == "deterministic"
    assert "Counter" in iteration["proposal"]["source"]


def test_agent_falls_back_when_llm_returns_garbage(tmp_path, monkeypatch, sample_repo):
    repo, _ = sample_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))
    llm = FakeLLM("this is not json")
    agent = Agent(llm=llm, benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert llm.calls == 2  # initial attempt + one retry
    assert report["kept"] == 1
    iteration = report["iterations"][0]
    assert iteration["proposal"]["planner"] == "deterministic"
    assert iteration["planner_errors"]


def test_agent_reports_baseline_test_failure_without_aborting_infrastructure(
    tmp_path, monkeypatch, sample_repo
):
    repo, original = sample_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(passed=False))
    agent = Agent(llm=FakeLLM(payload(OPTIMIZED)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["status"] == "completed"
    assert report["outcome"] == "baseline_tests_failed"
    assert report["kept"] == 0
    assert "Baseline tests fail" in report["error"]
    assert report["failure"]["stage"] == "VALIDATE"
    assert report["failure"]["type"] == "BaselineTestFailure"
    assert report["final_diff"].strip() == ""
    assert (repo / "slow_mod.py").read_text(encoding="utf-8") == original


def test_agent_falls_back_to_generated_validation_when_pytest_collects_nothing(
    tmp_path, monkeypatch, sample_repo
):
    repo, _ = sample_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(ran=False, passed=False))
    agent = Agent(llm=FakeLLM(payload(OPTIMIZED)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["status"] == "completed"
    assert report["validation"]["strategy"] == "GENERATED_BEHAVIORAL_TESTS"
    assert report["validation"]["confidence"] == "MEDIUM"
    assert report["baseline_diagnostic"]["reason"] == "fake failure"
    assert report["kept"] == 1
    assert report["iterations"][0]["validation"]["passed"] is True


def test_agent_no_test_diagnostics_are_preserved_while_falling_back(
    tmp_path, monkeypatch, sample_repo
):
    repo, _ = sample_repo

    def runner(repo_dir, timeout=300.0, extra_args=None):
        return TestResult(
            ran=False,
            passed=False,
            passed_count=0,
            failed_count=0,
            error_count=0,
            skipped_count=0,
            returncode=5,
            duration_seconds=0.01,
            output="collected 0 items\nno tests ran",
            reason="no tests were collected; correctness cannot be verified (pytest exit code 5)",
            command=["python", "-m", "pytest"],
            cwd=str(repo_dir),
            python_executable="python",
        )

    monkeypatch.setattr("app.tools.run_pytest", runner)
    agent = Agent(llm=FakeLLM(payload(OPTIMIZED)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["status"] == "completed"
    assert report["validation"]["strategy"] == "GENERATED_BEHAVIORAL_TESTS"
    diagnostic = report["baseline_diagnostic"]
    assert diagnostic["runner"] == "pytest"
    assert diagnostic["returncode"] == 5
    assert "collected 0 items" in diagnostic["output"]
    assert diagnostic["command"] == ["python", "-m", "pytest"]
    assert report["kept"] == 1


def test_agent_abort_reports_baseline_test_output(tmp_path, monkeypatch, sample_repo):
    repo, _ = sample_repo

    def runner(repo_dir, timeout=300.0, extra_args=None):
        return TestResult(
            ran=True,
            passed=False,
            passed_count=0,
            failed_count=1,
            error_count=0,
            skipped_count=0,
            returncode=1,
            duration_seconds=0.01,
            output="FAILED test_slow_mod.py::test_basic - AssertionError: broken baseline",
            reason="pytest exited with code 1: 1 failed, 0 errors",
        )

    monkeypatch.setattr("app.tools.run_pytest", runner)
    agent = Agent(llm=FakeLLM(payload(OPTIMIZED)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["status"] == "completed"
    assert report["outcome"] == "baseline_tests_failed"
    assert "1 failed" in report["error"]
    failure = report["failure"]
    assert failure["type"] == "BaselineTestFailure"
    assert failure["returncode"] == 1
    assert "AssertionError" in failure["output"]
    assert report["final_diff"].strip() == ""


def test_agent_rejects_candidate_when_validation_raises(tmp_path, monkeypatch, membership_repo):
    repo, original = membership_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))

    def boom(*args, **kwargs):
        raise RuntimeError("unsupported annotation for benchmark")

    monkeypatch.setattr("app.agent.infer_params", boom)
    agent = Agent(llm=FakeLLM(payload(WRONG_HAS_DUPES)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["status"] == "completed"
    assert report["kept"] == 0
    assert report["rejected"] == 1
    attempt = report["iterations"][0]["attempts"][0]
    assert attempt["verification"]["status"] == "rejected"
    assert "could not be safely validated" in attempt["verification"]["reasons"][0]
    assert attempt["exception"]["type"] == "RuntimeError"
    assert "unsupported annotation" in attempt["exception"]["message"]
    assert report["final_diff"].strip() == ""
    assert (repo / "membership_mod.py").read_text(encoding="utf-8") == original


def test_agent_rejects_invalid_replacement(tmp_path, monkeypatch, membership_repo):
    from app.optimizer import OptimizationError

    repo, original = membership_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))

    def bad_replace(*args, **kwargs):
        raise OptimizationError("Replacement produced invalid Python")

    monkeypatch.setattr("app.agent.replace_function", bad_replace)
    agent = Agent(llm=FakeLLM(payload(WRONG_HAS_DUPES)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    assert report["status"] == "completed"
    assert report["rejected"] == 1
    attempt = report["iterations"][0]["attempts"][0]
    assert attempt["verification"]["status"] == "rejected"
    assert "replacement rejected" in attempt["verification"]["reasons"][0]
    assert attempt["exception"]["type"] == "OptimizationError"
    assert report["final_diff"].strip() == ""
    assert (repo / "membership_mod.py").read_text(encoding="utf-8") == original


def test_agent_apply_copies_verified_change(tmp_path, monkeypatch, sample_repo):
    repo, original = sample_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))
    agent = Agent(llm=FakeLLM(payload(OPTIMIZED)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1, apply=True)

    assert report["applied_to_original"] is True
    assert report["apply_result"]["files"] == ["slow_mod.py"]
    assert (repo / "slow_mod.py").read_text(encoding="utf-8") != original
    assert Path(report["apply_result"]["backup_dir"]).exists()
    assert Path(report["apply_result"]["patch_path"]).exists()


def test_agent_reports_missing_repository():
    agent = Agent(llm=DisabledLLM())
    report = agent.run("C:/definitely/not/a/real/repo", max_iterations=1)
    assert report["status"] == "aborted"
    assert "not a directory" in report["error"]
    assert report["failure"]["type"] == "ValueError"
    assert report["failure"]["stage"] == "OBSERVE"
    assert report["failure"]["traceback"]


def test_agent_records_decision_metadata_on_keep(tmp_path, monkeypatch, sample_repo):
    """The decision layer must be visible in the report, not just its status."""
    repo, _ = sample_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))
    agent = Agent(llm=FakeLLM(payload(OPTIMIZED)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    verification = report["iterations"][0]["verification"]
    assert verification["status"] == "kept"
    assert verification["action"] == "KEEP"
    assert verification["confidence"] == "HIGH"
    levels = {item["level"]: item for item in verification["evidence"]}
    # Repository tests, framework tests, differential outputs, benchmark and
    # static safety were all evaluated and satisfied.
    assert {1, 2, 4, 6, 7} <= set(levels)
    assert all(levels[level]["satisfied"] for level in (1, 2, 4, 6, 7))


def test_agent_records_decision_metadata_on_revert(tmp_path, monkeypatch, membership_repo):
    repo, _ = membership_repo
    monkeypatch.setattr("app.tools.run_pytest", fake_tests(True))
    agent = Agent(llm=FakeLLM(payload(SLOWER_HAS_DUPES)), benchmark_size=300, benchmark_repeats=2)

    report = agent.run(repo, max_iterations=1)

    verification = report["iterations"][0]["verification"]
    assert verification["status"] == "reverted"
    assert verification["action"] == "REVERT"
    assert verification["confidence"] == "HIGH"
    assert any("below the required" in reason for reason in verification["reasons"])
