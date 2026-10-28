"""Decision engine tests.

These tests pin down the acceptance rules as a unit: every decision state,
every fail-closed path, and the invariant that KEEP is impossible without
validation *and* a measured, output-matching benchmark above the threshold.
"""

from __future__ import annotations

from app.decision import (
    ABORT,
    KEEP,
    REJECT,
    REVERT,
    SKIP,
    STATUS_FOR_ACTION,
    Evidence,
    build_evidence,
    check_benchmark,
    check_safety,
    check_validation,
    decide_candidate,
    reject_replacement,
    reject_unexpected,
    revert_no_change,
    skip,
)


def safe_safety(**overrides):
    data = {
        "safe": True,
        "reasons": [],
        "module_reasons": [],
        "params": [{"name": "items", "annotation": "list[int]", "kwonly": False}],
        "input_error": None,
        "generated_possible": True,
    }
    data.update(overrides)
    return data


def suite_result(passed=True, ran=True):
    return {
        "ran": ran,
        "passed": passed and ran,
        "passed_count": 10 if passed and ran else 4,
        "failed_count": 0 if passed else 1,
        "error_count": 0,
        "reason": None if passed and ran else "4 passed, 1 failed",
    }


def existing_validation(passed=True, ran=True):
    return {
        "strategy": "EXISTING_TESTS",
        "confidence": "HIGH",
        "ran": ran,
        "passed": passed and ran,
        "cases_total": 10,
        "cases_matched": 10 if passed and ran else 4,
        "reason": None if passed and ran else "4 passed, 1 failed",
    }


def generated_behavior(passed=True, ran=True, total=50):
    return {
        "strategy": "GENERATED_BEHAVIORAL_TESTS",
        "confidence": "MEDIUM",
        "ran": ran,
        "passed": passed and ran,
        "cases_total": total,
        "cases_matched": total if passed and ran else total - 1,
        "mismatches": [] if passed else [{"case": 7, "reason": "return value differs"}],
        "reason": None if passed and ran else "behaviour differs from the original implementation",
    }


def benchmark(improvement=42.0, outputs_match=True, available=True, reason=None):
    return {
        "available": available,
        "function": "f",
        "before_seconds": 0.1,
        "after_seconds": 0.058,
        "improvement_pct": improvement,
        "repeats": 3,
        "outputs_match": outputs_match,
        "reason": reason,
    }


def cases():
    return [{"args": [[]], "kwargs": {}}]


# ---------------------------------------------------------------------------
# KEEP is the only action that requires full evidence
# ---------------------------------------------------------------------------


def test_keep_requires_validation_and_benchmark():
    decision = check_benchmark(
        benchmark(improvement=42.0),
        5.0,
        validation=existing_validation(),
        safety=safe_safety(),
    )
    assert decision.action == KEEP
    assert decision.status == "kept"
    assert decision.confidence == "HIGH"
    assert any("existing tests passed" in reason for reason in decision.reasons)
    assert any("42.0% faster (threshold 5.0%)" in reason for reason in decision.reasons)


def test_keep_never_without_benchmark():
    decision = decide_candidate(
        Evidence(
            safety=safe_safety(),
            validation_strategy="EXISTING_TESTS",
            tests=suite_result(),
        )
    )
    assert decision.action == REJECT
    assert "benchmark evidence was not collected" in decision.reasons[0]


def test_keep_never_without_validation_evidence():
    decision = decide_candidate(
        Evidence(
            safety=safe_safety(),
            validation_strategy="GENERATED_BEHAVIORAL_TESTS",
            cases=cases(),
            benchmark=benchmark(),
        )
    )
    assert decision.action == REJECT
    assert "behavioural validation was not run" in decision.reasons[0]


def test_benchmark_gate_refuses_missing_validation_record():
    decision = check_benchmark(benchmark(), 5.0, validation=None, safety=safe_safety())
    assert decision.action == REJECT
    assert "validation evidence was not recorded" in decision.reasons[0]


def test_generated_keep_reports_medium_confidence():
    decision = check_benchmark(
        benchmark(improvement=12.0),
        5.0,
        validation=generated_behavior(),
        safety=safe_safety(),
    )
    assert decision.action == KEEP
    assert decision.confidence == "MEDIUM"
    assert any("50/50 cases" in reason for reason in decision.reasons)


# ---------------------------------------------------------------------------
# REVERT: the change was applied but the evidence does not support it
# ---------------------------------------------------------------------------


def test_revert_when_tests_fail():
    decision = check_validation(
        validation_strategy="EXISTING_TESTS",
        tests=suite_result(passed=False),
        safety=safe_safety(),
    )
    assert decision is not None
    assert decision.action == REVERT
    assert decision.status == "reverted"
    assert decision.confidence == "HIGH"
    assert decision.reasons == ["4 passed, 1 failed"]


def test_revert_when_behaviour_differs():
    decision = check_validation(
        validation_strategy="GENERATED_BEHAVIORAL_TESTS",
        cases=cases(),
        behavior=generated_behavior(passed=False),
        safety=safe_safety(),
    )
    assert decision is not None
    assert decision.action == REVERT
    assert decision.confidence == "MEDIUM"
    assert any("behaviour differs" in reason for reason in decision.reasons)
    assert any("case 7" in reason for reason in decision.reasons)


def test_revert_below_threshold():
    decision = check_benchmark(
        benchmark(improvement=1.5),
        5.0,
        validation=existing_validation(),
        safety=safe_safety(),
    )
    assert decision.action == REVERT
    assert any("below the required" in reason for reason in decision.reasons)


def test_revert_when_outputs_differ():
    decision = check_benchmark(
        benchmark(improvement=90.0, outputs_match=False),
        5.0,
        validation=existing_validation(),
        safety=safe_safety(),
    )
    assert decision.action == REVERT
    assert any("outputs differ" in reason for reason in decision.reasons)


def test_revert_when_improvement_missing():
    decision = check_benchmark(
        benchmark(improvement=None),
        5.0,
        validation=existing_validation(),
        safety=safe_safety(),
    )
    assert decision.action == REVERT
    assert any("did not produce an improvement value" in reason for reason in decision.reasons)


def test_no_change_revert():
    decision = revert_no_change()
    assert decision.action == REVERT
    assert decision.status == "reverted"
    assert decision.reasons == ["proposal produced no code change"]


# ---------------------------------------------------------------------------
# REJECT: the candidate itself is unsafe, invalid, or unverifiable
# ---------------------------------------------------------------------------


def test_reject_unsafe_candidate():
    decision = check_safety(
        safe_safety(safe=False, reasons=["calls unsafe function 'open' at line 3"]),
        "GENERATED_BEHAVIORAL_TESTS",
    )
    assert decision is not None
    assert decision.action == REJECT
    assert decision.status == "rejected"
    assert decision.confidence == "UNVERIFIED"
    assert "not safe to execute" in decision.reasons[0]
    assert "open" in decision.reasons[0]


def test_reject_when_generated_validation_impossible():
    decision = check_safety(
        safe_safety(generated_possible=False, input_error="methods are not supported"),
        "GENERATED_BEHAVIORAL_TESTS",
    )
    assert decision is not None
    assert decision.action == REJECT
    assert "cannot be validated behaviourally" in decision.reasons[0]
    assert "methods are not supported" in decision.reasons[0]


def test_safety_gate_passes_for_safe_candidate():
    assert check_safety(safe_safety(), "GENERATED_BEHAVIORAL_TESTS") is None
    # With existing tests, generated inputs are not required.
    assert check_safety(safe_safety(generated_possible=False), "EXISTING_TESTS") is None


def test_reject_replacement_error():
    decision = reject_replacement("Replacement produced invalid Python")
    assert decision.action == REJECT
    assert "replacement rejected" in decision.reasons[0]


def test_reject_unexpected_error_includes_revert_failure():
    decision = reject_unexpected("RuntimeError: boom", "checkout failed")
    assert decision.action == REJECT
    assert "could not be safely validated" in decision.reasons[0]
    assert "RuntimeError: boom" in decision.reasons[0]
    assert "automatic revert also failed" in decision.reasons[0]


def test_reject_benchmark_unavailable():
    decision = check_benchmark(
        benchmark(available=False, improvement=None, outputs_match=None,
                  reason="parameter 'items' has no type annotation"),
        5.0,
        validation=existing_validation(),
        safety=safe_safety(),
    )
    assert decision.action == REJECT
    assert any("benchmark unavailable" in reason for reason in decision.reasons)


def test_reject_behavioural_validation_did_not_run():
    decision = check_validation(
        validation_strategy="GENERATED_BEHAVIORAL_TESTS",
        cases=cases(),
        behavior=generated_behavior(ran=False, passed=False),
        safety=safe_safety(),
    )
    assert decision is not None
    assert decision.action == REJECT
    assert any("did not run" in reason for reason in decision.reasons)


def test_reject_when_no_generated_cases():
    decision = check_validation(
        validation_strategy="GENERATED_BEHAVIORAL_TESTS",
        cases=[],
        case_error="unsupported annotation for generated inputs: set[str]",
        safety=safe_safety(),
    )
    assert decision is not None
    assert decision.action == REJECT
    assert "unsupported annotation" in decision.reasons[0]


def test_reject_when_repository_tests_were_not_run():
    decision = check_validation(
        validation_strategy="EXISTING_TESTS",
        tests=None,
        safety=safe_safety(),
    )
    assert decision is not None
    assert decision.action == REJECT
    assert "repository tests were not run" in decision.reasons[0]


def test_reject_static_only_strategy():
    decision = check_validation(validation_strategy="STATIC_ONLY", safety=safe_safety())
    assert decision is not None
    assert decision.action == REJECT
    assert "cannot execute evidence" in decision.reasons[0]


def test_skip_when_no_proposal():
    decision = skip("no valid proposal from the model or deterministic fallback")
    assert decision.action == SKIP
    assert decision.status == "skipped"
    assert decision.confidence == "UNVERIFIED"


# ---------------------------------------------------------------------------
# Composite ordering and evidence hierarchy
# ---------------------------------------------------------------------------


def test_replacement_error_wins_over_everything():
    decision = decide_candidate(
        Evidence(
            safety=safe_safety(safe=False),
            validation_strategy="EXISTING_TESTS",
            tests=suite_result(),
            benchmark=benchmark(),
            replacement_error="signature changed",
        )
    )
    assert decision.action == REJECT
    assert "replacement rejected" in decision.reasons[0]


def test_safety_rejects_before_validation():
    decision = decide_candidate(
        Evidence(
            safety=safe_safety(safe=False, reasons=["mutates global state at line 2"]),
            validation_strategy="EXISTING_TESTS",
            tests=suite_result(),
            benchmark=benchmark(),
        )
    )
    assert decision.action == REJECT
    assert "not safe to execute" in decision.reasons[0]


def test_full_evidence_decides_keep():
    decision = decide_candidate(
        Evidence(
            safety=safe_safety(),
            validation_strategy="EXISTING_TESTS",
            tests=suite_result(),
            benchmark=benchmark(improvement=80.0),
        )
    )
    assert decision.action == KEEP
    assert decision.confidence == "HIGH"


def test_evidence_levels_recorded_for_existing_tests():
    decision = check_benchmark(
        benchmark(),
        5.0,
        validation=existing_validation(),
        safety=safe_safety(),
    )
    levels = {item["level"]: item for item in decision.evidence}
    assert set(levels) == {1, 2, 4, 6, 7}
    assert all(item["satisfied"] for item in levels.values())
    # Generated levels were not evaluated and must not be claimed.
    assert 3 not in levels
    assert 5 not in levels


def test_evidence_levels_recorded_for_generated_validation():
    decision = check_benchmark(
        benchmark(),
        5.0,
        validation=generated_behavior(),
        safety=safe_safety(),
    )
    levels = {item["level"]: item for item in decision.evidence}
    assert set(levels) == {3, 4, 5, 6, 7}
    assert all(item["satisfied"] for item in levels.values())


def test_failed_validation_is_recorded_as_unsatisfied_evidence():
    levels = build_evidence(
        safety=safe_safety(),
        validation=existing_validation(passed=False),
        benchmark=benchmark(available=False, improvement=None, outputs_match=None),
    )
    by_level = {item["level"]: item for item in levels}
    assert by_level[1]["satisfied"] is False
    assert by_level[4]["satisfied"] is False
    assert by_level[6]["satisfied"] is False
    assert by_level[7]["satisfied"] is True


def test_every_action_has_a_distinct_legacy_status():
    statuses = {STATUS_FOR_ACTION[action] for action in (KEEP, REVERT, REJECT, SKIP, ABORT)}
    assert statuses == {"kept", "reverted", "rejected", "skipped", "aborted"}


def test_verification_shape_keeps_status_and_reasons():
    verification = revert_no_change().to_verification()
    assert verification["status"] == "reverted"
    assert verification["action"] == "REVERT"
    assert verification["reasons"]
    assert verification["confidence"] == "UNVERIFIED"
    assert isinstance(verification["evidence"], list)
