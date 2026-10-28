"""Decision engine: the single place where evidence becomes a decision.

Validation and benchmarking produce *evidence*.  This module decides what to
do with it.  Keeping this separate from the agent loop makes the acceptance
rules auditable and unit-testable, and it enforces one invariant above all:

    A candidate is only ever KEEP when validation passed, the benchmark
    measured both versions on the same inputs, the outputs matched, and the
    measured improvement meets the configured threshold.

Decision states (they are not interchangeable):

    KEEP    validated and measurably better; the change stays.
    REVERT  the change was applied but the evidence does not support it.
    REJECT  the candidate itself is unsafe, invalid or unverifiable.
    SKIP    no candidate existed to evaluate (nothing was applied).
    ABORT   infrastructure failure prevented the workflow.  This is a
            job-level state set by the agent, never a candidate decision.

Generated behavioural validation is evidence, not proof.  Confidence is
reported honestly: HIGH for the repository's own test suite, MEDIUM for
generated behavioural cases, LOW for static-only reasoning, and UNVERIFIED
when nothing was executed.

Every rule here is fail-closed: missing evidence leads to REJECT, never to
KEEP.  The functions are pure so they can be tested without a workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KEEP = "KEEP"
REVERT = "REVERT"
REJECT = "REJECT"
SKIP = "SKIP"
ABORT = "ABORT"

# Legacy lowercase statuses used in reports and by the web UI.
STATUS_FOR_ACTION = {
    KEEP: "kept",
    REVERT: "reverted",
    REJECT: "rejected",
    SKIP: "skipped",
    ABORT: "aborted",
}

EXISTING_TESTS = "EXISTING_TESTS"
GENERATED_BEHAVIORAL_TESTS = "GENERATED_BEHAVIORAL_TESTS"
STATIC_ONLY = "STATIC_ONLY"

CONFIDENCE_BY_STRATEGY = {
    EXISTING_TESTS: "HIGH",
    GENERATED_BEHAVIORAL_TESTS: "MEDIUM",
    STATIC_ONLY: "LOW",
}


@dataclass
class Decision:
    """The outcome of evaluating one candidate against its evidence."""

    action: str
    reasons: list[str]
    confidence: str = "UNVERIFIED"
    evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Lowercase status kept for compatibility with existing reports."""
        return STATUS_FOR_ACTION.get(self.action, self.action.lower())

    @property
    def kept(self) -> bool:
        return self.action == KEEP

    def to_verification(self) -> dict[str, Any]:
        """Shape consumed by agent reports and the web UI."""
        return {
            "status": self.status,
            "action": self.action,
            "reasons": list(self.reasons),
            "confidence": self.confidence,
            "evidence": [dict(item) for item in self.evidence],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_verification()


def _decision(
    action: str,
    reasons: list[str],
    confidence: str = "UNVERIFIED",
    evidence: list[dict[str, Any]] | None = None,
) -> Decision:
    return Decision(
        action=action,
        reasons=list(reasons),
        confidence=confidence,
        evidence=list(evidence or []),
    )


# ---------------------------------------------------------------------------
# Validation hierarchy evidence (levels described in the design spec)
# ---------------------------------------------------------------------------


def build_evidence(
    safety: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    benchmark: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Record which validation-hierarchy levels were actually evaluated.

    Levels that were not evaluated are absent rather than claimed, so an
    evidence list never overstates what was proven.
    """
    levels: list[dict[str, Any]] = []
    strategy = (validation or {}).get("strategy")

    if validation is not None and strategy == EXISTING_TESTS:
        ran = bool(validation.get("ran"))
        passed = ran and bool(validation.get("passed"))
        runner = validation.get("runner") or "repository test suite"
        levels.append(
            {
                "level": 1,
                "name": "repository_tests",
                "satisfied": passed,
                "detail": (
                    f"{validation.get('cases_matched', 0)} tests passed via {runner}"
                    if passed
                    else str(validation.get("reason") or "repository tests did not pass")
                ),
            }
        )
        levels.append(
            {
                "level": 2,
                "name": "framework_tests",
                "satisfied": passed,
                "detail": f"runner: {runner}",
            }
        )

    if validation is not None and strategy == GENERATED_BEHAVIORAL_TESTS:
        total = int(validation.get("cases_total") or 0)
        matched = int(validation.get("cases_matched") or 0)
        passed = bool(validation.get("passed"))
        levels.append(
            {
                "level": 3,
                "name": "generated_behavioral",
                "satisfied": passed and total > 0 and matched == total,
                "detail": f"{matched}/{total} generated cases matched",
            }
        )
        levels.append(
            {
                "level": 5,
                "name": "edge_case_inputs",
                "satisfied": total > 0,
                "detail": f"{total} generated edge-case inputs exercised",
            }
        )

    if benchmark is not None:
        outputs_match = benchmark.get("outputs_match") is True
        levels.append(
            {
                "level": 4,
                "name": "differential_outputs",
                "satisfied": outputs_match,
                "detail": (
                    "original and optimized outputs are equal on benchmark inputs"
                    if outputs_match
                    else "outputs were not compared or differ"
                ),
            }
        )
        if benchmark.get("available"):
            levels.append(
                {
                    "level": 6,
                    "name": "benchmark_verification",
                    "satisfied": True,
                    "detail": (
                        f"measured improvement {benchmark.get('improvement_pct')}% "
                        f"over {benchmark.get('repeats', 0)} repeats (median)"
                    ),
                }
            )
        else:
            levels.append(
                {
                    "level": 6,
                    "name": "benchmark_verification",
                    "satisfied": False,
                    "detail": f"benchmark unavailable: {benchmark.get('reason')}",
                }
            )

    if safety is not None:
        levels.append(
            {
                "level": 7,
                "name": "static_safety",
                "satisfied": bool(safety.get("safe")),
                "detail": (
                    "static safety checks passed"
                    if safety.get("safe")
                    else "; ".join(safety.get("reasons") or []) or "unsafe"
                ),
            }
        )

    return levels


# ---------------------------------------------------------------------------
# Decision helpers for the agent's control-flow points
# ---------------------------------------------------------------------------


def skip(reason: str) -> Decision:
    """No candidate existed to evaluate."""
    return _decision(SKIP, [reason])


def revert_no_change() -> Decision:
    """The proposal produced no code change at all."""
    return _decision(REVERT, ["proposal produced no code change"])


def reject_replacement(error: str) -> Decision:
    """The replacement itself is invalid; nothing was executed."""
    return _decision(
        REJECT,
        [f"candidate could not be safely validated: replacement rejected: {error}"],
    )


def reject_unexpected(error: str, revert_error: str | None = None) -> Decision:
    """An unexpected failure while validating; the candidate is unverifiable."""
    reason = f"candidate could not be safely validated: {error}"
    if revert_error:
        reason += f"; automatic revert also failed: {revert_error}"
    return _decision(REJECT, [reason])


def abort(reason: str) -> Decision:
    """Job-level infrastructure failure (never produced for a candidate)."""
    return _decision(ABORT, [reason])


def check_safety(
    safety: dict[str, Any],
    validation_strategy: str,
) -> Decision | None:
    """Pre-modification gate: refuse candidates that must not be executed.

    Returns ``None`` when the candidate may proceed to modification.
    """
    evidence = build_evidence(safety=safety)

    if not safety.get("safe"):
        return _decision(
            REJECT,
            [
                "candidate is not safe to execute with generated inputs: "
                + "; ".join(safety.get("reasons") or [])
            ],
            confidence="UNVERIFIED",
            evidence=evidence,
        )

    if (
        validation_strategy == GENERATED_BEHAVIORAL_TESTS
        and not safety.get("generated_possible")
    ):
        return _decision(
            REJECT,
            [
                "candidate cannot be validated behaviourally: "
                + str(safety.get("input_error") or "no safe generated inputs")
            ],
            confidence="UNVERIFIED",
            evidence=evidence,
        )

    return None


def _tests_validation(tests: dict[str, Any]) -> dict[str, Any]:
    """Normalize a TestResult dict into a validation-evidence dict."""
    ran = bool(tests.get("ran"))
    passed = ran and bool(tests.get("passed"))
    return {
        "strategy": EXISTING_TESTS,
        "confidence": "HIGH",
        "ran": ran,
        "passed": passed,
        "cases_total": int(tests.get("passed_count") or 0)
        + int(tests.get("failed_count") or 0)
        + int(tests.get("error_count") or 0),
        "cases_matched": int(tests.get("passed_count") or 0) if passed else 0,
        "reason": tests.get("reason"),
    }


def check_generated_cases(
    *,
    cases: list[dict[str, Any]] | None,
    case_error: str | None = None,
    safety: dict[str, Any] | None = None,
) -> Decision | None:
    """Pre-execution gate: generated inputs must exist before running them.

    Returns ``None`` when behavioural validation can proceed.
    """
    if case_error or not cases:
        return _decision(
            REJECT,
            [
                "candidate could not be safely validated: "
                + str(case_error or "no generated cases were available")
            ],
            confidence="UNVERIFIED",
            evidence=build_evidence(safety=safety),
        )
    return None


def check_validation(
    *,
    validation_strategy: str,
    safety: dict[str, Any] | None = None,
    tests: dict[str, Any] | None = None,
    cases: list[dict[str, Any]] | None = None,
    case_error: str | None = None,
    behavior: dict[str, Any] | None = None,
) -> Decision | None:
    """Evaluate the validation evidence gathered after a change was applied.

    Returns ``None`` when the evidence supports proceeding to the benchmark,
    otherwise a REVERT/REJECT decision.  Missing evidence is REJECT: a
    candidate that cannot be validated is never allowed to continue.
    """
    if validation_strategy == EXISTING_TESTS:
        if tests is None:
            return _decision(
                REJECT,
                ["candidate could not be safely validated: repository tests were not run"],
                confidence="UNVERIFIED",
                evidence=build_evidence(safety=safety),
            )
        validation = _tests_validation(tests)
        evidence = build_evidence(safety=safety, validation=validation)
        if not validation["passed"]:
            return _decision(
                REVERT,
                [str(tests.get("reason") or "tests did not pass")],
                confidence="HIGH",
                evidence=evidence,
            )
        return None

    if validation_strategy == GENERATED_BEHAVIORAL_TESTS:
        cases_decision = check_generated_cases(
            cases=cases, case_error=case_error, safety=safety
        )
        if cases_decision is not None:
            return cases_decision
        if behavior is None:
            return _decision(
                REJECT,
                [
                    "candidate could not be safely validated: "
                    "behavioural validation was not run"
                ],
                confidence="UNVERIFIED",
                evidence=build_evidence(safety=safety),
            )
        confidence = str(behavior.get("confidence") or "MEDIUM")
        evidence = build_evidence(safety=safety, validation=behavior)
        if not behavior.get("ran"):
            return _decision(
                REJECT,
                [
                    "candidate could not be safely validated: "
                    f"behavioural validation did not run: {behavior.get('reason')}"
                ],
                confidence=confidence,
                evidence=evidence,
            )
        if not behavior.get("passed"):
            reasons = [
                str(
                    behavior.get("reason")
                    or "behaviour differs from the original implementation"
                )
            ]
            for mismatch in (behavior.get("mismatches") or [])[:3]:
                reasons.append(
                    f"case {mismatch.get('case')}: {mismatch.get('reason')} "
                    f"(input {mismatch.get('input')}; original "
                    f"{mismatch.get('original')}; optimized "
                    f"{mismatch.get('optimized')})"
                )
            return _decision(REVERT, reasons, confidence=confidence, evidence=evidence)
        return None

    # STATIC_ONLY (or an unknown strategy): nothing may proceed.
    return _decision(
        REJECT,
        [
            "candidate could not be safely validated: "
            f"validation strategy '{validation_strategy}' cannot execute evidence"
        ],
        confidence="UNVERIFIED",
        evidence=build_evidence(safety=safety),
    )


def _validation_phrase(validation: dict[str, Any] | None) -> str:
    if not validation:
        return "validation evidence recorded"
    if validation.get("strategy") == EXISTING_TESTS:
        return f"existing tests passed ({validation.get('cases_matched')} tests)"
    return (
        "generated behavioural validation matched "
        f"{validation.get('cases_matched')}/{validation.get('cases_total')} cases "
        f"({validation.get('confidence')} confidence)"
    )


def _validation_confidence(validation: dict[str, Any] | None) -> str:
    if not validation:
        return "UNVERIFIED"
    strategy = validation.get("strategy")
    if strategy in (EXISTING_TESTS, GENERATED_BEHAVIORAL_TESTS):
        return str(validation.get("confidence") or CONFIDENCE_BY_STRATEGY[strategy])
    return "UNVERIFIED"


def check_benchmark(
    benchmark: dict[str, Any],
    min_improvement_pct: float,
    *,
    validation: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
) -> Decision:
    """Final gate: KEEP only with validation + a measured, matching benchmark.

    This is the only function that can return KEEP, and it does so only when
    every required piece of evidence is present and sufficient.
    """
    evidence = build_evidence(safety=safety, validation=validation, benchmark=benchmark)
    confidence = _validation_confidence(validation)

    if validation is None:
        return _decision(
            REJECT,
            ["candidate could not be safely validated: validation evidence was not recorded"],
            confidence="UNVERIFIED",
            evidence=evidence,
        )

    if not benchmark.get("available"):
        return _decision(
            REJECT,
            [
                "candidate could not be safely validated: benchmark unavailable: "
                + str(benchmark.get("reason"))
            ],
            confidence=confidence,
            evidence=evidence,
        )

    reasons: list[str] = []
    if benchmark.get("outputs_match") is not True:
        reasons.append("benchmark outputs differ from the original implementation")
    improvement = benchmark.get("improvement_pct")
    if improvement is None:
        reasons.append("benchmark did not produce an improvement value")
    elif improvement < float(min_improvement_pct):
        reasons.append(
            f"improvement {improvement}% is below the required {min_improvement_pct}%"
        )

    if reasons:
        return _decision(REVERT, reasons, confidence=confidence, evidence=evidence)

    return _decision(
        KEEP,
        [
            _validation_phrase(validation),
            "outputs match on benchmark inputs",
            f"{improvement}% faster (threshold {min_improvement_pct}%)",
        ],
        confidence=confidence,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Composite decision over a complete evidence set
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Everything that can be known about one candidate."""

    safety: dict[str, Any] | None = None
    validation_strategy: str = STATIC_ONLY
    tests: dict[str, Any] | None = None
    cases: list[dict[str, Any]] | None = None
    case_error: str | None = None
    behavior: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None
    replacement_error: str | None = None
    unexpected_error: str | None = None


def _validation_from(evidence: Evidence) -> dict[str, Any] | None:
    if evidence.validation_strategy == EXISTING_TESTS and evidence.tests is not None:
        return _tests_validation(evidence.tests)
    if (
        evidence.validation_strategy == GENERATED_BEHAVIORAL_TESTS
        and evidence.behavior is not None
    ):
        return dict(evidence.behavior)
    return None


def decide_candidate(
    evidence: Evidence,
    min_improvement_pct: float = 5.0,
) -> Decision:
    """One-shot decision over a complete evidence set (fail-closed).

    The agent evaluates candidates stage by stage; this composite exists so
    the full rule set can be asserted in one place: no combination of
    evidence can produce KEEP unless validation passed *and* the benchmark
    is available, output-matching, and above the threshold.
    """
    if evidence.replacement_error:
        return reject_replacement(evidence.replacement_error)
    if evidence.unexpected_error:
        return reject_unexpected(evidence.unexpected_error)

    if evidence.safety is not None:
        decision = check_safety(evidence.safety, evidence.validation_strategy)
        if decision is not None:
            return decision

    decision = check_validation(
        validation_strategy=evidence.validation_strategy,
        safety=evidence.safety,
        tests=evidence.tests,
        cases=evidence.cases,
        case_error=evidence.case_error,
        behavior=evidence.behavior,
    )
    if decision is not None:
        return decision

    validation = _validation_from(evidence)
    if validation is None:
        return _decision(
            REJECT,
            ["candidate could not be safely validated: validation evidence was not recorded"],
            confidence="UNVERIFIED",
            evidence=build_evidence(safety=evidence.safety),
        )

    if evidence.benchmark is None:
        return _decision(
            REJECT,
            ["candidate could not be safely validated: benchmark evidence was not collected"],
            confidence=_validation_confidence(validation),
            evidence=build_evidence(safety=evidence.safety, validation=validation),
        )

    return check_benchmark(
        evidence.benchmark,
        min_improvement_pct,
        validation=validation,
        safety=evidence.safety,
    )


class DecisionEngine:
    """Configured facade used by the agent loop.

    The engine holds the acceptance threshold; every method delegates to the
    pure functions above so behaviour is identical wherever it is called.
    """

    def __init__(self, min_improvement_pct: float = 5.0) -> None:
        self.min_improvement_pct = float(min_improvement_pct)

    def check_safety(
        self, safety: dict[str, Any], validation_strategy: str
    ) -> Decision | None:
        return check_safety(safety, validation_strategy)

    def check_validation(self, **kwargs: Any) -> Decision | None:
        return check_validation(**kwargs)

    def check_generated_cases(
        self,
        *,
        cases: list[dict[str, Any]] | None,
        case_error: str | None = None,
        safety: dict[str, Any] | None = None,
    ) -> Decision | None:
        return check_generated_cases(cases=cases, case_error=case_error, safety=safety)

    def check_benchmark(
        self,
        benchmark: dict[str, Any],
        *,
        validation: dict[str, Any] | None = None,
        safety: dict[str, Any] | None = None,
    ) -> Decision:
        return check_benchmark(
            benchmark,
            self.min_improvement_pct,
            validation=validation,
            safety=safety,
        )

    def decide(self, evidence: Evidence) -> Decision:
        return decide_candidate(evidence, self.min_improvement_pct)
