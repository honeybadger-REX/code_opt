"""The agent loop: OBSERVE -> ANALYZE -> PLAN -> MODIFY -> TEST -> BENCHMARK -> VERIFY."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .analyzer import ACTIONABLE_KINDS, analyze_repository, is_test_file, rank_findings
from .benchmark import infer_params
from .git_manager import Workspace, apply_to_original, create_workspace
from .llm import LLMError, OpenCodeLLM, build_optimization_prompt, parse_json_object
from .models import Finding, Proposal
from .optimizer import (
    OptimizationError,
    deterministic_proposal,
    get_function_source,
    replace_function,
    validate_optimized_function,
)
from .tools import ToolRegistry, ToolError
from .validation import (
    assess_function_safety,
    choose_validation_strategy,
    discover_repository,
    evaluate_candidates,
    generate_cases,
)

MAX_LOG_ENTRIES = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _module_imports(source: str) -> list[str]:
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(stripped)
    return lines[:30]


def _excerpt(text: str | None, limit: int = 4000) -> str:
    """Keep the tail of long subprocess output (the useful part for failures)."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


class Agent:
    """An optimization agent that only keeps changes it can prove are safe."""

    def __init__(
        self,
        llm: Any | None = None,
        on_update: Callable[[str, str, dict[str, Any]], None] | None = None,
        min_improvement_pct: float = 5.0,
        benchmark_size: int = 3000,
        benchmark_repeats: int = 3,
        test_timeout: float = 300.0,
        max_llm_retries: int = 1,
        workspace_base: Path | None = None,
    ) -> None:
        self.llm = llm if llm is not None else OpenCodeLLM()
        self.on_update = on_update
        self.min_improvement_pct = float(min_improvement_pct)
        self.benchmark_size = int(benchmark_size)
        self.benchmark_repeats = int(benchmark_repeats)
        self.test_timeout = float(test_timeout)
        self.max_llm_retries = int(max_llm_retries)
        self.workspace_base = Path(workspace_base) if workspace_base else None
        self.logs: list[dict[str, Any]] = []

    # -- logging ---------------------------------------------------------
    def _emit(self, stage: str, message: str, **data: Any) -> None:
        entry = {"time": _now(), "stage": stage, "message": message}
        if data:
            entry["data"] = data
        self.logs.append(entry)
        if len(self.logs) > MAX_LOG_ENTRIES:
            del self.logs[: len(self.logs) - MAX_LOG_ENTRIES]
        if self.on_update is not None:
            try:
                self.on_update(stage, message, data)
            except Exception:
                pass

    # -- configuration ---------------------------------------------------
    @staticmethod
    def _load_repo_config(source: Path) -> dict[str, Any]:
        path = source / ".codeoptimizer.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    # -- planning --------------------------------------------------------
    def _plan(
        self,
        finding: Finding,
        file_source: str,
    ) -> tuple[Proposal | None, list[str]]:
        errors: list[str] = []
        function = finding.function
        if not function:
            return None, ["finding has no enclosing function"]
        try:
            original_function = get_function_source(file_source, function)
        except (OptimizationError, SyntaxError) as exc:
            return None, [f"cannot extract function: {exc}"]

        imports = _module_imports(file_source)

        if self.llm is not None and getattr(self.llm, "available", lambda: False)():
            for attempt in range(self.max_llm_retries + 1):
                prompt = build_optimization_prompt(
                    finding.to_dict(),
                    original_function,
                    imports,
                    previous_error=errors[-1] if errors else None,
                )
                try:
                    raw = self.llm.complete(prompt)
                    payload = parse_json_object(raw)
                    optimized_source = payload.get("optimized_function")
                    if not isinstance(optimized_source, str) or not optimized_source.strip():
                        raise LLMError("optimized_function is missing or empty")
                    problems = validate_optimized_function(file_source, optimized_source, function)
                    if problems:
                        raise LLMError("; ".join(problems))
                    proposal = Proposal(
                        finding_id=finding.id,
                        file=finding.file,
                        function=function,
                        strategy=str(payload.get("strategy") or "model-proposed optimization"),
                        rationale=str(payload.get("rationale") or ""),
                        risk=str(payload.get("risk") or "unknown"),
                        source=optimized_source,
                        planner="llm",
                        expected_gain=str(payload.get("expected_gain") or "unknown"),
                    )
                    return proposal, errors
                except Exception as exc:
                    errors.append(str(exc))
                    self._emit(
                        "PLAN",
                        f"Model proposal rejected (attempt {attempt + 1}): {exc}",
                        function=function,
                    )
                    # A timeout means the model is unavailable/slow; retrying would
                    # only double the wait. Validation errors are worth one retry.
                    if "timed out" in str(exc).lower():
                        break

        try:
            fallback = deterministic_proposal(finding, file_source, function)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"deterministic fallback failed: {exc}")
            fallback = None
        if fallback is not None:
            self._emit(
                "PLAN",
                f"Using deterministic rewrite for '{function}': {fallback.strategy}",
                function=function,
            )
        return fallback, errors

    # -- finding selection ----------------------------------------------
    def _select_finding(
        self,
        analysis: Any,
        attempted: set[str],
        locked_functions: set[str],
    ) -> tuple[Finding | None, str | None]:
        for finding in rank_findings(analysis.findings):
            if finding.kind not in ACTIONABLE_KINDS or not finding.auto_optimizable:
                continue
            if not finding.function or is_test_file(finding.file):
                continue
            key = f"{finding.file}::{finding.function}::{finding.kind}"
            function_key = f"{finding.file}::{finding.function}"
            if key in attempted or function_key in locked_functions:
                continue
            if Path(finding.file).is_absolute():
                continue
            return finding, key
        return None, None

    # -- main loop -------------------------------------------------------
    def run(
        self,
        repo_path: str | Path,
        max_iterations: int = 2,
        apply: bool = False,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        started_perf = time.perf_counter()
        max_iterations = max(0, min(int(max_iterations), 10))
        report: dict[str, Any] = {
            "job_id": job_id,
            "repo_path": str(Path(repo_path).expanduser().resolve()),
            "status": "running",
            "stage": "OBSERVE",
            "started_at": _now(),
            "finished_at": None,
            "duration_seconds": None,
            "workspace": None,
            "baseline_commit": None,
            "analysis": None,
            "final_analysis": None,
            "repository": None,
            "validation": None,
            "candidate_assessment": None,
            "candidates_found": 0,
            "candidates_attempted": 0,
            "baseline_tests": None,
            "baseline_diagnostic": None,
            "iterations": [],
            "kept": 0,
            "reverted": 0,
            "skipped": 0,
            "rejected": 0,
            "outcome_reason": None,
            "final_diff": "",
            "patch_path": None,
            "applied_to_original": False,
            "apply_result": None,
            "options": {
                "max_iterations": max_iterations,
                "min_improvement_pct": self.min_improvement_pct,
                "benchmark_size": self.benchmark_size,
                "benchmark_repeats": self.benchmark_repeats,
                "apply": apply,
            },
            "limitations": [
                "Only Python source files are analyzed.",
                "Automatic benchmarks require type-annotated parameters and functions that do not mutate their inputs.",
                "Test files (test_*.py, *_test.py, tests/) are never modified.",
                "The original repository is never edited unless apply=true; changes otherwise live in an isolated copy.",
            ],
            "error": None,
            "failure": None,
            "logs": self.logs,
        }
        workspace: Workspace | None = None
        tools: ToolRegistry | None = None

        try:
            source = Path(repo_path).expanduser().resolve()
            if not source.is_dir():
                raise ValueError(f"repository path is not a directory: {source}")

            config = self._load_repo_config(source)
            bench_config = config.get("benchmark", {}) if isinstance(config.get("benchmark"), dict) else {}
            if "size" in bench_config:
                self.benchmark_size = int(bench_config["size"])
            if "repeats" in bench_config:
                self.benchmark_repeats = int(bench_config["repeats"])
            if "min_improvement_pct" in bench_config:
                self.min_improvement_pct = float(bench_config["min_improvement_pct"])
            report["options"].update(
                {
                    "min_improvement_pct": self.min_improvement_pct,
                    "benchmark_size": self.benchmark_size,
                    "benchmark_repeats": self.benchmark_repeats,
                }
            )

            self._emit("OBSERVE", f"Repository: {source}")
            workspace = create_workspace(source, base_dir=self.workspace_base)
            report["workspace"] = str(workspace.root)
            report["baseline_commit"] = workspace.baseline_sha
            self._emit(
                "OBSERVE",
                f"Isolated workspace created at {workspace.root} (git={workspace.use_git})",
            )

            tools = ToolRegistry(workspace.root)
            tools.attach_workspace(workspace)

            self._set_stage(report, "ANALYZE")
            analysis = analyze_repository(workspace.root)
            report["analysis"] = analysis.to_dict()
            self._emit(
                "ANALYZE",
                f"Analyzed {analysis.files_analyzed} files, found {len(analysis.findings)} findings",
                findings_by_kind=analysis.metrics.get("findings_by_kind", {}),
            )

            self._set_stage(report, "DISCOVERY")
            capabilities = discover_repository(workspace.root)
            report["repository"] = capabilities.to_dict()
            candidate_assessment = evaluate_candidates(workspace.root, analysis)
            report["candidate_assessment"] = candidate_assessment
            report["candidates_found"] = candidate_assessment["candidates_found"]
            validation = choose_validation_strategy(capabilities, candidate_assessment)
            report["validation"] = validation
            self._emit(
                "DISCOVERY",
                f"Detected {capabilities.test_framework} test framework; "
                f"validation strategy {validation['strategy']} ({validation['confidence']})",
                test_framework=capabilities.test_framework,
                safe_candidates=candidate_assessment["safe_count"],
                candidates_found=candidate_assessment["candidates_found"],
            )

            self._set_stage(report, "VALIDATE")
            framework = capabilities.test_framework
            if framework == "pytest":
                baseline_tests = tools.run_tests(timeout=self.test_timeout)
            elif framework == "unittest":
                baseline_tests = tools.run_unittest(timeout=self.test_timeout)
            else:
                baseline_tests = {
                    "ran": False,
                    "passed": False,
                    "passed_count": 0,
                    "failed_count": 0,
                    "error_count": 0,
                    "skipped_count": 0,
                    "returncode": None,
                    "duration_seconds": 0.0,
                    "output": "",
                    "timed_out": False,
                    "reason": "no test framework detected",
                    "runner": "none",
                    "command": [],
                    "cwd": str(workspace.root),
                    "python_executable": None,
                    "sanitized_env": [],
                }
            report["baseline_tests"] = baseline_tests

            if baseline_tests["ran"] and not baseline_tests["passed"]:
                reason = (
                    baseline_tests.get("reason")
                    or f"{framework} exited with code {baseline_tests.get('returncode')}"
                )
                report["status"] = "completed"
                report["outcome_reason"] = "baseline_tests_failed"
                report["error"] = (
                    "Baseline tests fail before any change; refusing to optimize. "
                    f"{reason}."
                )
                report["failure"] = {
                    "stage": "VALIDATE",
                    "type": "BaselineTestFailure",
                    "message": report["error"],
                    "reason": reason,
                    "returncode": baseline_tests.get("returncode"),
                    "command": baseline_tests.get("command"),
                    "cwd": baseline_tests.get("cwd"),
                    "output": _excerpt(baseline_tests.get("output")),
                    "candidate": None,
                    "traceback": None,
                }
                self._emit("VALIDATE", report["error"], reason=reason)
                return self._finalize(report, workspace, tools, started_perf, apply)

            if baseline_tests["ran"]:
                self._emit(
                    "VALIDATE",
                    f"Baseline tests passed ({baseline_tests['passed_count']} passed)",
                )
            else:
                reason = (
                    baseline_tests.get("reason")
                    or "the baseline suite did not collect any tests"
                )
                report["baseline_diagnostic"] = {
                    "runner": baseline_tests.get("runner"),
                    "reason": reason,
                    "returncode": baseline_tests.get("returncode"),
                    "command": baseline_tests.get("command"),
                    "cwd": baseline_tests.get("cwd"),
                    "output": _excerpt(baseline_tests.get("output")),
                }
                if validation["strategy"] == "EXISTING_TESTS":
                    if candidate_assessment["safe_count"] > 0:
                        validation.update(
                            {
                                "strategy": "GENERATED_BEHAVIORAL_TESTS",
                                "confidence": "MEDIUM",
                                "runner": "generated",
                                "reason": (
                                    "The existing test suite did not run; falling back to "
                                    f"generated behavioural validation ({reason})."
                                ),
                            }
                        )
                    else:
                        validation.update(
                            {
                                "strategy": "STATIC_ONLY",
                                "confidence": "LOW",
                                "runner": "none",
                                "reason": (
                                    "The existing test suite did not run and no safe "
                                    f"behavioural validation is available ({reason})."
                                ),
                            }
                        )
                    report["validation"] = validation
                self._emit(
                    "VALIDATE",
                    "Baseline suite did not run; continuing with "
                    f"{validation['strategy']} ({validation['confidence']})",
                    reason=reason,
                )

            # Static-only validation cannot prove correctness, so refuse to
            # modify anything.  This is a controlled completion, not a crash.
            if validation["strategy"] == "STATIC_ONLY":
                rejected = candidate_assessment["candidates_found"]
                report["rejected"] = rejected
                report["outcome_reason"] = "no_safe_optimization_candidate"
                report["error"] = (
                    "No tests and no safe behavioural validation available. "
                    f"{rejected} candidate(s) were rejected without execution."
                )
                self._emit("VALIDATE", report["error"])
                return self._finalize(report, workspace, tools, started_perf, apply)

            if (
                validation["strategy"] == "EXISTING_TESTS"
                and candidate_assessment["execution_safe_count"] == 0
                and candidate_assessment["candidates_found"] > 0
            ):
                rejected = candidate_assessment["candidates_found"]
                report["rejected"] = rejected
                report["outcome_reason"] = "no_safe_optimization_candidate"
                report["error"] = (
                    "No candidate is safe to execute with generated benchmark inputs. "
                    f"{rejected} candidate(s) were rejected without execution."
                )
                self._emit("VALIDATE", report["error"])
                return self._finalize(report, workspace, tools, started_perf, apply)

            attempted: set[str] = set()
            locked_functions: set[str] = set()
            llm_used = 0
            deterministic_used = 0

            for index in range(max_iterations):
                self._set_stage(report, "ANALYZE")
                analysis = analyze_repository(workspace.root)
                finding, key = self._select_finding(analysis, attempted, locked_functions)
                if finding is None:
                    self._emit("ANALYZE", "No more actionable findings")
                    break
                attempted.add(key)
                report["candidates_attempted"] += 1
                function = finding.function or ""
                record: dict[str, Any] = {
                    "index": index + 1,
                    "finding": finding.to_dict(),
                    "proposal": None,
                    "planner_errors": [],
                    "safety": None,
                    "validation": None,
                    "diff": "",
                    "tests": None,
                    "benchmark": None,
                    "verification": {"status": "skipped", "reasons": []},
                    "duration_seconds": None,
                }
                iteration_started = time.perf_counter()
                self._emit(
                    "PLAN",
                    f"Considering {finding.kind} in {finding.file}:{finding.line} ({function})",
                    finding_id=finding.id,
                )

                path = workspace.root / finding.file
                try:
                    file_source = tools.read_file(finding.file)["content"]
                except ToolError as exc:
                    record["verification"]["reasons"].append(str(exc))
                    report["iterations"].append(record)
                    report["skipped"] += 1
                    continue

                safety = assess_function_safety(file_source, function)
                record["safety"] = safety
                if not safety["safe"]:
                    reason = (
                        "candidate is not safe to execute with generated inputs: "
                        + "; ".join(safety["reasons"])
                    )
                    record["verification"] = {"status": "rejected", "reasons": [reason]}
                    report["iterations"].append(record)
                    report["rejected"] += 1
                    self._emit("VALIDATE", f"REJECTED '{function}': {reason}")
                    continue
                if (
                    validation["strategy"] == "GENERATED_BEHAVIORAL_TESTS"
                    and not safety["generated_possible"]
                ):
                    reason = (
                        "candidate cannot be validated behaviourally: "
                        f"{safety.get('input_error') or 'no safe generated inputs'}"
                    )
                    record["verification"] = {"status": "rejected", "reasons": [reason]}
                    report["iterations"].append(record)
                    report["rejected"] += 1
                    self._emit("VALIDATE", f"REJECTED '{function}': {reason}")
                    continue

                proposal, plan_errors = self._plan(finding, file_source)
                record["planner_errors"] = plan_errors

                candidates: list[Proposal] = []
                if proposal is not None:
                    candidates.append(proposal)
                try:
                    fallback = deterministic_proposal(finding, file_source, function)
                except Exception as exc:
                    fallback = None
                    plan_errors.append(f"deterministic fallback failed: {exc}")
                if fallback is not None and all(
                    fallback.source.strip() != candidate.source.strip() for candidate in candidates
                ):
                    candidates.append(fallback)

                if not candidates:
                    record["verification"] = {
                        "status": "skipped",
                        "reasons": ["no valid proposal from the model or deterministic fallback"],
                    }
                    report["iterations"].append(record)
                    report["skipped"] += 1
                    self._emit("PLAN", f"Skipping '{function}': no safe proposal")
                    continue

                record["attempts"] = []
                kept = False
                final_attempt: dict[str, Any] | None = None

                for candidate in candidates:
                    attempt: dict[str, Any] = {
                        "proposal": candidate.to_dict(),
                        "diff": "",
                        "validation": None,
                        "tests": None,
                        "benchmark": None,
                        "verification": {"status": "skipped", "reasons": []},
                    }
                    if candidate.planner == "llm":
                        llm_used += 1
                    else:
                        deterministic_used += 1
                    self._emit(
                        "PLAN",
                        f"Trying proposal ({candidate.planner}): {candidate.strategy}",
                        strategy=candidate.strategy,
                    )

                    try:
                        self._set_stage(report, "MODIFY")
                        try:
                            replace_function(path, function, candidate.source)
                        except OptimizationError as exc:
                            attempt["verification"] = {
                                "status": "rejected",
                                "reasons": [
                                    "candidate could not be safely validated: "
                                    f"replacement rejected: {exc}"
                                ],
                            }
                            attempt["exception"] = {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc(limit=8),
                            }
                            record["attempts"].append(attempt)
                            final_attempt = attempt
                            self._emit("MODIFY", f"REJECTED change: {exc}")
                            continue

                        diff = tools.git_diff(file=finding.file)["diff"]
                        attempt["diff"] = diff
                        if not diff.strip():
                            tools.git_revert(file=finding.file)
                            attempt["verification"] = {
                                "status": "reverted",
                                "reasons": ["proposal produced no code change"],
                            }
                            record["attempts"].append(attempt)
                            final_attempt = attempt
                            continue
                        self._emit(
                            "MODIFY", f"Applied change to {finding.file} ({len(diff)} diff chars)"
                        )

                        self._set_stage(report, "VALIDATE")
                        if validation["strategy"] == "EXISTING_TESTS":
                            if validation.get("runner") == "unittest":
                                tests = tools.run_unittest(timeout=self.test_timeout)
                            else:
                                tests = tools.run_tests(timeout=self.test_timeout)
                            attempt["tests"] = tests
                            attempt["validation"] = {
                                "strategy": "EXISTING_TESTS",
                                "confidence": "HIGH",
                                "ran": tests.get("ran", False),
                                "passed": tests.get("passed", False),
                                "cases_total": tests.get("passed_count", 0)
                                + tests.get("failed_count", 0)
                                + tests.get("error_count", 0),
                                "cases_matched": tests.get("passed_count", 0),
                                "reason": tests.get("reason"),
                            }
                            self._emit(
                                "VALIDATE",
                                f"Existing tests: {tests['passed_count']} passed, "
                                f"{tests['failed_count']} failed, {tests['error_count']} errors",
                            )
                            if not tests["ran"] or not tests["passed"]:
                                tools.git_revert(file=finding.file)
                                attempt["verification"] = {
                                    "status": "reverted",
                                    "reasons": [tests.get("reason") or "tests did not pass"],
                                }
                                record["attempts"].append(attempt)
                                final_attempt = attempt
                                self._emit("VERIFY", f"Reverted '{function}': tests failed")
                                continue
                        else:
                            cases, case_error = generate_cases(safety.get("params") or [])
                            if case_error or not cases:
                                tools.git_revert(file=finding.file)
                                reason = (
                                    "candidate could not be safely validated: "
                                    f"{case_error or 'no generated cases were available'}"
                                )
                                attempt["verification"] = {
                                    "status": "rejected",
                                    "reasons": [reason],
                                }
                                record["attempts"].append(attempt)
                                final_attempt = attempt
                                self._emit("VERIFY", f"REJECTED '{function}': {reason}")
                                continue
                            behavior = tools.validate_behavior(
                                function=function,
                                baseline_source=file_source,
                                optimized_path=finding.file,
                                cases=cases,
                                timeout=min(self.test_timeout, 120.0),
                            )
                            attempt["validation"] = behavior
                            self._emit(
                                "VALIDATE",
                                f"Generated behavioural validation: "
                                f"{behavior.get('cases_matched')}/{behavior.get('cases_total')} "
                                f"cases matched",
                            )
                            if not behavior.get("ran"):
                                tools.git_revert(file=finding.file)
                                reason = (
                                    "candidate could not be safely validated: "
                                    f"behavioural validation did not run: "
                                    f"{behavior.get('reason')}"
                                )
                                attempt["verification"] = {
                                    "status": "rejected",
                                    "reasons": [reason],
                                }
                                record["attempts"].append(attempt)
                                final_attempt = attempt
                                self._emit("VERIFY", f"REJECTED '{function}': {reason}")
                                continue
                            if not behavior.get("passed"):
                                tools.git_revert(file=finding.file)
                                reasons = [
                                    behavior.get("reason")
                                    or "behaviour differs from the original implementation"
                                ]
                                for mismatch in (behavior.get("mismatches") or [])[:3]:
                                    reasons.append(
                                        f"case {mismatch.get('case')}: {mismatch.get('reason')} "
                                        f"(input {mismatch.get('input')}; original "
                                        f"{mismatch.get('original')}; optimized "
                                        f"{mismatch.get('optimized')})"
                                    )
                                attempt["verification"] = {
                                    "status": "reverted",
                                    "reasons": reasons,
                                }
                                record["attempts"].append(attempt)
                                final_attempt = attempt
                                self._emit(
                                    "VERIFY",
                                    f"Reverted '{function}': behavioural mismatch",
                                )
                                continue

                        self._set_stage(report, "BENCHMARK")
                        try:
                            optimized_file_source = tools.read_file(finding.file)["content"]
                            params, param_error = infer_params(optimized_file_source, function)
                        except ToolError as exc:
                            params, param_error = [], str(exc)

                        if param_error:
                            bench: dict[str, Any] = {
                                "available": False,
                                "function": function,
                                "reason": param_error,
                            }
                            self._emit("BENCHMARK", f"Benchmark unavailable: {param_error}")
                        else:
                            bench = tools.benchmark(
                                function=function,
                                baseline_source=file_source,
                                optimized_path=finding.file,
                                params=params,
                                size=self.benchmark_size,
                                repeats=self.benchmark_repeats,
                            )
                            if bench["available"]:
                                self._emit(
                                    "BENCHMARK",
                                    f"Before {bench['before_seconds']}s -> after {bench['after_seconds']}s "
                                    f"({bench['improvement_pct']}% improvement)",
                                    outputs_match=bench.get("outputs_match"),
                                )
                            else:
                                self._emit(
                                    "BENCHMARK", f"Benchmark unavailable: {bench.get('reason')}"
                                )
                        attempt["benchmark"] = bench

                        self._set_stage(report, "VERIFY")
                        if not bench.get("available"):
                            # The candidate cannot be measured, so it cannot be
                            # validated.  This is a controlled rejection, not a
                            # job crash and not a silent success.
                            tools.git_revert(file=finding.file)
                            reason = (
                                "candidate could not be safely validated: benchmark "
                                f"unavailable: {bench.get('reason')}"
                            )
                            attempt["verification"] = {
                                "status": "rejected",
                                "reasons": [reason],
                            }
                            record["attempts"].append(attempt)
                            final_attempt = attempt
                            self._emit("VERIFY", f"REJECTED '{function}': {reason}")
                            continue

                        reasons: list[str] = []
                        if bench.get("outputs_match") is not True:
                            reasons.append("benchmark outputs differ from the original implementation")
                        improvement = bench.get("improvement_pct")
                        if improvement is None:
                            reasons.append("benchmark did not produce an improvement value")
                        elif improvement < self.min_improvement_pct:
                            reasons.append(
                                f"improvement {improvement}% is below the required "
                                f"{self.min_improvement_pct}%"
                            )

                        if reasons:
                            tools.git_revert(file=finding.file)
                            attempt["verification"] = {"status": "reverted", "reasons": reasons}
                            record["attempts"].append(attempt)
                            final_attempt = attempt
                            self._emit("VERIFY", f"Reverted '{function}': {'; '.join(reasons)}")
                            continue

                        commit = workspace.commit(
                            f"codeoptimizer: optimize {function} ({finding.kind})"
                        )
                        validation_result = attempt.get("validation") or {}
                        if validation_result.get("strategy") == "EXISTING_TESTS":
                            validation_reason = (
                                f"existing tests passed "
                                f"({validation_result.get('cases_matched')} tests)"
                            )
                        else:
                            validation_reason = (
                                "generated behavioural validation matched "
                                f"{validation_result.get('cases_matched')}/"
                                f"{validation_result.get('cases_total')} cases "
                                f"({validation_result.get('confidence')} confidence)"
                            )
                        attempt["verification"] = {
                            "status": "kept",
                            "reasons": [
                                validation_reason,
                                "outputs match on benchmark inputs",
                                f"{bench['improvement_pct']}% faster (threshold {self.min_improvement_pct}%)",
                            ],
                            "commit": commit,
                        }
                        record["attempts"].append(attempt)
                        final_attempt = attempt
                        kept = True
                        locked_functions.add(f"{finding.file}::{function}")
                        self._emit(
                            "VERIFY",
                            f"Kept '{function}': {bench['improvement_pct']}% faster and validated",
                        )
                        break
                    except Exception as exc:
                        # One bad candidate must not crash the whole job.  Revert
                        # the workspace and report the exact failure in the attempt.
                        revert_error = self._revert_candidate(tools, workspace, finding.file)
                        reason = (
                            "candidate could not be safely validated: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        if revert_error:
                            reason += f"; automatic revert also failed: {revert_error}"
                        attempt["verification"] = {
                            "status": "rejected",
                            "reasons": [reason],
                        }
                        attempt["exception"] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                        record["attempts"].append(attempt)
                        final_attempt = attempt
                        self._emit(
                            "VERIFY",
                            f"REJECTED '{function}': {type(exc).__name__}: {exc}",
                        )
                        continue

                if final_attempt is not None:
                    record["proposal"] = final_attempt["proposal"]
                    record["diff"] = final_attempt["diff"]
                    record["validation"] = final_attempt.get("validation")
                    record["tests"] = final_attempt["tests"]
                    record["benchmark"] = final_attempt["benchmark"]
                    record["verification"] = final_attempt["verification"]

                if kept:
                    report["kept"] += 1
                elif record["verification"]["status"] == "reverted":
                    report["reverted"] += 1
                elif record["verification"]["status"] == "rejected":
                    report["rejected"] += 1
                else:
                    report["skipped"] += 1

                record["duration_seconds"] = round(time.perf_counter() - iteration_started, 3)
                report["iterations"].append(record)

            report["planner_stats"] = {"llm": llm_used, "deterministic": deterministic_used}
            if llm_used == 0 and deterministic_used > 0:
                report["limitations"].append(
                    "The OpenCode model did not produce a valid proposal; deterministic rewrites were used."
                )

            self._set_stage(report, "DONE")
            return self._finalize(report, workspace, tools, started_perf, apply)

        except Exception as exc:
            # Only genuine infrastructure/system failures abort the job.
            report["status"] = "aborted"
            report["error"] = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc(limit=8)
            report["traceback"] = tb
            report["failure"] = {
                "stage": report.get("stage"),
                "type": type(exc).__name__,
                "message": str(exc),
                "reason": str(exc),
                "traceback": tb,
                "candidate": None,
            }
            self._emit("ERROR", report["error"])
            if workspace is not None:
                try:
                    workspace.revert_all()
                except Exception as revert_exc:
                    report["limitations"].append(
                        f"could not revert the isolated workspace after the failure: {revert_exc}"
                    )
            return self._finalize(report, workspace, tools, started_perf, apply)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _set_stage(report: dict[str, Any], stage: str) -> None:
        report["stage"] = stage

    @staticmethod
    def _revert_candidate(
        tools: ToolRegistry,
        workspace: Workspace | None,
        file: str,
    ) -> str | None:
        """Best-effort revert of one candidate's file changes.

        Returns ``None`` when the workspace was restored, otherwise a message
        describing why even the fallback revert failed.
        """
        try:
            tools.git_revert(file=file)
            return None
        except Exception as exc:
            if workspace is not None:
                try:
                    workspace.revert_all()
                    return None
                except Exception as fallback_exc:
                    return (
                        f"{type(exc).__name__}: {exc}; workspace revert failed: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}"
                    )
            return f"{type(exc).__name__}: {exc}"

    def _finalize(
        self,
        report: dict[str, Any],
        workspace: Workspace | None,
        tools: ToolRegistry | None,
        started_perf: float,
        apply: bool,
    ) -> dict[str, Any]:
        report["finished_at"] = _now()
        report["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
        report["logs"] = self.logs
        if report["status"] == "running":
            report["status"] = "completed"
        if report["status"] == "completed":
            if report.get("outcome_reason"):
                report["outcome"] = report["outcome_reason"]
            elif report["kept"] > 0:
                report["outcome"] = "optimized"
            elif report.get("candidates_found", 0) == 0:
                report["outcome"] = "no_actionable_findings"
            elif report["rejected"] > 0 and report["reverted"] == 0:
                report["outcome"] = "rejected"
            else:
                report["outcome"] = "no_beneficial_change"
        else:
            report["outcome"] = report["status"]
        report["candidates_kept"] = report["kept"]
        report["candidates_reverted"] = report["reverted"]
        report["candidates_rejected"] = report["rejected"]
        report["candidates_skipped"] = report["skipped"]

        if workspace is not None:
            try:
                report["final_diff"] = workspace.final_diff()
                patch_path = workspace.root.parent / f"{workspace.root.name}.patch"
                workspace.save_patch(patch_path)
                report["patch_path"] = str(patch_path)
            except Exception as exc:
                report["limitations"].append(f"could not save final diff: {exc}")

            if apply and report["kept"] > 0 and report["status"] == "completed":
                try:
                    files = workspace.changed_files()
                    result = apply_to_original(workspace, files)
                    report["applied_to_original"] = True
                    report["apply_result"] = result
                    self._emit("APPLY", f"Applied {len(result['files'])} files to the original repository")
                except Exception as exc:
                    report["applied_to_original"] = False
                    report["apply_result"] = {"error": str(exc)}
                    self._emit("APPLY", f"Failed to apply changes: {exc}")

            if report["status"] == "completed":
                try:
                    final_analysis = analyze_repository(workspace.root)
                    report["final_analysis"] = final_analysis.to_dict()
                except Exception:
                    pass

        if workspace is not None:
            report["workspace"] = str(workspace.root)
        return report
