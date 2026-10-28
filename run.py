#!/usr/bin/env python
"""Command line entry point for CodeOptimizer AI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent import Agent  # noqa: E402
from app.analyzer import analyze_repository  # noqa: E402
from app.models import Analysis  # noqa: E402


class DisabledLLM:
    """Explicit 'no model' adapter used by --no-llm and offline tests."""

    def available(self) -> bool:
        return False

    def complete(self, prompt: str) -> str:  # pragma: no cover - never called
        raise RuntimeError("LLM disabled")


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return f"{value * 1e6:.0f} us"
    if value < 1:
        return f"{value * 1000:.1f} ms"
    return f"{value:.3f} s"


def print_analysis(analysis: Analysis) -> None:
    print(f"Repository: {analysis.repo_path}")
    print(
        f"Files: {analysis.files_analyzed}  Lines: {analysis.total_lines}  "
        f"Functions: {len(analysis.functions)}  Classes: {len(analysis.classes)}"
    )
    print(f"Findings: {len(analysis.findings)}")
    for finding in analysis.findings:
        location = f"{finding.file}:{finding.line}"
        function = finding.function or "<module>"
        print(f"  [{finding.severity:6}] {finding.kind:26} {location} ({function})")
        print(f"           {finding.message}")
        print(f"           -> {finding.suggestion}")


def _print_failure(failure: dict) -> None:
    if not failure:
        return
    print("")
    print("Failure:")
    print(f"  Stage      : {failure.get('stage')}")
    print(f"  Type       : {failure.get('type')}")
    print(f"  Message    : {failure.get('message')}")
    reason = failure.get("reason")
    if reason and reason != failure.get("message"):
        print(f"  Reason     : {reason}")
    if failure.get("returncode") is not None:
        print(f"  Exit code  : {failure.get('returncode')}")
    if failure.get("command"):
        print(f"  Command    : {' '.join(str(part) for part in failure['command'])}")
    if failure.get("cwd"):
        print(f"  CWD        : {failure.get('cwd')}")
    output = failure.get("output")
    if output:
        print("  Output (tail):")
        for line in output.splitlines()[-20:]:
            print(f"    {line}")
    tb = failure.get("traceback")
    if tb:
        print("  Traceback:")
        for line in tb.strip().splitlines()[-12:]:
            print(f"    {line}")


def print_report(report: dict, show_diff: bool = True) -> None:
    print("")
    print("=" * 74)
    print("CodeOptimizer AI report")
    print("=" * 74)
    print(f"Repository : {report.get('repo_path')}")
    print(f"Status     : {report.get('status')} ({report.get('outcome')})")
    print(f"Workspace  : {report.get('workspace')}")
    print(f"Duration   : {report.get('duration_seconds')} s")

    repository = report.get("repository") or {}
    if repository:
        print(f"Language   : {repository.get('language', 'Python')}")
        print(
            f"Python     : {repository.get('python_files', 0)} files, "
            f"{repository.get('lines', 0)} lines "
            f"({repository.get('source_files', 0)} source, "
            f"{repository.get('test_files', 0)} test)"
        )
        print(f"Test suite : {str(repository.get('test_framework', 'none')).upper()}")
        if repository.get("config_files"):
            print(f"Config     : {', '.join(repository['config_files'])}")
        if repository.get("notes"):
            for note in repository["notes"]:
                print(f"             {note}")

    validation = report.get("validation") or {}
    if validation:
        print(f"Validation : {validation.get('strategy')}")
        print(f"Confidence : {validation.get('confidence')}")
        if validation.get("reason"):
            print(f"             {validation['reason']}")

    print(
        f"Candidates : found {report.get('candidates_found', 0)}   "
        f"attempted {report.get('candidates_attempted', 0)}   "
        f"kept {report.get('kept')}   reverted {report.get('reverted')}   "
        f"rejected {report.get('rejected', 0)}   skipped {report.get('skipped')}"
    )
    if report.get("error"):
        print(f"Error      : {report['error']}")
    if report.get("outcome_reason"):
        print(f"Outcome    : {report['outcome_reason']}")
    baseline = report.get("baseline_tests") or {}
    if baseline:
        if baseline.get("passed"):
            baseline_state = f"passed ({baseline.get('passed_count')} passed)"
        elif not baseline.get("ran"):
            baseline_state = f"not run - {baseline.get('reason')}"
        else:
            baseline_state = f"failed - {baseline.get('reason')}"
        print(f"Baseline   : {baseline_state}")
        if baseline.get("sanitized_env"):
            print(
                "             ignored inherited env vars: "
                + ", ".join(baseline["sanitized_env"])
            )
    print(f"Patch      : {report.get('patch_path')}")
    _print_failure(report.get("failure") or {})

    for iteration in report.get("iterations", []):
        finding = iteration.get("finding") or {}
        proposal = iteration.get("proposal") or {}
        validation = iteration.get("validation") or {}
        tests = iteration.get("tests") or {}
        benchmark = iteration.get("benchmark") or {}
        verification = iteration.get("verification") or {}
        print("")
        print("-" * 74)
        print(
            f"Iteration {iteration.get('index')}: {finding.get('kind')} "
            f"in {finding.get('file')}:{finding.get('line')} ({finding.get('function')})"
        )
        print(f"  Finding     : {finding.get('message')}")
        if proposal:
            print(f"  Strategy    : {proposal.get('strategy')} [{proposal.get('planner')}]")
            if proposal.get("rationale"):
                print(f"  Rationale   : {proposal['rationale']}")
        else:
            print("  Strategy    : no valid proposal")
        if validation:
            print(
                f"  Validation  : {validation.get('strategy')} "
                f"({validation.get('confidence')} confidence)"
            )
            if validation.get("strategy") == "GENERATED_BEHAVIORAL_TESTS":
                print(
                    f"  Cases       : {validation.get('cases_matched')}/"
                    f"{validation.get('cases_total')} matched"
                )
            elif validation.get("reason"):
                print(f"  Validation  : {validation.get('reason')}")
        if tests:
            print(
                f"  Tests       : {'PASSED' if tests.get('passed') else 'FAILED'} "
                f"({tests.get('passed_count')} passed, {tests.get('failed_count')} failed, "
                f"{tests.get('error_count')} errors)"
            )
        if benchmark.get("available"):
            print("")
            print("  Before:")
            print(f"  Runtime: {_fmt_seconds(benchmark.get('before_seconds'))}")
            print("")
            print("  After:")
            print(f"  Runtime: {_fmt_seconds(benchmark.get('after_seconds'))}")
            print("")
            print(f"  Improvement: {benchmark.get('improvement_pct')}%")
            print(f"  Outputs match: {benchmark.get('outputs_match')}")
            print(f"  Inputs: {benchmark.get('input_description')} ({benchmark.get('repeats')} repeats, median)")
        elif benchmark:
            print(f"  Benchmark   : UNAVAILABLE - {benchmark.get('reason')}")
        print(f"  Verdict     : {str(verification.get('status', 'skipped')).upper()}")
        for reason in verification.get("reasons", []):
            print(f"                - {reason}")
        attempts = iteration.get("attempts") or []
        if len(attempts) > 1:
            print("  Attempts    :")
            for attempt in attempts:
                candidate = attempt.get("proposal") or {}
                result = attempt.get("verification") or {}
                print(
                    f"                - {candidate.get('planner', '?')}: "
                    f"{candidate.get('strategy', '?')} -> {result.get('status', '?')}"
                )
                for reason in result.get("reasons", []):
                    print(f"                  {reason}")
                exception = attempt.get("exception") or {}
                if exception:
                    print(
                        f"                  {exception.get('type')}: {exception.get('message')}"
                    )

    if show_diff:
        print("")
        print("-" * 74)
        print("Final diff (kept changes only):")
        print(report.get("final_diff") or "(no changes kept)")

    if report.get("limitations"):
        print("")
        print("Limitations:")
        for item in report["limitations"]:
            print(f"  - {item}")


def build_agent(args: argparse.Namespace) -> Agent:
    llm = DisabledLLM() if getattr(args, "no_llm", False) else None
    return Agent(
        llm=llm,
        min_improvement_pct=args.min_improvement,
        benchmark_size=args.benchmark_size,
        benchmark_repeats=args.benchmark_repeats,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    analysis = analyze_repository(Path(args.repo_path))
    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print_analysis(analysis)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    agent = build_agent(args)
    report = agent.run(
        args.repo_path,
        max_iterations=args.max_iterations,
        apply=args.apply,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, show_diff=not args.no_diff)
    return 0 if report.get("status") == "completed" else 1


def cmd_demo(args: argparse.Namespace) -> int:
    args.repo_path = str(ROOT / "examples")
    args.max_iterations = args.max_iterations
    args.apply = False
    args.json = False
    args.no_diff = False
    print("CodeOptimizer AI demo")
    print(f"Target: {args.repo_path}")
    return cmd_optimize(args)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeOptimizer AI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start the FastAPI server and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    analyze = sub.add_parser("analyze", help="Analyze a repository (read-only)")
    analyze.add_argument("repo_path", nargs="?", default=str(ROOT / "examples"))
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    for name, func, help_text in (
        ("optimize", cmd_optimize, "Run the optimization agent"),
        ("demo", cmd_demo, "Run the end-to-end demo on examples/"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("repo_path", nargs="?", default=str(ROOT / "examples"))
        cmd.add_argument("--max-iterations", type=int, default=2)
        cmd.add_argument("--min-improvement", type=float, default=5.0)
        cmd.add_argument("--benchmark-size", type=int, default=3000)
        cmd.add_argument("--benchmark-repeats", type=int, default=3)
        cmd.add_argument("--apply", action="store_true", help="Copy verified changes back to the original repo")
        cmd.add_argument("--no-llm", action="store_true", help="Use deterministic rewrites only")
        cmd.add_argument("--json", action="store_true")
        cmd.add_argument("--no-diff", action="store_true")
        cmd.set_defaults(func=func)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
