"""Run the repository's own test suite in an isolated workspace."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .models import TestResult

# Environment variables that change what pytest collects or how it loads
# plugins.  They belong to the *caller's* session, not to the repository being
# optimized, so they must never leak into the verification gate.
PYTEST_ENV_VARS = (
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
    "PYTEST_DEBUG",
    "PYTEST_CURRENT_TEST",
)


def _parse_counts(output: str) -> tuple[int, int, int, int]:
    passed = failed = errors = skipped = 0
    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) failed", output)
    if match:
        failed = int(match.group(1))
    match = re.search(r"(\d+) errors?", output)
    if match:
        errors = int(match.group(1))
    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))
    return passed, failed, errors, skipped


def _sanitize_env() -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    removed = [name for name in PYTEST_ENV_VARS if name in env]
    for name in removed:
        env.pop(name, None)
    return env, removed


def _result(
    *,
    cmd: list[str],
    repo_dir: Path,
    sanitized_env: list[str],
    duration: float,
    output: str,
    ran: bool,
    passed: bool,
    passed_count: int = 0,
    failed_count: int = 0,
    error_count: int = 0,
    skipped_count: int = 0,
    returncode: int | None = None,
    timed_out: bool = False,
    reason: str | None = None,
    runner: str = "pytest",
) -> TestResult:
    return TestResult(
        ran=ran,
        passed=passed,
        passed_count=passed_count,
        failed_count=failed_count,
        error_count=error_count,
        skipped_count=skipped_count,
        returncode=returncode,
        duration_seconds=round(duration, 3),
        output=output[-20000:],
        timed_out=timed_out,
        reason=reason,
        runner=runner,
        command=list(cmd),
        cwd=str(repo_dir),
        python_executable=sys.executable,
        sanitized_env=sanitized_env,
    )


def _pytest_available() -> bool:
    try:
        return importlib.util.find_spec("pytest") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def run_tests(
    repo_dir: Path,
    timeout: float = 300.0,
    extra_args: list[str] | None = None,
) -> TestResult:
    """Run ``pytest`` in ``repo_dir`` and return structured results.

    ``ran`` is False when no tests were collected.  The caller must treat that
    as "cannot verify" rather than "passed".  ``reason`` always explains *why*
    a run did not pass so the agent and the UI can show an actionable message
    instead of a generic exit code.
    """
    repo_dir = Path(repo_dir).resolve()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--color=no",
        "-p",
        "no:cacheprovider",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env, sanitized_env = _sanitize_env()

    if not repo_dir.is_dir():
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=0.0,
            output="",
            ran=False,
            passed=False,
            reason=f"not a directory: {repo_dir}",
        )

    if not _pytest_available():
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=0.0,
            output="",
            ran=False,
            passed=False,
            reason=(
                "pytest is not installed in the Python environment used to run the "
                "agent (python -m pytest is unavailable). Install pytest in that "
                "environment to enable verification."
            ),
        )

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_dir) + (os.pathsep + existing if existing else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        output = ""
        for stream in (exc.stdout, exc.stderr):
            if isinstance(stream, bytes):
                output += stream.decode("utf-8", "replace")
            elif stream:
                output += stream
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=duration,
            output=output,
            ran=True,
            passed=False,
            error_count=1,
            timed_out=True,
            reason=f"test suite timed out after {timeout}s",
        )
    except FileNotFoundError:
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=time.perf_counter() - started,
            output="",
            ran=False,
            passed=False,
            reason=(
                "could not start pytest: the Python interpreter "
                f"'{sys.executable}' was not found"
            ),
        )
    except OSError as exc:
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=time.perf_counter() - started,
            output="",
            ran=False,
            passed=False,
            reason=f"could not start pytest: {type(exc).__name__}: {exc}",
        )

    duration = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    passed, failed, errors, skipped = _parse_counts(output)
    lowered = output.lower()

    missing_pytest = "no module named pytest" in lowered or "no module named 'pytest'" in lowered
    if missing_pytest:
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=duration,
            output=output,
            ran=False,
            passed=False,
            returncode=completed.returncode,
            reason=(
                "pytest is not installed in the Python environment used to run the "
                "agent (python -m pytest failed with 'No module named pytest'). "
                "Install pytest in that environment to enable verification."
            ),
        )

    no_tests = (
        completed.returncode == 5
        or "no tests ran" in lowered
        or "collected 0 items" in lowered
    )
    ran = not no_tests
    ok = completed.returncode == 0 and ran and failed == 0 and errors == 0

    reason = None
    if no_tests:
        deselected = re.search(r"(\d+) deselected", output)
        if deselected or "all tests deselected" in lowered:
            count = f"{deselected.group(1)} " if deselected else "all "
            reason = (
                f"no tests ran because {count}tests were deselected (pytest exit code "
                f"{completed.returncode}); check test selection options such as -k/-m"
            )
        else:
            reason = (
                "no tests were collected; correctness cannot be verified "
                f"(pytest exit code {completed.returncode})"
            )
    elif not ok:
        if completed.returncode == 2:
            reason = (
                "pytest collection or usage error (exit code 2); see the test output "
                "for the collection failure"
            )
        elif completed.returncode == 3:
            reason = "pytest internal error (exit code 3); see the test output"
        elif completed.returncode == 4:
            reason = "pytest usage error (exit code 4); check the test command/options"
        elif failed or errors:
            reason = (
                f"pytest exited with code {completed.returncode}: "
                f"{failed} failed, {errors} errors"
            )
        else:
            reason = f"pytest exited with code {completed.returncode}"

    return _result(
        cmd=cmd,
        repo_dir=repo_dir,
        sanitized_env=sanitized_env,
        duration=duration,
        output=output,
        ran=ran,
        passed=ok,
        passed_count=passed,
        failed_count=failed,
        error_count=errors,
        skipped_count=skipped,
        returncode=completed.returncode,
        reason=reason,
    )


def run_unittest(repo_dir: Path, timeout: float = 300.0) -> TestResult:
    """Run ``unittest discover`` in ``repo_dir`` and return structured results.

    This is used when a repository has unittest tests but no pytest tests.  It
    deliberately does not force pytest onto a unittest-only repository.
    """
    repo_dir = Path(repo_dir).resolve()
    cmd = [sys.executable, "-m", "unittest", "discover", "-v"]
    env, sanitized_env = _sanitize_env()

    if not repo_dir.is_dir():
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=0.0,
            output="",
            ran=False,
            passed=False,
            reason=f"not a directory: {repo_dir}",
            runner="unittest",
        )

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_dir) + (os.pathsep + existing if existing else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        output = ""
        for stream in (exc.stdout, exc.stderr):
            if isinstance(stream, bytes):
                output += stream.decode("utf-8", "replace")
            elif stream:
                output += stream
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=duration,
            output=output,
            ran=True,
            passed=False,
            error_count=1,
            timed_out=True,
            reason=f"unittest suite timed out after {timeout}s",
            runner="unittest",
        )
    except (OSError, FileNotFoundError) as exc:
        return _result(
            cmd=cmd,
            repo_dir=repo_dir,
            sanitized_env=sanitized_env,
            duration=time.perf_counter() - started,
            output="",
            ran=False,
            passed=False,
            reason=f"could not start unittest: {type(exc).__name__}: {exc}",
            runner="unittest",
        )

    duration = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    match = re.search(r"Ran (\d+) tests?", output)
    total = int(match.group(1)) if match else 0

    failed = errors = skipped = 0
    failed_match = re.search(
        r"FAILED\s*\((?:failures=(\d+))?(?:,?\s*errors=(\d+))?(?:,?\s*skipped=(\d+))?\)",
        output,
    )
    if failed_match:
        failed = int(failed_match.group(1) or 0)
        errors = int(failed_match.group(2) or 0)
        skipped = int(failed_match.group(3) or 0)
    if failed == 0 and errors == 0 and re.search(r"FAILED", output):
        # Older/newer unittest formats may omit the counts.  Do not report a
        # failure as a pass just because the summary shape changed.
        errors = 1

    ran = total > 0
    passed = ran and completed.returncode == 0 and "FAILED" not in output
    passed_count = max(0, total - failed - errors - skipped)

    reason = None
    if not ran:
        if completed.returncode != 0:
            ran = True
            errors = max(errors, 1)
            passed = False
            reason = (
                "unittest discovery failed (exit code "
                f"{completed.returncode}); see the test output"
            )
        else:
            reason = "no unittest tests were collected"
    elif not passed:
        reason = (
            f"unittest exited with code {completed.returncode}: "
            f"{failed} failed, {errors} errors"
        )

    return _result(
        cmd=cmd,
        repo_dir=repo_dir,
        sanitized_env=sanitized_env,
        duration=duration,
        output=output,
        ran=ran,
        passed=passed,
        passed_count=passed_count,
        failed_count=failed,
        error_count=errors,
        skipped_count=skipped,
        returncode=completed.returncode,
        reason=reason,
        runner="unittest",
    )
