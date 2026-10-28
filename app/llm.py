"""LLM access through the already-configured OpenCode CLI.

No API keys are stored or required by this project.  The adapter shells out to
``opencode run`` which uses the model and credentials the user already
configured in OpenCode.  If the CLI is missing or the call fails, the agent
falls back to conservative deterministic rewrites and reports that fact.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence


class LLMError(Exception):
    """Raised when the model cannot produce a usable response."""


Completed = subprocess.CompletedProcess


def _default_runner(
    cmd: Sequence[str],
    cwd: str,
    timeout: float,
    env: dict[str, str],
) -> Completed:
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _resolve_binary(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env_binary = os.environ.get("CODEOPTIMIZER_OPENCODE_BIN")
    if env_binary:
        return env_binary

    candidates: list[Path] = []
    for name in ("opencode.exe", "opencode"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(
            Path(appdata) / "npm" / "node_modules" / "@opencode" / "cli" / "bin" / "opencode.exe"
        )

    expanded: list[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.suffix.lower() in {".ps1", ".cmd", ".bat"}:
            expanded.append(candidate.with_suffix(".exe"))
            expanded.append(
                candidate.parent / "node_modules" / "@opencode" / "cli" / "bin" / "opencode.exe"
            )

    for candidate in expanded:
        if candidate.suffix.lower() == ".exe" and candidate.is_file():
            return str(candidate)
    for candidate in expanded:
        if candidate.is_file():
            return str(candidate)
    return None


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _extract_text(stdout: str) -> str:
    texts: list[str] = []
    for line in _strip_ansi(stdout).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "text":
            continue
        part = payload.get("part") or {}
        if part.get("type") == "text" and part.get("text"):
            texts.append(part["text"])
    if not texts:
        return _strip_ansi(stdout).strip()
    if "{" in texts[-1]:
        return texts[-1]
    return "\n".join(texts)


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from an LLM response."""
    cleaned = _strip_ansi(text).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError("response did not contain a valid JSON object")


class OpenCodeLLM:
    """Thin wrapper around ``opencode run`` using the user's configuration."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 300.0,
        binary: str | None = None,
        runner: Callable[[Sequence[str], str, float, dict[str, str]], Completed] | None = None,
    ) -> None:
        self.model = model or os.environ.get("CODEOPTIMIZER_MODEL") or None
        self.timeout = float(os.environ.get("CODEOPTIMIZER_LLM_TIMEOUT", timeout))
        self.binary = _resolve_binary(binary)
        self.runner = runner or _default_runner

    def available(self) -> bool:
        return bool(self.binary)

    def complete(self, prompt: str) -> str:
        if not self.binary:
            raise LLMError(
                "OpenCode CLI was not found. Install OpenCode or set CODEOPTIMIZER_OPENCODE_BIN."
            )

        cmd: list[str] = [self.binary, "run", "--format", "json", "--auto"]
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.append(prompt)

        lowered = self.binary.lower()
        if lowered.endswith(".ps1"):
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", *cmd]
        elif lowered.endswith((".cmd", ".bat")):
            cmd = ["cmd", "/c", *cmd]

        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        workdir = tempfile.mkdtemp(prefix="codeopt_llm_")
        started = time.perf_counter()
        try:
            completed = self.runner(cmd, workdir, self.timeout, env)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"OpenCode model call timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise LLMError(f"could not execute OpenCode CLI: {exc}") from exc
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        if completed.returncode != 0:
            detail = ((completed.stderr or "") + (completed.stdout or "")).strip()
            raise LLMError(f"OpenCode CLI exited with {completed.returncode}: {detail[-2000:]}")

        text = _extract_text(completed.stdout or "")
        if not text.strip():
            raise LLMError("OpenCode CLI returned an empty response")
        return text


def build_optimization_prompt(
    finding: dict,
    function_source: str,
    module_imports: list[str],
    previous_error: str | None = None,
) -> str:
    """Build a compact, strict prompt for one function-level optimization."""
    imports_text = "\n".join(module_imports[:20]) or "(none)"
    error_block = ""
    if previous_error:
        error_block = (
            "\nYour previous answer was rejected for this reason:\n"
            f"{previous_error}\nReturn a corrected answer.\n"
        )

    return f"""You are a senior Python performance engineer working inside an automated optimizer.

Optimize exactly ONE function. The optimizer will run the repository's tests and a
differential benchmark before keeping your change, so incorrect or slower code will be
rejected automatically.

Detected issue:
- kind: {finding.get('kind')}
- severity: {finding.get('severity')}
- file: {finding.get('file')}
- function: {finding.get('function')}
- line: {finding.get('line')}
- message: {finding.get('message')}
- suggestion: {finding.get('suggestion')}
- details: {json.dumps(finding.get('details', {}), sort_keys=True)}

Module-level imports available:
{imports_text}

Original function:
```python
{function_source}
```

Correctness warning:
- A plain `set(b)` membership test is NOT equivalent to a nested equality loop when
  duplicate matches matter. If an outer element matches the inner element k times,
  it must be emitted k times. Preserve multiplicity exactly.

Rules:
1. Preserve observable behaviour EXACTLY, including ordering, duplicates, edge cases,
   exceptions, and return types. The original is the specification.
2. Preserve the docstring EXACTLY. Do not remove, shorten, or rewrite it.
3. Keep the function name, signature, defaults, annotations, and async/sync nature.
4. Do NOT include decorators. They are preserved by the caller.
5. Only use the Python standard library. Allowed imports inside the function:
   collections, itertools, functools, math, bisect, heapq, operator, re, string,
   statistics, typing, dataclasses.
6. Do NOT add module-level code. Do NOT use eval, exec, open, subprocess, os, sys,
   network, filesystem, or dynamic imports.
7. Prefer a simple, obviously-correct optimization over a clever one. If no safe
   improvement exists, return the original function unchanged and explain why.
8. Return ONLY a JSON object. No markdown fences, no prose before or after.

JSON schema:
{{
  "finding_id": "{finding.get('id')}",
  "strategy": "short name of the approach",
  "rationale": "why this preserves behaviour and improves performance",
  "risk": "low | medium | high",
  "optimized_function": "complete replacement function source with \\n newlines"
}}
{error_block}"""
