"""Real tools available to the agent.

Every tool performs the action it describes.  The registry is bound to an
isolated workspace root, and file paths are constrained to that root.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .analyzer import EXCLUDED_DIRS, analyze_file
from .benchmark import compare_functions
from .git_manager import Workspace
from .tester import run_tests as run_pytest, run_unittest as run_unittest_suite
from .validation import run_behavioral_validation


class ToolError(Exception):
    """Raised for invalid tool invocations."""


class ToolRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.workspace: Workspace | None = None
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "search_repository": self.search_repository,
            "analyze_python": self.analyze_python,
            "run_tests": self.run_tests,
            "run_unittest": self.run_unittest,
            "validate_behavior": self.validate_behavior,
            "benchmark": self.benchmark,
            "git_diff": self.git_diff,
            "git_revert": self.git_revert,
        }

    def attach_workspace(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def _resolve(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        target = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"path escapes the workspace: {relative_path}")
        return target

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return path.name

    # -- files -----------------------------------------------------------
    def read_file(self, path: str, max_bytes: int = 200_000) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"file not found: {path}")
        raw = target.read_bytes()
        truncated = len(raw) > max_bytes
        content = raw[:max_bytes].decode("utf-8", errors="replace")
        return {
            "path": self._relative(target),
            "content": content,
            "bytes": len(raw),
            "truncated": truncated,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": self._relative(target), "bytes": len(content.encode("utf-8")), "written": True}

    def search_repository(
        self,
        pattern: str,
        include: str = "*.py",
        max_results: int = 100,
    ) -> dict[str, Any]:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc

        matches: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob(include)):
            if not path.is_file():
                continue
            if any(part in EXCLUDED_DIRS for part in path.relative_to(self.root).parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "file": self._relative(path),
                            "line": number,
                            "text": line.strip()[:300],
                        }
                    )
                    if len(matches) >= max_results:
                        return {"pattern": pattern, "matches": matches, "truncated": True}
        return {"pattern": pattern, "matches": matches, "truncated": False}

    # -- analysis --------------------------------------------------------
    def analyze_python(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"file not found: {path}")
        return analyze_file(target, self.root).to_dict()

    # -- verification ----------------------------------------------------
    def run_tests(self, timeout: float = 300.0) -> dict[str, Any]:
        return run_pytest(self.root, timeout=timeout).to_dict()

    def run_unittest(self, timeout: float = 300.0) -> dict[str, Any]:
        return run_unittest_suite(self.root, timeout=timeout).to_dict()

    def validate_behavior(
        self,
        function: str,
        baseline_source: str,
        optimized_path: str,
        cases: list[dict[str, Any]],
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        target = self._resolve(optimized_path)
        result = run_behavioral_validation(
            repo_dir=self.root,
            baseline_source=baseline_source,
            optimized_path=target,
            function_name=function,
            cases=cases,
            timeout=timeout,
        )
        return result.to_dict()

    def benchmark(
        self,
        function: str,
        baseline_source: str,
        optimized_path: str,
        params: list[dict[str, str]],
        size: int = 1000,
        repeats: int = 3,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        target = self._resolve(optimized_path)
        result = compare_functions(
            repo_dir=self.root,
            baseline_source=baseline_source,
            optimized_path=target,
            function_name=function,
            params=params,
            size=size,
            repeats=repeats,
            timeout=timeout,
        )
        return result.to_dict()

    # -- git -------------------------------------------------------------
    def git_diff(self, file: str | None = None, cumulative: bool = False) -> dict[str, Any]:
        if self.workspace is None:
            raise ToolError("no workspace attached")
        if cumulative:
            diff = self.workspace.final_diff()
        elif file:
            diff = self.workspace.diff_file(file)
        else:
            diff = self.workspace.final_diff()
        return {"diff": diff, "empty": not diff.strip()}

    def git_revert(self, file: str | None = None) -> dict[str, Any]:
        if self.workspace is None:
            raise ToolError("no workspace attached")
        if file:
            self.workspace.revert_file(file)
            return {"reverted": file}
        self.workspace.revert_all()
        return {"reverted": "all"}

    # -- generic dispatch ------------------------------------------------
    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name}")
        return tool(**kwargs)

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        return [
            {"name": "read_file", "description": "Read a UTF-8 file inside the workspace."},
            {"name": "write_file", "description": "Write a file inside the workspace."},
            {"name": "search_repository", "description": "Regex search across repository files."},
            {"name": "analyze_python", "description": "AST analysis of one Python file."},
            {"name": "run_tests", "description": "Run pytest in the workspace."},
            {"name": "run_unittest", "description": "Run unittest discovery in the workspace."},
            {"name": "validate_behavior", "description": "Compare original vs optimized behaviour on generated inputs."},
            {"name": "benchmark", "description": "Differential benchmark of original vs optimized function."},
            {"name": "git_diff", "description": "Show the workspace diff."},
            {"name": "git_revert", "description": "Revert workspace changes."},
        ]
