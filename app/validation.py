"""Validation engine for repositories with or without a test suite.

The optimizer must never treat "no pytest tests" as an infrastructure failure.
This module discovers what a repository actually provides, decides how strongly
a candidate can be validated, refuses to execute unsafe functions, and can run a
differential behavioural check between an original and an optimized function.

Generated behavioural tests are evidence, not a proof of correctness.  The
confidence values returned here are deliberately conservative.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .analyzer import ACTIONABLE_KINDS, EXCLUDED_DIRS, is_test_file, rank_findings
from .models import Analysis, ValidationResult

BEHAVIOR_MARKER = "BEHAVIOR_JSON:"

# Modules whose use can touch the outside world, consume nondeterministic
# state, or mutate global state.  Functions that use them are not executed with
# generated inputs.
UNSAFE_MODULES = {
    "builtins",
    "ctypes",
    "datetime",
    "importlib",
    "logging",
    "marshal",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "psycopg2",
    "pymysql",
    "random",
    "requests",
    "secrets",
    "shutil",
    "socket",
    "sqlite3",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "urllib",
    "uuid",
    "warnings",
    "http",
    "httpx",
}

UNSAFE_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "input",
    "locals",
    "open",
    "quit",
    "vars",
}

# Method names that are unambiguously external side effects.  Generic names such
# as ``replace`` or ``remove`` are intentionally excluded because they are also
# common pure list/string operations.
UNSAFE_METHODS = {
    "connect",
    "commit",
    "execute",
    "executemany",
    "flush",
    "mkdir",
    "makedirs",
    "popen",
    "recv",
    "rollback",
    "send",
    "sendall",
    "unlink",
    "urlopen",
    "write",
    "writelines",
}

MAX_GENERATED_CASES = 200


@dataclass
class RepositoryCapabilities:
    """What the repository provides for validation and benchmarking."""

    repo_path: str
    name: str
    language: str = "Python"
    python_files: int = 0
    source_files: int = 0
    test_files: int = 0
    lines: int = 0
    test_framework: str = "none"
    test_file_paths: list[str] = field(default_factory=list)
    test_directories: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    unittest_detected: bool = False
    pytest_detected: bool = False
    custom_tests_detected: bool = False
    automated_execution_supported: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "name": self.name,
            "language": self.language,
            "python_files": self.python_files,
            "source_files": self.source_files,
            "test_files": self.test_files,
            "lines": self.lines,
            "test_framework": self.test_framework,
            "test_file_paths": self.test_file_paths,
            "test_directories": self.test_directories,
            "config_files": self.config_files,
            "dependencies": self.dependencies,
            "entry_points": self.entry_points,
            "unittest_detected": self.unittest_detected,
            "pytest_detected": self.pytest_detected,
            "custom_tests_detected": self.custom_tests_detected,
            "automated_execution_supported": self.automated_execution_supported,
            "notes": self.notes,
        }


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        files.append(path)
    return files


def _read_text(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _pytest_config_files(root: Path) -> list[str]:
    found: list[str] = []
    if (root / "pytest.ini").is_file():
        found.append("pytest.ini")
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and "[tool.pytest" in _read_text(pyproject):
        found.append("pyproject.toml")
    setup_cfg = root / "setup.cfg"
    if setup_cfg.is_file() and "[tool:pytest]" in _read_text(setup_cfg):
        found.append("setup.cfg")
    tox_ini = root / "tox.ini"
    if tox_ini.is_file() and "[pytest]" in _read_text(tox_ini):
        found.append("tox.ini")
    return found


def _test_usage(path: Path) -> tuple[bool, bool, bool]:
    """Return ``(uses_unittest, uses_pytest, has_plain_test_functions)``."""
    source = _read_text(path)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, False, False

    uses_unittest = False
    uses_pytest = False
    plain_tests = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "unittest":
                    uses_unittest = True
                if root == "pytest":
                    uses_pytest = True
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root == "unittest":
                uses_unittest = True
            if root == "pytest":
                uses_pytest = True
        elif isinstance(node, ast.Name) and node.id == "TestCase":
            uses_unittest = True
        elif isinstance(node, ast.Attribute) and node.attr == "TestCase":
            uses_unittest = True

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            plain_tests = True
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                if name == "TestCase":
                    uses_unittest = True

    return uses_unittest, uses_pytest, plain_tests


def _dependencies(root: Path) -> list[str]:
    values: list[str] = []
    requirements = root / "requirements.txt"
    if requirements.is_file():
        for line in _read_text(requirements).splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                values.append(line)
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = _read_text(pyproject)
        marker = "dependencies"
        index = text.find(marker)
        if index != -1:
            start = text.find("[", index)
            end = text.find("]", start)
            if start != -1 and end != -1:
                for item in text[start + 1 : end].split(","):
                    item = item.strip().strip('"').strip("'")
                    if item:
                        values.append(item)
    setup_cfg = root / "setup.cfg"
    if setup_cfg.is_file():
        text = _read_text(setup_cfg)
        if "install_requires" in text:
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if "install_requires" in line:
                    for follow in lines[index + 1 : index + 20]:
                        stripped = follow.strip()
                        if not stripped:
                            break
                        if stripped.startswith("["):
                            break
                        values.append(stripped)
    # Keep the report readable and stable.
    return sorted(dict.fromkeys(values))[:50]


def _entry_points(root: Path) -> list[str]:
    values: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = _read_text(pyproject)
        if "[project.scripts]" in text:
            block = text.split("[project.scripts]", 1)[1]
            for line in block.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("[") and not stripped.startswith("[project"):
                    break
                if "=" in stripped:
                    values.append(stripped)
    setup_cfg = root / "setup.cfg"
    if setup_cfg.is_file():
        text = _read_text(setup_cfg)
        if "[options.entry_points]" in text:
            block = text.split("[options.entry_points]", 1)[1]
            for line in block.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("["):
                    break
                if "=" in stripped:
                    values.append(stripped)
    setup_py = root / "setup.py"
    if setup_py.is_file():
        text = _read_text(setup_py)
        if "entry_points" in text:
            values.append("setup.py declares entry_points")
    return values[:30]


def discover_repository(root: str | Path) -> RepositoryCapabilities:
    """Inspect a repository without running any of its code."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")

    python_files = _iter_python_files(root)
    test_paths = [path for path in python_files if is_test_file(str(path.relative_to(root)))]
    source_paths = [path for path in python_files if path not in test_paths]

    config_files: list[str] = []
    for name in (
        "pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "poetry.lock",
    ):
        if (root / name).is_file():
            config_files.append(name)

    pytest_config = _pytest_config_files(root)
    unittest_detected = False
    pytest_detected = bool(pytest_config)
    plain_tests = False
    for path in test_paths:
        uses_unittest, uses_pytest, has_plain = _test_usage(path)
        unittest_detected = unittest_detected or uses_unittest
        pytest_detected = pytest_detected or uses_pytest
        plain_tests = plain_tests or has_plain

    test_directories = sorted(
        {
            str(Path(path.relative_to(root)).parts[0])
            for path in test_paths
            if len(Path(path.relative_to(root)).parts) > 1
        }
    )

    if test_paths:
        if unittest_detected and not pytest_detected and not plain_tests:
            framework = "unittest"
        elif pytest_detected or plain_tests:
            framework = "pytest"
        else:
            framework = "custom"
    elif pytest_config:
        framework = "pytest"
    else:
        framework = "none"

    notes: list[str] = []
    automated = framework in {"pytest", "unittest"}
    if framework == "custom":
        notes.append(
            "Test files were detected, but this MVP does not support automated "
            "execution of that test framework."
        )
    if framework == "none":
        notes.append("No test framework was detected; generated behavioural validation may be used.")

    lines = 0
    for path in python_files:
        lines += len(_read_text(path, limit=2_000_000).splitlines())

    return RepositoryCapabilities(
        repo_path=str(root),
        name=root.name,
        python_files=len(python_files),
        source_files=len(source_paths),
        test_files=len(test_paths),
        lines=lines,
        test_framework=framework,
        test_file_paths=[str(path.relative_to(root)).replace("\\", "/") for path in test_paths],
        test_directories=test_directories,
        config_files=config_files,
        dependencies=_dependencies(root),
        entry_points=_entry_points(root),
        unittest_detected=unittest_detected,
        pytest_detected=pytest_detected,
        custom_tests_detected=framework == "custom",
        automated_execution_supported=automated,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Safety assessment
# ---------------------------------------------------------------------------


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_literal(key) and _is_literal(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_literal(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_literal(node.left) and _is_literal(node.right)
    return False


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = test.left
    comparator = test.comparators[0]
    return (
        isinstance(test.ops[0], ast.Eq)
        and isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(comparator, ast.Constant)
        and comparator.value == "__main__"
    )


@lru_cache(maxsize=128)
def assess_module_safety(source: str) -> tuple[str, ...]:
    """Return reasons why importing this module could execute side effects."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return (f"module cannot be parsed: {exc}",)

    problems: list[str] = []
    for node in tree.body:
        if isinstance(
            node,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            continue
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) for target in node.targets
        ) and _is_literal(node.value):
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and (
            node.value is None or _is_literal(node.value)
        ):
            continue
        if isinstance(node, ast.If) and _is_main_guard(node):
            continue
        problems.append(
            f"module-level {type(node).__name__} at line {node.lineno} runs on import"
        )
    return tuple(problems)


def _unsafe_aliases(tree: ast.Module) -> set[str]:
    aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in UNSAFE_MODULES:
                    aliases.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in UNSAFE_MODULES:
                for alias in node.names:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _function_problems(node: ast.AST, aliases: set[str]) -> list[str]:
    problems: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            problems.append(f"mutates global state at line {child.lineno}")
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            problems.append("generator functions are not supported by generated validation")
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.name.split(".")[0] in UNSAFE_MODULES:
                    problems.append(
                        f"imports unsafe module '{alias.name}' at line {child.lineno}"
                    )
        if isinstance(child, ast.ImportFrom):
            module = (child.module or "").split(".")[0]
            if module in UNSAFE_MODULES:
                problems.append(
                    f"imports unsafe module '{child.module}' at line {child.lineno}"
                )
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            if func.id in UNSAFE_CALLS:
                problems.append(f"calls unsafe function '{func.id}' at line {child.lineno}")
            elif func.id in aliases:
                problems.append(
                    f"calls external/unsafe function '{func.id}' at line {child.lineno}"
                )
        elif isinstance(func, ast.Attribute):
            root = func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and (root.id in UNSAFE_MODULES or root.id in aliases):
                problems.append(
                    f"uses external/unsafe module '{root.id}.{func.attr}' at line {child.lineno}"
                )
            if func.attr in UNSAFE_METHODS:
                problems.append(
                    f"calls side-effecting method '{func.attr}' at line {child.lineno}"
                )
    # Preserve order but remove duplicates.
    return list(dict.fromkeys(problems))


def infer_validation_params(
    source: str, function_name: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Infer parameters and annotations for generated validation.

    Returns ``(params, error)``.  Parameters are ordered and tagged with
    ``kwonly`` so the generated caller can build positional and keyword
    arguments correctly.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], f"cannot parse module: {exc}"

    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    if len(matches) != 1:
        return [], (
            f"function '{function_name}' is not a unique top-level function; "
            "generated validation only supports top-level functions"
        )
    node = matches[0]
    if isinstance(node, ast.AsyncFunctionDef):
        return [], "async functions are not supported by generated validation"
    if node.args.vararg or node.args.kwarg:
        return [], "variadic parameters are not supported by generated validation"
    if node.decorator_list:
        return [], "decorated functions are not supported by generated validation"

    params: list[dict[str, Any]] = []

    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    for arg, default in zip(positional, defaults):
        if arg.arg in {"self", "cls"}:
            return [], "methods are not supported by generated validation"
        annotation = ast.unparse(arg.annotation) if arg.annotation else None
        if annotation is None and isinstance(default, ast.Constant):
            annotation = type(default.value).__name__
        if annotation is None:
            return [], (
                f"parameter '{arg.arg}' has no type annotation and no literal default; "
                "generated inputs cannot be created safely"
            )
        params.append({"name": arg.arg, "annotation": annotation, "kwonly": False})

    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if arg.arg in {"self", "cls"}:
            return [], "methods are not supported by generated validation"
        annotation = ast.unparse(arg.annotation) if arg.annotation else None
        if annotation is None and isinstance(default, ast.Constant):
            annotation = type(default.value).__name__
        if annotation is None:
            return [], (
                f"parameter '{arg.arg}' has no type annotation and no literal default; "
                "generated inputs cannot be created safely"
            )
        params.append({"name": arg.arg, "annotation": annotation, "kwonly": True})

    return params, None


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def _normalize_annotation(annotation: str) -> str:
    text = annotation.strip().strip("'\"")
    text = text.replace("typing.", "")
    return text.replace(" ", "")


def _values_for(annotation: str, depth: int = 0) -> list[Any] | None:
    """Return representative values for an annotation, or None if unsupported."""
    if depth > 3:
        return None
    text = _normalize_annotation(annotation)

    if text in {"None", "NoneType"}:
        return [None]
    if text in {"int"}:
        return [0, 1, -1, 2, 5, 100, -7]
    if text in {"float"}:
        return [0.0, 1.0, -1.5, 3.14, 100.0]
    if text in {"bool"}:
        return [True, False]
    if text in {"str"}:
        return ["", "a", "abc", "hello world", "123", "a,b", "  padded  "]
    if text in {"bytes"}:
        return [b"", b"a", b"abc"]
    if text in {"complex"}:
        return [0j, 1 + 2j]

    if text.startswith("Optional[") and text.endswith("]"):
        inner = text[len("Optional[") : -1]
        values = _values_for(inner, depth + 1)
        if values is None:
            return None
        return [None] + values

    if text.startswith("Union[") and text.endswith("]"):
        inner = text[len("Union[") : -1]
        values: list[Any] = []
        for part in _split_top_level(inner):
            part_values = _values_for(part, depth + 1)
            if part_values is None:
                return None
            values.extend(part_values)
        return list(dict.fromkeys(values))[:10]

    if "|" in text:
        values = []
        for part in text.split("|"):
            part_values = _values_for(part, depth + 1)
            if part_values is None:
                return None
            values.extend(part_values)
        return list(dict.fromkeys(values))[:10]

    outer = None
    inner = ""
    if "[" in text and text.endswith("]"):
        outer = text[: text.index("[")]
        inner = text[text.index("[") + 1 : -1]
        outer = outer.lower()

    if outer in {"list", "sequence", "iterable", "collection", "mutablelist"}:
        inner_values = _values_for(inner, depth + 1) if inner else [0, 1, 2]
        if inner_values is None:
            return None
        if inner in {"int"}:
            return [[], [0], [1], [1, 2, 3], [0, -1, 100], [1, 1, 2, 2], [3, 2, 1], list(range(8))]
        if inner in {"float"}:
            return [[], [0.0], [1.0, -1.5], [0.0, 1.0, 3.14], [2.5, 2.5, -1.0]]
        if inner in {"str"}:
            return [[], [""], ["a"], ["a", "b", "c"], ["a", "a", "b"], ["c", "b", "a"]]
        if inner in {"bool"}:
            return [[], [True], [False, True], [True, False, True]]
        sample = inner_values[:4]
        return [[], sample[:1], sample[:2], sample[:3]]

    if outer in {"set", "frozenset"}:
        inner_values = _values_for(inner, depth + 1) if inner else [0, 1, 2]
        if inner_values is None:
            return None
        constructor = frozenset if outer == "frozenset" else set
        if inner in {"int"}:
            return [constructor(), constructor([0]), constructor([1, 2, 3]), constructor([0, -1, 100])]
        sample = inner_values[:3]
        return [constructor(), constructor(sample[:1]), constructor(sample)]

    if outer in {"dict", "mapping", "mutabledict"}:
        parts = _split_top_level(inner)
        if len(parts) != 2:
            return None
        key_values = _values_for(parts[0], depth + 1)
        value_values = _values_for(parts[1], depth + 1)
        if key_values is None or value_values is None:
            return None
        if parts[0] in {"str"} and parts[1] in {"int"}:
            return [{}, {"a": 1}, {"a": 1, "b": 2}, {"x": 0, "y": -1}, {"a": 0, "b": 0}]
        k1 = key_values[0]
        v1 = value_values[0]
        k2 = key_values[1 % len(key_values)]
        v2 = value_values[1 % len(value_values)]
        return [{}, {k1: v1}, {k1: v1, k2: v2}]

    if outer in {"tuple"}:
        if inner.endswith(",..."):
            inner = inner[:-4]
            inner_values = _values_for(inner, depth + 1)
            if inner_values is None:
                return None
            sample = inner_values[:3]
            return [(), tuple(sample[:1]), tuple(sample), tuple(inner_values[:4])]
        parts = _split_top_level(inner)
        if not parts:
            return [()]
        per_part = [_values_for(part, depth + 1) for part in parts]
        if any(values is None for values in per_part):
            return None
        first = tuple(values[0] for values in per_part if values)
        second = tuple(values[1 % len(values)] for values in per_part if values)
        return [(), first, second, first + second]

    # Bare container names are treated as integer containers.
    if text in {"list", "sequence", "iterable", "collection"}:
        return _values_for("list[int]", depth + 1)
    if text in {"set", "frozenset"}:
        return _values_for("set[int]", depth + 1)
    if text in {"dict", "mapping"}:
        return _values_for("dict[str,int]", depth + 1)
    if text in {"tuple"}:
        return _values_for("tuple[int,...]", depth + 1)

    return None


def generate_cases(
    params: list[dict[str, Any]], max_cases: int = MAX_GENERATED_CASES
) -> tuple[list[dict[str, Any]], str | None]:
    """Generate representative argument sets for the given parameters."""
    if not params:
        return [{"args": [], "kwargs": {}}], None

    per_param: list[list[Any]] = []
    for param in params:
        values = _values_for(str(param.get("annotation", "")))
        if values is None:
            return [], f"unsupported annotation for generated inputs: {param.get('annotation')}"
        if not values:
            return [], f"no safe inputs could be generated for {param.get('annotation')}"
        per_param.append(values)

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(combo: tuple[Any, ...]) -> None:
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for param, value in zip(params, combo):
            if param.get("kwonly"):
                kwargs[str(param["name"])] = value
            else:
                args.append(value)
        case = {"args": args, "kwargs": kwargs}
        key = repr(case)
        if key not in seen:
            seen.add(key)
            cases.append(case)

    # A balanced pass first so every parameter gets its edge values.
    longest = max(len(values) for values in per_param)
    for index in range(longest):
        add(tuple(values[index % len(values)] for values in per_param))
        if len(cases) >= max_cases:
            return cases, None

    # Then fill with combinations up to the cap.
    for combo in itertools.product(*per_param):
        add(combo)
        if len(cases) >= max_cases:
            break

    return cases, None


def assess_function_safety(source: str, function_name: str) -> dict[str, Any]:
    """Decide whether a function may be executed with generated inputs.

    This is a conservative static check.  It is not a sandbox and cannot prove
    that a function has no side effects; it only refuses the obvious cases.
    """
    reasons: list[str] = []
    module_reasons = list(assess_module_safety(source))
    reasons.extend(module_reasons)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "safe": False,
            "reasons": [f"module cannot be parsed: {exc}"],
            "module_reasons": module_reasons,
            "params": [],
            "input_error": "module cannot be parsed",
            "generated_possible": False,
        }

    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    if len(matches) != 1:
        reasons.append(
            f"function '{function_name}' is not a unique top-level function"
        )
        node = matches[0] if matches else None
    else:
        node = matches[0]

    if node is not None:
        if isinstance(node, ast.AsyncFunctionDef):
            reasons.append("async functions are not supported")
        if node.decorator_list:
            reasons.append("decorated functions are not supported")
        if node.args.vararg or node.args.kwarg:
            reasons.append("variadic parameters are not supported")
        reasons.extend(_function_problems(node, _unsafe_aliases(tree)))

    reasons = list(dict.fromkeys(reasons))
    params, input_error = infer_validation_params(source, function_name)
    safe = not reasons
    generated_possible = safe and input_error is None

    return {
        "safe": safe,
        "reasons": reasons,
        "module_reasons": module_reasons,
        "params": params,
        "input_error": input_error,
        "generated_possible": generated_possible,
    }


def evaluate_candidates(repo_dir: str | Path, analysis: Analysis) -> dict[str, Any]:
    """Assess every actionable finding for execution/generated-input safety."""
    root = Path(repo_dir).expanduser().resolve()
    sources: dict[str, str] = {}
    seen: set[str] = set()
    generated_safe: list[dict[str, Any]] = []
    execution_safe: list[dict[str, Any]] = []
    unsafe: list[dict[str, Any]] = []

    for finding in rank_findings(analysis.findings):
        if finding.kind not in ACTIONABLE_KINDS or not finding.auto_optimizable:
            continue
        if not finding.function or is_test_file(finding.file):
            continue
        key = f"{finding.file}::{finding.function}"
        if key in seen:
            continue
        seen.add(key)
        if finding.file not in sources:
            try:
                sources[finding.file] = (root / finding.file).read_text(encoding="utf-8")
            except OSError as exc:
                unsafe.append(
                    {
                        "file": finding.file,
                        "function": finding.function,
                        "reasons": [f"cannot read source: {exc}"],
                        "generated_possible": False,
                    }
                )
                continue
        assessment = assess_function_safety(sources[finding.file], finding.function)
        entry = {
            "file": finding.file,
            "function": finding.function,
            "kind": finding.kind,
            "safe": assessment["safe"],
            "generated_possible": assessment["generated_possible"],
            "reasons": assessment["reasons"],
            "input_error": assessment["input_error"],
            "params": assessment["params"],
        }
        if assessment["safe"]:
            execution_safe.append(entry)
        else:
            unsafe.append(entry)
        if assessment["generated_possible"]:
            generated_safe.append(entry)

    return {
        "safe_candidates": generated_safe,
        "execution_safe_candidates": execution_safe,
        "unsafe_candidates": unsafe,
        "safe_count": len(generated_safe),
        "execution_safe_count": len(execution_safe),
        "unsafe_count": len(unsafe),
        "candidates_found": len(seen),
    }


def choose_validation_strategy(
    capabilities: RepositoryCapabilities,
    candidate_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Pick the strongest available strategy and state its confidence."""
    framework = capabilities.test_framework
    safe_count = int(candidate_evaluation.get("safe_count", 0))

    if framework in {"pytest", "unittest"}:
        return {
            "strategy": "EXISTING_TESTS",
            "confidence": "HIGH",
            "runner": framework,
            "reason": f"{framework} test suite detected",
            "generated_validation_possible": safe_count > 0,
            "safe_candidates": safe_count,
        }

    if framework == "custom":
        if safe_count > 0:
            return {
                "strategy": "GENERATED_BEHAVIORAL_TESTS",
                "confidence": "MEDIUM",
                "runner": "generated",
                "reason": (
                    "A test framework was detected but automated execution is not "
                    "supported; safe candidates will use generated behavioural validation."
                ),
                "generated_validation_possible": True,
                "safe_candidates": safe_count,
            }
        return {
            "strategy": "STATIC_ONLY",
            "confidence": "LOW",
            "runner": "none",
            "reason": (
                "A test framework was detected but automated execution is not supported, "
                "and no safe candidate could be validated behaviourally."
            ),
            "generated_validation_possible": False,
            "safe_candidates": 0,
        }

    if safe_count > 0:
        return {
            "strategy": "GENERATED_BEHAVIORAL_TESTS",
            "confidence": "MEDIUM",
            "runner": "generated",
            "reason": (
                "No tests detected; safe typed functions can be validated by "
                "differential behavioural checks."
            ),
            "generated_validation_possible": True,
            "safe_candidates": safe_count,
        }

    return {
        "strategy": "STATIC_ONLY",
        "confidence": "LOW",
        "runner": "none",
        "reason": "No tests and no safe behavioural validation available.",
        "generated_validation_possible": False,
        "safe_candidates": 0,
    }


# ---------------------------------------------------------------------------
# Differential behavioural validation
# ---------------------------------------------------------------------------

BEHAVIOR_RUNNER = r'''
import copy
import importlib.util
import json
import math
import sys
import traceback

MARKER = "BEHAVIOR_JSON:"


def fail(message):
    print(MARKER + json.dumps({"ok": False, "error": message}))
    raise SystemExit(0)


def decode(value):
    if isinstance(value, dict) and value.get("__codeopt_type__") == "tuple":
        return tuple(decode(item) for item in value.get("items", []))
    if isinstance(value, dict) and value.get("__codeopt_type__") == "set":
        return set(decode(item) for item in value.get("items", []))
    if isinstance(value, dict) and value.get("__codeopt_type__") == "frozenset":
        return frozenset(decode(item) for item in value.get("items", []))
    if isinstance(value, dict) and value.get("__codeopt_type__") == "dict":
        return {decode(key): decode(item) for key, item in value.get("items", [])}
    if isinstance(value, dict) and value.get("__codeopt_type__") == "float":
        text = value.get("value")
        if text == "nan":
            return float("nan")
        if text == "inf":
            return float("inf")
        if text == "-inf":
            return float("-inf")
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def values_equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError, OverflowError):
            return False
    if type(left) is not type(right):
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        if len(left) != len(right):
            return False
        for key, value in left.items():
            try:
                if key not in right:
                    return False
            except Exception:
                return False
            if not values_equal(value, right[key]):
                return False
        return True
    if isinstance(left, (set, frozenset)):
        if len(left) != len(right):
            return False
        remaining = list(right)
        for item in left:
            for index, candidate in enumerate(remaining):
                if values_equal(item, candidate):
                    remaining.pop(index)
                    break
            else:
                return False
        return True
    try:
        return bool(left == right)
    except Exception:
        return False


def describe(value, limit=220):
    try:
        text = repr(value)
    except Exception:
        text = "<unrepresentable " + type(value).__name__ + ">"
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def call(function, args, kwargs):
    before_args = copy.deepcopy(args)
    before_kwargs = copy.deepcopy(kwargs)
    try:
        result = function(*args, **kwargs)
        status = "return"
        value = result
    except SystemExit as exc:
        status = "raise"
        value = ("SystemExit", str(exc))
    except BaseException as exc:
        status = "raise"
        value = (type(exc).__name__, str(exc))
    try:
        mutated = args != before_args or kwargs != before_kwargs
    except Exception:
        mutated = True
    return status, value, mutated


def load_module(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load module from " + path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


spec = json.loads(sys.argv[1])
for entry in spec.get("sys_paths", []):
    if entry and entry not in sys.path:
        sys.path.insert(0, entry)

try:
    baseline = load_module("_codeopt_behavior_baseline", spec["baseline_path"])
    optimized = load_module("_codeopt_behavior_optimized", spec["optimized_path"])
    before_fn = getattr(baseline, spec["function"], None)
    after_fn = getattr(optimized, spec["function"], None)
    if not callable(before_fn) or not callable(after_fn):
        fail("function is not callable in one of the modules")

    cases = [decode(case) for case in spec.get("cases", [])]
    if not cases:
        fail("no generated cases were supplied")

    matched = 0
    mismatches = []
    for index, case in enumerate(cases):
        args = list(case.get("args", []))
        kwargs = dict(case.get("kwargs", {}))
        status_before, value_before, mutated_before = call(before_fn, args, kwargs)
        status_after, value_after, mutated_after = call(after_fn, args, kwargs)

        if mutated_before or mutated_after:
            mismatches.append(
                {
                    "case": index,
                    "reason": "function mutates its input arguments",
                    "input": describe(case),
                    "original": "input mutated" if mutated_before else "input not mutated",
                    "optimized": "input mutated" if mutated_after else "input not mutated",
                }
            )
            break

        if status_before != status_after:
            mismatches.append(
                {
                    "case": index,
                    "reason": "return/exception behaviour differs",
                    "input": describe(case),
                    "original": status_before + ": " + describe(value_before),
                    "optimized": status_after + ": " + describe(value_after),
                }
            )
        elif status_before == "raise":
            if not values_equal(value_before, value_after):
                mismatches.append(
                    {
                        "case": index,
                        "reason": "raised exception differs",
                        "input": describe(case),
                        "original": describe(value_before),
                        "optimized": describe(value_after),
                    }
                )
            else:
                matched += 1
        elif values_equal(value_before, value_after):
            matched += 1
        else:
            mismatches.append(
                {
                    "case": index,
                    "reason": "return value differs",
                    "input": describe(case),
                    "original": describe(value_before),
                    "optimized": describe(value_after),
                }
            )

        if len(mismatches) >= 5:
            break

    print(
        MARKER
        + json.dumps(
            {
                "ok": True,
                "cases_total": len(cases),
                "cases_matched": matched,
                "passed": not mismatches and matched == len(cases),
                "mismatches": mismatches,
            }
        )
    )
except SystemExit:
    raise
except Exception:
    print(MARKER + json.dumps({"ok": False, "error": traceback.format_exc(limit=6)}))
'''


def _encode_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__codeopt_type__": "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, frozenset):
        return {"__codeopt_type__": "frozenset", "items": [_encode_value(item) for item in value]}
    if isinstance(value, set):
        return {"__codeopt_type__": "set", "items": [_encode_value(item) for item in value]}
    if isinstance(value, dict):
        return {
            "__codeopt_type__": "dict",
            "items": [[_encode_value(key), _encode_value(item)] for key, item in value.items()],
        }
    if isinstance(value, float):
        if math.isnan(value):
            return {"__codeopt_type__": "float", "value": "nan"}
        if math.isinf(value):
            return {"__codeopt_type__": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    return value


def _encode_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "args": [_encode_value(value) for value in case.get("args", [])],
            "kwargs": {
                str(key): _encode_value(value) for key, value in case.get("kwargs", {}).items()
            },
        }
        for case in cases
    ]


def _input_description(params: list[dict[str, Any]], cases: int) -> str:
    if not params:
        return f"{cases} generated cases (no arguments)"
    names = ", ".join(f"{param['name']}: {param['annotation']}" for param in params)
    return f"{cases} generated cases ({names})"


def run_behavioral_validation(
    repo_dir: str | Path,
    baseline_source: str,
    optimized_path: str | Path,
    function_name: str,
    cases: list[dict[str, Any]],
    timeout: float = 120.0,
) -> ValidationResult:
    """Compare original and optimized behaviour on generated inputs."""
    repo_dir = Path(repo_dir).expanduser().resolve()
    optimized_path = Path(optimized_path).expanduser().resolve()
    started = time.perf_counter()

    if not cases:
        return ValidationResult(
            strategy="GENERATED_BEHAVIORAL_TESTS",
            confidence="MEDIUM",
            ran=False,
            passed=False,
            reason="no generated cases were available",
        )

    baseline_path = optimized_path.parent / f"_codeopt_behavior_{uuid.uuid4().hex[:10]}.py"
    try:
        baseline_path.write_text(baseline_source, encoding="utf-8")
        spec: dict[str, Any] = {
            "baseline_path": str(baseline_path),
            "optimized_path": str(optimized_path),
            "function": function_name,
            "cases": _encode_cases(cases),
            "sys_paths": [str(repo_dir), str(optimized_path.parent)],
        }
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["NO_COLOR"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", BEHAVIOR_RUNNER, json.dumps(spec)],
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        duration = time.perf_counter() - started
        payload = None
        for line in reversed((completed.stdout or "").splitlines()):
            if line.startswith(BEHAVIOR_MARKER):
                payload = json.loads(line[len(BEHAVIOR_MARKER) :])
                break
        if payload is None:
            detail = ((completed.stdout or "") + (completed.stderr or ""))[-4000:].strip()
            return ValidationResult(
                strategy="GENERATED_BEHAVIORAL_TESTS",
                confidence="MEDIUM",
                ran=False,
                passed=False,
                cases_total=len(cases),
                duration_seconds=round(duration, 3),
                reason=f"behavioural validation produced no result: {detail or 'no output'}",
            )
        if not payload.get("ok"):
            return ValidationResult(
                strategy="GENERATED_BEHAVIORAL_TESTS",
                confidence="MEDIUM",
                ran=False,
                passed=False,
                cases_total=len(cases),
                duration_seconds=round(duration, 3),
                reason=str(payload.get("error") or "behavioural validation failed"),
            )
        return ValidationResult(
            strategy="GENERATED_BEHAVIORAL_TESTS",
            confidence="MEDIUM",
            ran=True,
            passed=bool(payload.get("passed")),
            cases_total=int(payload.get("cases_total", len(cases))),
            cases_matched=int(payload.get("cases_matched", 0)),
            mismatches=list(payload.get("mismatches") or []),
            duration_seconds=round(duration, 3),
            reason=(
                None
                if payload.get("passed")
                else "behaviour differs from the original implementation"
            ),
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            strategy="GENERATED_BEHAVIORAL_TESTS",
            confidence="MEDIUM",
            ran=False,
            passed=False,
            cases_total=len(cases),
            duration_seconds=round(time.perf_counter() - started, 3),
            reason=f"behavioural validation timed out after {timeout}s",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ValidationResult(
            strategy="GENERATED_BEHAVIORAL_TESTS",
            confidence="MEDIUM",
            ran=False,
            passed=False,
            cases_total=len(cases),
            duration_seconds=round(time.perf_counter() - started, 3),
            reason=f"behavioural validation error: {exc}",
        )
    finally:
        try:
            baseline_path.unlink(missing_ok=True)
        except OSError:
            pass


def candidate_case_count(params: list[dict[str, Any]]) -> tuple[int, str | None]:
    """Convenience helper used by reports and tests."""
    cases, error = generate_cases(params)
    return len(cases), error
