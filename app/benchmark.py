"""Differential benchmarking of an original and an optimized function.

The benchmark runs both implementations in one isolated subprocess, with the
same generated inputs, alternating the order of measurement.  It also compares
the return values, which gives the agent a real correctness signal beyond the
unit tests.
"""

from __future__ import annotations

import ast
import json
import os
import statistics
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .models import BenchmarkResult
from .optimizer import OptimizationError, find_function_node

BENCH_MARKER = "BENCH_JSON:"

RUNNER = r'''
import copy
import importlib.util
import json
import random
import re
import statistics
import sys
import time
import traceback


def fail(message):
    print("BENCH_JSON:" + json.dumps({"ok": False, "error": message}))
    raise SystemExit(0)


def split_top_level(text):
    parts, depth, current = [], 0, ""
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)
    return parts


def parse_annotation(annotation):
    ann = annotation.replace("typing.", "").replace(" ", "")
    if ann.startswith("Optional[") and ann.endswith("]"):
        ann = ann[9:-1]
    if "|" in ann:
        ann = ann.split("|")[0]
    if ann in ("list", "List", "Sequence", "Iterable"):
        return ("list", "int")
    if ann in ("set", "Set", "FrozenSet"):
        return ("set", "int")
    if ann in ("tuple", "Tuple"):
        return ("tuple", "int")
    if ann in ("dict", "Dict", "Mapping"):
        return ("dict", "int")
    if ann in ("int", "float", "str", "bool"):
        return ("scalar", ann)
    match = re.match(r"^(list|set|tuple|dict)\[(.*)\]$", ann)
    if match:
        outer, inner = match.group(1), match.group(2)
        if outer == "dict":
            key, value = split_top_level(inner)[:2]
            return ("dict", (key, value))
        if outer == "tuple" and inner.endswith(",..."):
            return ("tuple", inner[:-4])
        return (outer, inner)
    raise ValueError("unsupported annotation: " + annotation)


def generate(annotation, size, rng):
    kind, inner = parse_annotation(annotation)
    if kind == "scalar":
        if inner == "int":
            return rng.randint(0, size)
        if inner == "float":
            return round(rng.uniform(0.0, 1.0), 6)
        if inner == "str":
            return "s" + str(rng.randint(0, size))
        return bool(rng.getrandbits(1))
    if kind == "list":
        return [generate(inner, size, rng) for _ in range(size)]
    if kind == "set":
        return {generate(inner, size, rng) for _ in range(size)}
    if kind == "tuple":
        return tuple(generate(inner, size, rng) for _ in range(size))
    if kind == "dict":
        key_type, value_type = inner
        return {generate(key_type, size, rng): generate(value_type, size, rng) for _ in range(size)}
    raise ValueError("unsupported kind: " + kind)


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
    rng = random.Random(spec.get("seed", 1234))
    size = int(spec.get("size", 1000))
    args = [generate(param["annotation"], size, rng) for param in spec["params"]]
    snapshot = copy.deepcopy(args)

    baseline = load_module("_codeopt_baseline_mod", spec["baseline_path"])
    optimized = load_module("_codeopt_optimized_mod", spec["optimized_path"])

    before_fn = getattr(baseline, spec["function"], None)
    after_fn = getattr(optimized, spec["function"], None)
    if not callable(before_fn) or not callable(after_fn):
        fail("function is not callable in one of the modules")

    out_before = before_fn(*args)
    if args != snapshot:
        fail("function mutates its input arguments; differential benchmarking is not supported")
    out_after = after_fn(*args)
    if args != snapshot:
        fail("function mutates its input arguments; differential benchmarking is not supported")

    try:
        outputs_match = type(out_before) is type(out_after) and out_before == out_after
    except Exception:
        outputs_match = None

    warmup = int(spec.get("warmup", 1))
    repeats = int(spec.get("repeats", 3))
    for _ in range(warmup):
        before_fn(*args)
        after_fn(*args)

    before_times, after_times = [], []
    for index in range(repeats):
        if index % 2 == 0:
            start = time.perf_counter()
            before_fn(*args)
            before_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            after_fn(*args)
            after_times.append(time.perf_counter() - start)
        else:
            start = time.perf_counter()
            after_fn(*args)
            after_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            before_fn(*args)
            before_times.append(time.perf_counter() - start)

    print(
        "BENCH_JSON:"
        + json.dumps(
            {
                "ok": True,
                "before_times": before_times,
                "after_times": after_times,
                "before_median": statistics.median(before_times),
                "after_median": statistics.median(after_times),
                "outputs_match": outputs_match,
                "output_type_before": type(out_before).__name__,
                "output_type_after": type(out_after).__name__,
            }
        )
    )
except SystemExit:
    raise
except Exception:
    print("BENCH_JSON:" + json.dumps({"ok": False, "error": traceback.format_exc(limit=4)}))
'''


def infer_params(source: str, function_name: str) -> tuple[list[dict[str, str]], str | None]:
    """Infer benchmark parameter types from annotations (or literal defaults)."""
    try:
        tree = ast.parse(source)
        node = find_function_node(tree, function_name)
    except (SyntaxError, OptimizationError) as exc:
        return [], f"cannot inspect signature: {exc}"

    if node.args.vararg or node.args.kwarg:
        return [], "variadic parameters are not supported by the automatic benchmark"

    params: list[dict[str, str]] = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    for arg, default in zip(positional, defaults):
        if arg.arg in {"self", "cls"}:
            continue
        annotation = ast.unparse(arg.annotation) if arg.annotation else None
        if annotation is None and isinstance(default, ast.Constant):
            annotation = type(default.value).__name__
        if annotation is None:
            return [], f"parameter '{arg.arg}' has no type annotation; benchmark cannot be generated safely"
        params.append({"name": arg.arg, "annotation": annotation})

    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if arg.arg in {"self", "cls"}:
            continue
        annotation = ast.unparse(arg.annotation) if arg.annotation else None
        if annotation is None and isinstance(default, ast.Constant):
            annotation = type(default.value).__name__
        if annotation is None:
            return [], f"parameter '{arg.arg}' has no type annotation; benchmark cannot be generated safely"
        params.append({"name": arg.arg, "annotation": annotation})

    return params, None


def _input_description(params: list[dict[str, str]], size: int) -> str:
    if not params:
        return "no arguments"
    return ", ".join(f"{p['name']}: {p['annotation']} (size {size})" for p in params)


def compare_functions(
    repo_dir: Path,
    baseline_source: str,
    optimized_path: Path,
    function_name: str,
    params: list[dict[str, str]],
    size: int = 1000,
    repeats: int = 3,
    warmup: int = 1,
    seed: int = 1234,
    timeout: float = 180.0,
) -> BenchmarkResult:
    """Benchmark the original source against the modified file.

    Returns ``available=False`` with an explicit reason whenever a reliable
    measurement cannot be produced.  It never fabricates numbers.
    """
    repo_dir = Path(repo_dir).resolve()
    optimized_path = Path(optimized_path).resolve()
    repeats = max(1, int(repeats))
    size = max(1, int(size))

    baseline_path = optimized_path.parent / f"_codeopt_baseline_{uuid.uuid4().hex[:10]}.py"
    try:
        baseline_path.write_text(baseline_source, encoding="utf-8")
        spec: dict[str, Any] = {
            "baseline_path": str(baseline_path),
            "optimized_path": str(optimized_path),
            "function": function_name,
            "params": params,
            "size": size,
            "repeats": repeats,
            "warmup": max(0, int(warmup)),
            "seed": seed,
            "sys_paths": [str(repo_dir), str(optimized_path.parent)],
        }
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", RUNNER, json.dumps(spec)],
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        payload = None
        for line in reversed((completed.stdout or "").splitlines()):
            if line.startswith(BENCH_MARKER):
                payload = json.loads(line[len(BENCH_MARKER) :])
                break
        if payload is None:
            detail = ((completed.stdout or "") + (completed.stderr or ""))[-4000:]
            return BenchmarkResult(
                available=False,
                function=function_name,
                input_description=_input_description(params, size),
                reason=f"benchmark subprocess produced no result: {detail.strip() or 'no output'}",
            )
        if not payload.get("ok"):
            return BenchmarkResult(
                available=False,
                function=function_name,
                input_description=_input_description(params, size),
                reason=payload.get("error", "benchmark failed"),
            )

        before_times = [float(x) for x in payload["before_times"]]
        after_times = [float(x) for x in payload["after_times"]]
        before_median = float(payload["before_median"])
        after_median = float(payload["after_median"])
        improvement = None
        if before_median > 0:
            improvement = round((before_median - after_median) / before_median * 100.0, 2)

        return BenchmarkResult(
            available=True,
            function=function_name,
            before_seconds=round(before_median, 6),
            after_seconds=round(after_median, 6),
            improvement_pct=improvement,
            before_times=[round(x, 6) for x in before_times],
            after_times=[round(x, 6) for x in after_times],
            repeats=repeats,
            outputs_match=payload.get("outputs_match"),
            input_description=_input_description(params, size),
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            available=False,
            function=function_name,
            input_description=_input_description(params, size),
            reason=f"benchmark timed out after {timeout}s",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return BenchmarkResult(
            available=False,
            function=function_name,
            input_description=_input_description(params, size),
            reason=f"benchmark error: {exc}",
        )
    finally:
        try:
            baseline_path.unlink(missing_ok=True)
        except OSError:
            pass


def compare_sources(
    repo_dir: Path,
    baseline_source: str,
    optimized_source: str,
    function_name: str,
    params: list[dict[str, str]],
    size: int = 1000,
    repeats: int = 3,
    timeout: float = 180.0,
) -> BenchmarkResult:
    """Convenience wrapper for callers that only have source strings."""
    repo_dir = Path(repo_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="codeopt_bench_") as tmp:
        optimized_path = Path(tmp) / "optimized_target.py"
        optimized_path.write_text(optimized_source, encoding="utf-8")
        return compare_functions(
            repo_dir=repo_dir,
            baseline_source=baseline_source,
            optimized_path=optimized_path,
            function_name=function_name,
            params=params,
            size=size,
            repeats=repeats,
            timeout=timeout,
        )
