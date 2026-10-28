"""Static analysis for Python source code.

The analyzer is deliberately conservative: it only reports patterns that can be
described precisely from the AST.  It never edits code and never guesses at
runtime behaviour.
"""

from __future__ import annotations

import ast
import hashlib
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import (
    Analysis,
    ClassInfo,
    FileAnalysis,
    Finding,
    FunctionInfo,
    ImportInfo,
)

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".codeoptimizer",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "site-packages",
}

# Builtins that are pure and cheap to reason about when hoisted out of a loop.
PURE_BUILTINS = {
    "len",
    "sum",
    "min",
    "max",
    "abs",
    "sorted",
    "str",
    "int",
    "float",
    "any",
    "all",
}

# Finding kinds that the agent is allowed to attempt automatically.
ACTIONABLE_KINDS = {
    "nested_loops",
    "membership_in_loop",
    "repeated_computation",
    "string_concat_in_loop",
    "inefficient_data_structure",
}

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _finding_id(file: str, kind: str, line: int, function: str | None) -> str:
    digest = hashlib.sha1(f"{file}:{kind}:{line}:{function}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}-{digest}"


def _complexity_label(depth: int) -> str:
    if depth <= 0:
        return "O(1)"
    if depth == 1:
        return "O(n)"
    if depth == 2:
        return "O(n^2)"
    if depth == 3:
        return "O(n^3)"
    return f"O(n^{depth})"


def iter_python_files(root: Path, max_files: int = 1000) -> list[Path]:
    """Return analyzable Python files below ``root`` in a stable order."""
    root = Path(root)
    files: list[Path] = []
    for path in sorted(root.rglob("*.py"), key=lambda p: str(p).lower()):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in EXCLUDED_DIRS for part in parts):
            continue
        files.append(path)
        if len(files) >= max_files:
            break
    return files


def _walk_loops(
    node: ast.AST, depth: int = 0, parent: ast.AST | None = None
) -> Iterator[tuple[ast.AST, int, ast.AST | None]]:
    """Yield (loop, depth, enclosing_loop) skipping nested function/class scopes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            yield child, depth + 1, parent
            yield from _walk_loops(child, depth + 1, child)
        else:
            yield from _walk_loops(child, depth, parent)


def _walk_no_defs(node: ast.AST) -> Iterator[ast.AST]:
    """Walk a subtree but do not descend into nested function/class definitions."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from _walk_no_defs(child)


def _loop_target(node: ast.AST) -> str:
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return ast.unparse(node.target)
    return "while"


def _assigned_containers(node: ast.AST) -> dict[str, str]:
    """Map simple variable names to the container type they are assigned."""
    assigned: dict[str, str] = {}

    def classify(value: ast.AST) -> str | None:
        if isinstance(value, ast.List):
            return "list"
        if isinstance(value, ast.Tuple):
            return "tuple"
        if isinstance(value, ast.Set):
            return "set"
        if isinstance(value, ast.Dict):
            return "dict"
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id in {"list", "tuple", "set", "dict", "sorted"}:
                return "set" if value.func.id == "sorted" else value.func.id
        return None

    for child in _walk_no_defs(node):
        if isinstance(child, ast.Assign):
            kind = classify(child.value)
            if kind:
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        assigned[target.id] = kind
        elif isinstance(child, ast.AnnAssign) and child.value is not None:
            kind = classify(child.value)
            if kind and isinstance(child.target, ast.Name):
                assigned[child.target.id] = kind
    return assigned


def _classify_container(node: ast.AST, assigned: dict[str, str]) -> str | None:
    if isinstance(node, ast.List):
        return "list_literal"
    if isinstance(node, ast.Tuple):
        return "tuple_literal"
    if isinstance(node, ast.Set):
        return "set_literal"
    if isinstance(node, ast.Dict):
        return "dict_literal"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"list", "tuple", "sorted"}:
            return f"{node.func.id}_call"
        if node.func.id in {"set", "dict"}:
            return f"{node.func.id}_call"
    if isinstance(node, ast.Name):
        kind = assigned.get(node.id)
        if kind:
            return f"name_{kind}"
    return None


def _detect_duplicate_match(func: ast.AST) -> dict[str, str] | None:
    """Detect ``for x in a: for y in b: if x == y: result.append(x)``.

    Returns the involved variable names, or ``None`` when the shape does not
    match exactly.  The transformation that uses this result is still verified
    with tests and a differential benchmark before being kept.
    """
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or func.args.args == []:
        return None
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if len(body) != 3:
        return None

    init, outer, ret = body
    if not (
        isinstance(init, ast.Assign)
        and len(init.targets) == 1
        and isinstance(init.targets[0], ast.Name)
        and isinstance(init.value, ast.List)
        and not init.value.elts
    ):
        return None
    result_name = init.targets[0].id

    if not (
        isinstance(outer, (ast.For, ast.AsyncFor))
        and isinstance(outer.target, ast.Name)
        and isinstance(outer.iter, ast.Name)
        and len(outer.body) == 1
    ):
        return None
    outer_var = outer.target.id
    outer_iter = outer.iter.id

    inner = outer.body[0]
    if not (
        isinstance(inner, (ast.For, ast.AsyncFor))
        and isinstance(inner.target, ast.Name)
        and isinstance(inner.iter, ast.Name)
        and len(inner.body) == 1
    ):
        return None
    inner_var = inner.target.id
    inner_iter = inner.iter.id

    cond = inner.body[0]
    if not (
        isinstance(cond, ast.If)
        and len(cond.body) == 1
        and isinstance(cond.test, ast.Compare)
        and len(cond.test.ops) == 1
        and isinstance(cond.test.ops[0], ast.Eq)
        and len(cond.test.comparators) == 1
    ):
        return None
    left, right = cond.test.left, cond.test.comparators[0]
    if not (isinstance(left, ast.Name) and isinstance(right, ast.Name)):
        return None
    if {left.id, right.id} != {outer_var, inner_var}:
        return None

    stmt = cond.body[0]
    if not (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "append"
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == result_name
        and len(stmt.value.args) == 1
        and isinstance(stmt.value.args[0], ast.Name)
        and stmt.value.args[0].id == outer_var
    ):
        return None

    if not (isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name) and ret.value.id == result_name):
        return None

    return {
        "result": result_name,
        "outer_var": outer_var,
        "outer_iter": outer_iter,
        "inner_var": inner_var,
        "inner_iter": inner_iter,
    }


def _scan_dead_code(
    body: list[ast.stmt],
    file_rel: str,
    function: str | None,
    findings: list[Finding],
    skip_defs: bool = False,
) -> None:
    for index, stmt in enumerate(body):
        if skip_defs and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)) and index < len(body) - 1:
            nxt = body[index + 1]
            findings.append(
                Finding(
                    id=_finding_id(file_rel, "dead_code", nxt.lineno, function),
                    kind="dead_code",
                    file=file_rel,
                    line=nxt.lineno,
                    function=function,
                    severity="low",
                    message=f"Unreachable code after '{type(stmt).__name__.lower()}'",
                    suggestion="Remove the unreachable statements.",
                    confidence=0.95,
                    auto_optimizable=False,
                    details={"terminator": type(stmt).__name__},
                )
            )
            break

        if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Constant) and not stmt.test.value:
            findings.append(
                Finding(
                    id=_finding_id(file_rel, "dead_code", stmt.lineno, function),
                    kind="dead_code",
                    file=file_rel,
                    line=stmt.lineno,
                    function=function,
                    severity="low",
                    message="'if False' branch is never executed",
                    suggestion="Remove the dead branch.",
                    confidence=0.99,
                    auto_optimizable=False,
                    details={"test": ast.unparse(stmt.test)},
                )
            )

        if isinstance(stmt, ast.While) and isinstance(stmt.test, ast.Constant) and not stmt.test.value:
            findings.append(
                Finding(
                    id=_finding_id(file_rel, "dead_code", stmt.lineno, function),
                    kind="dead_code",
                    file=file_rel,
                    line=stmt.lineno,
                    function=function,
                    severity="low",
                    message="'while False' loop is never executed",
                    suggestion="Remove the dead loop.",
                    confidence=0.99,
                    auto_optimizable=False,
                    details={"test": ast.unparse(stmt.test)},
                )
            )

        for field_name in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field_name, None)
            if isinstance(child, list):
                _scan_dead_code(child, file_rel, function, findings, skip_defs=skip_defs)
        if isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                _scan_dead_code(handler.body, file_rel, function, findings, skip_defs=skip_defs)


def _analyze_scope(
    node: ast.AST,
    file_rel: str,
    function: str | None,
    findings: list[Finding],
) -> None:
    """Detect performance findings inside one function or module body."""
    loops = list(_walk_loops(node))
    assigned = _assigned_containers(node)
    loop_ids_with_children = {id(parent) for _, _, parent in loops if parent is not None}

    for loop, depth, _parent in loops:
        if depth == 1 and id(loop) in loop_ids_with_children:
            inner_loops = [
                (candidate, d)
                for candidate, d, ancestor in loops
                if ancestor is not None and _is_descendant_loop(loop, candidate)
            ]
            has_equality = False
            has_membership = False
            for inner, _ in inner_loops:
                for child in _walk_no_defs(inner):
                    if isinstance(child, ast.Compare):
                        if any(isinstance(op, ast.Eq) for op in child.ops):
                            has_equality = True
                        if any(isinstance(op, (ast.In, ast.NotIn)) for op in child.ops):
                            has_membership = True
            pattern = _detect_duplicate_match(node) if isinstance(node, ast.FunctionDef) else None
            details: dict[str, Any] = {
                "outer_line": loop.lineno,
                "outer_target": _loop_target(loop),
                "inner_loops": len(inner_loops),
                "has_equality_comparison": has_equality,
                "has_membership_test": has_membership,
                "complexity": _complexity_label(2),
            }
            if pattern:
                details["pattern"] = "duplicate_match"
                details["variables"] = pattern
            findings.append(
                Finding(
                    id=_finding_id(file_rel, "nested_loops", loop.lineno, function),
                    kind="nested_loops",
                    file=file_rel,
                    line=loop.lineno,
                    function=function,
                    severity="high" if (has_equality or has_membership) else "medium",
                    message=(
                        f"Nested loops in '{function or '<module>'}' give {_complexity_label(2)} behaviour"
                    ),
                    suggestion=(
                        "Replace the inner scan with set/dict lookups (or a Counter when duplicates matter)."
                    ),
                    confidence=0.9 if pattern else 0.7,
                    auto_optimizable=True,
                    details=details,
                )
            )

    # Membership tests inside loops.
    for loop, depth, _parent in loops:
        for child in _walk_no_defs(loop):
            if not isinstance(child, ast.Compare):
                continue
            for op, comparator in zip(child.ops, child.comparators):
                if not isinstance(op, (ast.In, ast.NotIn)):
                    continue
                container = _classify_container(comparator, assigned)
                if container in {"list_literal", "tuple_literal", "list_call", "sorted_call", "name_list", "name_tuple"}:
                    findings.append(
                        Finding(
                            id=_finding_id(file_rel, "membership_in_loop", child.lineno, function),
                            kind="membership_in_loop",
                            file=file_rel,
                            line=child.lineno,
                            function=function,
                            severity="high" if depth >= 2 else "medium",
                            message=(
                                f"'{ast.unparse(child)}' scans a linear container on every iteration"
                            ),
                            suggestion="Use a set or dict for membership tests.",
                            confidence=0.85,
                            auto_optimizable=True,
                            details={
                                "container": container,
                                "expression": ast.unparse(child),
                                "loop_depth": depth,
                            },
                        )
                    )

    # Repeated pure calls inside loops.
    for loop, _depth, _parent in loops:
        calls: dict[str, list[ast.Call]] = {}
        modified = {
            target.id
            for child in _walk_no_defs(loop)
            if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            for target in (
                child.targets
                if isinstance(child, ast.Assign)
                else [child.target]
            )
            if isinstance(target, ast.Name)
        }
        for child in _walk_no_defs(loop):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Name):
                continue
            if child.func.id not in PURE_BUILTINS:
                continue
            if not all(isinstance(arg, (ast.Name, ast.Constant)) for arg in child.args):
                continue
            arg_names = {arg.id for arg in child.args if isinstance(arg, ast.Name)}
            if arg_names & modified:
                continue
            key = ast.dump(child)
            calls.setdefault(key, []).append(child)
        for key, occurrences in calls.items():
            call = occurrences[0]
            findings.append(
                Finding(
                    id=_finding_id(file_rel, "repeated_computation", call.lineno, function),
                    kind="repeated_computation",
                    file=file_rel,
                    line=call.lineno,
                    function=function,
                    severity="low",
                    message=(
                        f"'{ast.unparse(call)}' is recomputed on every loop iteration"
                    ),
                    suggestion="Hoist the computation before the loop.",
                    confidence=0.6,
                    auto_optimizable=True,
                    details={"call": ast.unparse(call), "occurrences": len(occurrences)},
                )
            )

    # String concatenation in a loop (quadratic behaviour).
    string_vars = {
        child.targets[0].id
        for child in _walk_no_defs(node)
        if isinstance(child, ast.Assign)
        and len(child.targets) == 1
        and isinstance(child.targets[0], ast.Name)
        and isinstance(child.value, ast.Constant)
        and isinstance(child.value.value, str)
    }
    for loop, _depth, _parent in loops:
        for child in _walk_no_defs(loop):
            if not isinstance(child, ast.AugAssign):
                continue
            if not isinstance(child.op, ast.Add) or not isinstance(child.target, ast.Name):
                continue
            if child.target.id not in string_vars:
                continue
            findings.append(
                Finding(
                    id=_finding_id(file_rel, "string_concat_in_loop", child.lineno, function),
                    kind="string_concat_in_loop",
                    file=file_rel,
                    line=child.lineno,
                    function=function,
                    severity="high",
                    message=(
                        f"Repeated string concatenation into '{child.target.id}' inside a loop"
                    ),
                    suggestion="Collect parts in a list and use ''.join(parts) after the loop.",
                    confidence=0.85,
                    auto_optimizable=True,
                    details={"variable": child.target.id},
                )
            )

    # pop(0) / insert(0, ...) inside loops are O(n) per call.
    for loop, _depth, _parent in loops:
        for child in _walk_no_defs(loop):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                continue
            if child.func.attr == "pop" and child.args and isinstance(child.args[0], ast.Constant) and child.args[0].value == 0:
                findings.append(
                    Finding(
                        id=_finding_id(file_rel, "inefficient_data_structure", child.lineno, function),
                        kind="inefficient_data_structure",
                        file=file_rel,
                        line=child.lineno,
                        function=function,
                        severity="medium",
                        message="pop(0) shifts every remaining list element",
                        suggestion="Use collections.deque for O(1) pops from the left.",
                        confidence=0.9,
                        auto_optimizable=True,
                        details={"call": ast.unparse(child)},
                    )
                )
            if child.func.attr == "insert" and child.args and isinstance(child.args[0], ast.Constant) and child.args[0].value == 0:
                findings.append(
                    Finding(
                        id=_finding_id(file_rel, "inefficient_data_structure", child.lineno, function),
                        kind="inefficient_data_structure",
                        file=file_rel,
                        line=child.lineno,
                        function=function,
                        severity="medium",
                        message="insert(0, ...) shifts every remaining list element",
                        suggestion="Use collections.deque.appendleft instead.",
                        confidence=0.9,
                        auto_optimizable=True,
                        details={"call": ast.unparse(child)},
                    )
                )

    _scan_dead_code(
        node.body if hasattr(node, "body") else [],
        file_rel,
        function,
        findings,
        skip_defs=function is None,
    )


def _is_descendant_loop(outer: ast.AST, candidate: ast.AST) -> bool:
    for node in _walk_no_defs(outer):
        if node is candidate:
            return True
    return False


def _function_info(node: ast.FunctionDef | ast.AsyncFunctionDef, file_rel: str, qualname: str) -> FunctionInfo:
    loops = list(_walk_loops(node))
    max_depth = max((depth for _, depth, _ in loops), default=0)
    branches = sum(
        1
        for child in _walk_no_defs(node)
        if isinstance(child, (ast.If, ast.IfExp, ast.Match))
    )
    calls = sum(1 for child in _walk_no_defs(node) if isinstance(child, ast.Call))
    args = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return FunctionInfo(
        name=node.name,
        qualname=qualname,
        file=file_rel,
        lineno=node.lineno,
        end_lineno=node.end_lineno or node.lineno,
        args=args,
        returns=ast.unparse(node.returns) if node.returns else None,
        decorators=[ast.unparse(d) for d in node.decorator_list],
        is_async=isinstance(node, ast.AsyncFunctionDef),
        loops=len(loops),
        max_loop_depth=max_depth,
        branches=branches,
        calls=calls,
        lines=(node.end_lineno or node.lineno) - node.lineno + 1,
        complexity=_complexity_label(max_depth),
    )


def _collect_definitions(
    node: ast.AST,
    file_rel: str,
    functions: list[FunctionInfo],
    classes: list[ClassInfo],
    prefix: str = "",
) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            methods = [
                item.name
                for item in child.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append(
                ClassInfo(
                    name=child.name,
                    file=file_rel,
                    lineno=child.lineno,
                    end_lineno=child.end_lineno or child.lineno,
                    bases=[ast.unparse(b) for b in child.bases],
                    methods=methods,
                )
            )
            _collect_definitions(child, file_rel, functions, classes, prefix + child.name + ".")
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = prefix + child.name
            functions.append(_function_info(child, file_rel, qualname))
            _collect_definitions(child, file_rel, functions, classes, qualname + ".")
        else:
            _collect_definitions(child, file_rel, functions, classes, prefix)


def _collect_imports(node: ast.AST, file_rel: str) -> list[ImportInfo]:
    imports: list[ImportInfo] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                imports.append(
                    ImportInfo(
                        file=file_rel,
                        lineno=child.lineno,
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        level=0,
                    )
                )
        elif isinstance(child, ast.ImportFrom):
            imports.append(
                ImportInfo(
                    file=file_rel,
                    lineno=child.lineno,
                    module=child.module or "",
                    names=[alias.asname or alias.name for alias in child.names],
                    level=child.level or 0,
                )
            )
    return imports


def _detect_duplicate_functions(
    functions: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, FunctionInfo]],
    file_rel: str,
    findings: list[Finding],
) -> None:
    seen: dict[str, FunctionInfo] = {}
    for node, info in functions:
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        if len(body) < 3:
            continue
        try:
            normalized = ast.dump(ast.Module(body=body, type_ignores=[]))
        except Exception:
            continue
        if normalized in seen:
            original = seen[normalized]
            findings.append(
                Finding(
                    id=_finding_id(file_rel, "duplicate_code", info.lineno, info.qualname),
                    kind="duplicate_code",
                    file=file_rel,
                    line=info.lineno,
                    function=info.qualname,
                    severity="low",
                    message=f"Function '{info.qualname}' is identical to '{original.qualname}'",
                    suggestion="Extract the shared implementation into a helper.",
                    confidence=0.8,
                    auto_optimizable=False,
                    details={"duplicate_of": original.qualname, "duplicate_line": original.lineno},
                )
            )
        else:
            seen[normalized] = info


def analyze_file(path: Path, root: Path | None = None) -> FileAnalysis:
    """Analyze a single Python file."""
    path = Path(path)
    root = Path(root) if root else path.parent
    try:
        file_rel = str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        file_rel = path.name

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return FileAnalysis(
            file=file_rel,
            lines=0,
            functions=[],
            classes=[],
            imports=[],
            findings=[],
            parse_error=f"cannot read file: {exc}",
        )

    lines = source.count("\n") + (1 if source and not source.endswith("\n") else 0)
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return FileAnalysis(
            file=file_rel,
            lines=lines,
            functions=[],
            classes=[],
            imports=[],
            findings=[],
            parse_error=f"SyntaxError at line {exc.lineno}: {exc.msg}",
        )

    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []
    _collect_definitions(tree, file_rel, functions, classes)
    imports = _collect_imports(tree, file_rel)

    findings: list[Finding] = []
    for child in ast.walk(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _analyze_scope(child, file_rel, child.name, findings)
    _analyze_scope(tree, file_rel, None, findings)

    function_nodes = [
        (child, info)
        for child in ast.walk(tree)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        for info in functions
        if info.lineno == child.lineno and info.file == file_rel
    ]
    _detect_duplicate_functions(function_nodes, file_rel, findings)

    # Deduplicate findings that were reported from overlapping scopes.
    deduped: dict[str, Finding] = {}
    for finding in findings:
        deduped.setdefault(finding.id, finding)

    return FileAnalysis(
        file=file_rel,
        lines=lines,
        functions=functions,
        classes=classes,
        imports=imports,
        findings=list(deduped.values()),
    )


def analyze_python_source(source: str, filename: str = "<memory>") -> FileAnalysis:
    """Analyze source text without touching the filesystem (used by tools/tests)."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="codeopt_analyze_") as tmp:
        target = Path(tmp) / (Path(filename).name or "snippet.py")
        target.write_text(source, encoding="utf-8")
        result = analyze_file(target, Path(tmp))
        result.file = filename.replace("\\", "/")
        for finding in result.findings:
            finding.file = result.file
            finding.id = _finding_id(result.file, finding.kind, finding.line, finding.function)
        return result


def analyze_repository(root: Path, max_files: int = 1000) -> Analysis:
    """Analyze every Python file in a repository."""
    started = time.perf_counter()
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")

    files = iter_python_files(root, max_files=max_files)
    all_functions: list[FunctionInfo] = []
    all_classes: list[ClassInfo] = []
    all_imports: list[ImportInfo] = []
    all_findings: list[Finding] = []
    parse_errors: list[dict[str, Any]] = []
    total_lines = 0

    for path in files:
        result = analyze_file(path, root)
        total_lines += result.lines
        all_functions.extend(result.functions)
        all_classes.extend(result.classes)
        all_imports.extend(result.imports)
        all_findings.extend(result.findings)
        if result.parse_error:
            parse_errors.append({"file": result.file, "error": result.parse_error})

    findings_by_kind: dict[str, int] = {}
    findings_by_severity: dict[str, int] = {}
    for finding in all_findings:
        findings_by_kind[finding.kind] = findings_by_kind.get(finding.kind, 0) + 1
        findings_by_severity[finding.severity] = findings_by_severity.get(finding.severity, 0) + 1

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics = {
        "files": len(files),
        "lines": total_lines,
        "functions": len(all_functions),
        "classes": len(all_classes),
        "imports": len(all_imports),
        "loops": sum(info.loops for info in all_functions),
        "max_loop_depth": max((info.max_loop_depth for info in all_functions), default=0),
        "findings_by_kind": findings_by_kind,
        "findings_by_severity": findings_by_severity,
        "parse_errors": len(parse_errors),
        "duration_ms": duration_ms,
    }

    return Analysis(
        repo_path=str(root),
        files_analyzed=len(files),
        total_lines=total_lines,
        functions=all_functions,
        classes=all_classes,
        imports=all_imports,
        findings=all_findings,
        metrics=metrics,
        parse_errors=parse_errors,
        duration_ms=duration_ms,
    )


def is_test_file(file_rel: str) -> bool:
    """Return True for test modules.  The agent must never optimize its own tests."""
    parts = Path(file_rel).parts
    name = parts[-1].lower()
    if any(part.lower() in {"tests", "test"} for part in parts[:-1]):
        return True
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def rank_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Order findings by how safe/valuable they are to attempt."""
    return sorted(
        findings,
        key=lambda f: (
            SEVERITY_RANK.get(f.severity, 3),
            0 if f.auto_optimizable else 1,
            -f.confidence,
            f.file,
            f.line,
        ),
    )
