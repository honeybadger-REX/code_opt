"""Safe application and validation of proposed optimizations.

The AI proposes a replacement for a single function.  This module is the only
place that writes code, and it refuses anything that is not a syntactically
valid, signature-preserving replacement.
"""

from __future__ import annotations

import ast
import copy
import textwrap
from pathlib import Path

from .models import Finding, Proposal

ALLOWED_IMPORT_MODULES = {
    "bisect",
    "collections",
    "dataclasses",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}

BANNED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
}

BANNED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "multiprocessing",
    "threading",
    "pickle",
    "marshal",
    "importlib",
    "pathlib",
    "requests",
    "urllib",
    "http",
}


class OptimizationError(Exception):
    """Raised when a proposed change cannot be applied safely."""


def find_function_node(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if not matches:
        raise OptimizationError(f"Function '{name}' not found")
    if len(matches) > 1:
        raise OptimizationError(
            f"Function name '{name}' is ambiguous ({len(matches)} definitions); refusing to guess"
        )
    return matches[0]


def get_function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = find_function_node(tree, name)
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "".join(lines[start:end])


def replace_function(path: Path, name: str, new_source: str) -> str:
    """Replace one function definition in ``path`` and return the old source.

    The replacement keeps the original indentation (methods) and the original
    decorators.  The new text must contain exactly one top-level function with
    the requested name.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = find_function_node(tree, name)

    candidate = textwrap.dedent(new_source).strip("\n")
    if node.col_offset:
        candidate = textwrap.indent(candidate, " " * node.col_offset)

    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    old_source = "".join(lines[start:end])

    newline = "\r\n" if "\r\n" in source else "\n"
    replacement = candidate if candidate.endswith("\n") else candidate + newline
    updated = "".join(lines[:start]) + replacement + "".join(lines[end:])

    # Validate the whole file before writing anything.
    try:
        updated_tree = ast.parse(updated)
    except SyntaxError as exc:
        raise OptimizationError(f"Replacement produced invalid Python: {exc}") from exc
    new_node = find_function_node(updated_tree, name)
    if ast.dump(new_node.args) != ast.dump(node.args):
        raise OptimizationError("Replacement changed the function signature")
    if isinstance(new_node, ast.AsyncFunctionDef) != isinstance(node, ast.AsyncFunctionDef):
        raise OptimizationError("Replacement changed sync/async nature of the function")

    path.write_text(updated, encoding="utf-8")
    return old_source


def _banned_usage(node: ast.AST) -> list[str]:
    problems: list[str] = []

    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            if child.func.id in BANNED_CALLS:
                problems.append(f"banned call '{child.func.id}' at line {child.lineno}")
        if isinstance(child, ast.Attribute):
            root = child
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in BANNED_MODULES:
                problems.append(f"banned module usage '{root.id}.{child.attr}' at line {child.lineno}")
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORT_MODULES:
                    problems.append(f"import of '{alias.name}' is not allowed")
        if isinstance(child, ast.ImportFrom):
            if child.level:
                problems.append("relative imports are not allowed inside an optimized function")
            module = (child.module or "").split(".")[0]
            if module and module not in ALLOWED_IMPORT_MODULES:
                problems.append(f"import from '{child.module}' is not allowed")
    return problems


def validate_optimized_function(
    original_source: str,
    new_source: str,
    function_name: str,
) -> list[str]:
    """Return a list of problems with a proposed function replacement."""
    problems: list[str] = []
    try:
        original_tree = ast.parse(original_source)
        original_node = find_function_node(original_tree, function_name)
    except (SyntaxError, OptimizationError) as exc:
        return [f"original source problem: {exc}"]

    candidate = textwrap.dedent(new_source).strip()
    if not candidate:
        return ["optimized_function is empty"]

    try:
        new_tree = ast.parse(candidate)
    except SyntaxError as exc:
        return [f"optimized function is not valid Python: {exc}"]

    body = [stmt for stmt in new_tree.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ["optimized_function must contain exactly one function definition and nothing else"]
    new_node = body[0]
    if new_node.name != function_name:
        problems.append(f"function name changed: expected '{function_name}', got '{new_node.name}'")
    if new_node.decorator_list:
        problems.append("do not include decorators in the replacement; the original decorators are preserved")
    if ast.dump(new_node.args) != ast.dump(original_node.args):
        problems.append("function signature must stay exactly the same")
    if isinstance(new_node, ast.AsyncFunctionDef) != isinstance(original_node, ast.AsyncFunctionDef):
        problems.append("sync/async nature must stay the same")
    if not new_node.body:
        problems.append("function body is empty")
    if ast.dump(ast.Module(body=new_node.body, type_ignores=[])) == ast.dump(
        ast.Module(body=original_node.body, type_ignores=[])
    ):
        problems.append("optimized function is identical to the original")

    original_doc = ast.get_docstring(original_node, clean=False)
    new_doc = ast.get_docstring(new_node, clean=False)
    if original_doc != new_doc:
        problems.append("the function docstring must be preserved exactly")

    problems.extend(_banned_usage(new_node))
    return problems


def deterministic_proposal(
    finding: Finding,
    file_source: str,
    function_name: str,
) -> Proposal | None:
    """Produce a conservative rewrite for well-known patterns.

    This is a fallback used when the LLM is unavailable or returns something
    invalid.  Every deterministic rewrite still goes through tests and the
    differential benchmark before it can be kept.
    """
    if finding.kind == "nested_loops":
        return _deterministic_duplicate_match(finding, file_source, function_name)
    if finding.kind == "membership_in_loop":
        return _deterministic_membership(finding, file_source, function_name)
    if finding.kind == "string_concat_in_loop":
        return _deterministic_string_concat(finding, file_source, function_name)
    return None


def _deterministic_duplicate_match(
    finding: Finding,
    file_source: str,
    function_name: str,
) -> Proposal | None:
    from .analyzer import _detect_duplicate_match

    tree = ast.parse(file_source)
    try:
        node = find_function_node(tree, function_name)
    except OptimizationError:
        return None
    if not isinstance(node, ast.FunctionDef):
        return None
    pattern = _detect_duplicate_match(node)
    if not pattern:
        return None

    result_name = pattern["result"]
    body_source = f"""from collections import Counter

counts = Counter({pattern['inner_iter']})
{result_name} = []
for {pattern['outer_var']} in {pattern['outer_iter']}:
    n = counts.get({pattern['outer_var']}, 0)
    if n:
        {result_name}.extend([{pattern['outer_var']}] * n)
return {result_name}
"""
    new_body = ast.parse(body_source).body
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        new_body = [copy.deepcopy(node.body[0])] + new_body

    new_node = copy.deepcopy(node)
    new_node.body = new_body
    # replace_function() preserves the original decorators, so do not emit them here.
    new_node.decorator_list = []
    ast.fix_missing_locations(new_node)
    source = ast.unparse(new_node)

    return Proposal(
        finding_id=finding.id,
        file=finding.file,
        function=function_name,
        strategy="Counter-based multiplicity-preserving lookup",
        rationale=(
            "The nested loop compares every pair. Counting the elements of the inner "
            "iterable once and looking them up by key keeps the same order and the same "
            "duplicate multiplicity while removing the O(n*m) scan."
        ),
        risk="low",
        source=source,
        planner="deterministic",
        expected_gain="O(n+m) instead of O(n*m)",
    )


class _LiteralSetTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        for index, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            if isinstance(comparator, (ast.List, ast.Tuple)) and all(
                isinstance(elt, ast.Constant) for elt in comparator.elts
            ):
                node.comparators[index] = ast.Set(elts=copy.deepcopy(comparator.elts))
                self.changed = True
        return node


def _deterministic_membership(
    finding: Finding,
    file_source: str,
    function_name: str,
) -> Proposal | None:
    tree = ast.parse(file_source)
    try:
        node = find_function_node(tree, function_name)
    except OptimizationError:
        return None
    if not isinstance(node, ast.FunctionDef):
        return None
    if node.decorator_list:
        return None

    new_node = copy.deepcopy(node)
    transformer = _LiteralSetTransformer()
    new_node = transformer.visit(new_node)  # type: ignore[assignment]
    if not transformer.changed:
        return None
    ast.fix_missing_locations(new_node)

    return Proposal(
        finding_id=finding.id,
        file=finding.file,
        function=function_name,
        strategy="Use a set literal for constant membership tests",
        rationale=(
            "Membership in a list/tuple literal is linear; the constant elements are "
            "hashable, so a set literal gives the same answer with O(1) lookup."
        ),
        risk="low",
        source=ast.unparse(new_node),
        planner="deterministic",
        expected_gain="O(1) membership instead of O(k)",
    )


def _deterministic_string_concat(
    finding: Finding,
    file_source: str,
    function_name: str,
) -> Proposal | None:
    """Rewrite ``acc += expr`` in a loop as ``"".join(parts)``.

    Only the exact, common shape is handled:
    ``acc = "literal"`` immediately followed by a loop whose entire body is
    ``acc += expr``, with ``acc`` returned after the loop.
    """
    tree = ast.parse(file_source)
    try:
        node = find_function_node(tree, function_name)
    except OptimizationError:
        return None
    if not isinstance(node, ast.FunctionDef) or node.decorator_list:
        return None

    body = node.body
    for index, stmt in enumerate(body):
        if not isinstance(stmt, ast.For) or stmt.orelse or len(stmt.body) != 1:
            continue
        aug = stmt.body[0]
        if not (
            isinstance(aug, ast.AugAssign)
            and isinstance(aug.target, ast.Name)
            and isinstance(aug.op, ast.Add)
        ):
            continue
        accumulator = aug.target.id

        if index == 0:
            continue
        init = body[index - 1]
        if not (
            isinstance(init, ast.Assign)
            and len(init.targets) == 1
            and isinstance(init.targets[0], ast.Name)
            and init.targets[0].id == accumulator
            and isinstance(init.value, ast.Constant)
            and isinstance(init.value.value, str)
        ):
            continue
        if not any(
            isinstance(after, ast.Return)
            and isinstance(after.value, ast.Name)
            and after.value.id == accumulator
            for after in body[index + 1 :]
        ):
            continue

        parts_name = "parts"
        used_names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        while parts_name in used_names:
            parts_name += "_"

        new_loop = copy.deepcopy(stmt)
        new_loop.body = [
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=parts_name, ctx=ast.Load()),
                        attr="append",
                        ctx=ast.Load(),
                    ),
                    args=[aug.value],
                    keywords=[],
                )
            )
        ]
        parts_init = ast.Assign(
            targets=[ast.Name(id=parts_name, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
        join_call = ast.Call(
            func=ast.Attribute(
                value=ast.Constant(value=""),
                attr="join",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=parts_name, ctx=ast.Load())],
            keywords=[],
        )
        if init.value.value:
            final_value: ast.expr = ast.BinOp(
                left=ast.Constant(value=init.value.value),
                op=ast.Add(),
                right=join_call,
            )
        else:
            final_value = join_call
        final_assign = ast.Assign(
            targets=[ast.Name(id=accumulator, ctx=ast.Store())],
            value=final_value,
        )

        new_body = body[: index - 1] + [parts_init, new_loop, final_assign] + body[index + 1 :]
        new_node = copy.deepcopy(node)
        new_node.body = new_body
        new_node.decorator_list = []
        ast.fix_missing_locations(new_node)
        source = ast.unparse(new_node)

        return Proposal(
            finding_id=finding.id,
            file=finding.file,
            function=function_name,
            strategy="Accumulate parts in a list and join once",
            rationale=(
                "Repeated 'acc += expr' copies the whole accumulated string on every "
                "iteration (quadratic). Appending the parts and joining once is linear "
                "and produces the same string."
            ),
            risk="low",
            source=source,
            planner="deterministic",
            expected_gain="O(n) string construction instead of O(n^2)",
        )
    return None


def apply_proposal(path: Path, proposal: Proposal) -> str:
    """Apply a proposal and return the previous function source."""
    return replace_function(path, proposal.function, proposal.source)
