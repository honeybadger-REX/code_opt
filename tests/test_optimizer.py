from __future__ import annotations

import textwrap

import pytest

from app.optimizer import (
    OptimizationError,
    deterministic_proposal,
    replace_function,
    validate_optimized_function,
)


def load_function(source: str, name: str):
    namespace: dict = {}
    exec(compile(source, "<test>", "exec"), namespace)
    return namespace[name]


def test_replace_function_preserves_rest_of_file(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(
        textwrap.dedent(
            """\
            def first(x):
                return x + 1

            def second(x):
                return x * 2
            """
        ),
        encoding="utf-8",
    )
    old = replace_function(path, "first", "def first(x):\n    return x + 10")
    assert "x + 1" in old
    updated = path.read_text(encoding="utf-8")
    assert "return x + 10" in updated
    assert "return x * 2" in updated


def test_replace_function_handles_method_indentation(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(
        textwrap.dedent(
            """\
            class Box:
                def value(self, x):
                    return x
            """
        ),
        encoding="utf-8",
    )
    replace_function(path, "value", "def value(self, x):\n    return x + 1")
    updated = path.read_text(encoding="utf-8")
    assert "        return x + 1" in updated
    compile(updated, str(path), "exec")


def test_replace_function_rejects_signature_change(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("def f(a, b):\n    return a\n", encoding="utf-8")
    with pytest.raises(OptimizationError):
        replace_function(path, "f", "def f(a):\n    return a")
    assert path.read_text(encoding="utf-8") == "def f(a, b):\n    return a\n"


def test_validate_accepts_safe_replacement():
    original = "def f(a, b):\n    return a + b\n"
    candidate = "def f(a, b):\n    return sum((a, b))\n"
    assert validate_optimized_function(original, candidate, "f") == []


def test_validate_rejects_banned_call():
    original = "def f(a):\n    return a\n"
    candidate = "def f(a):\n    return eval('a')\n"
    problems = validate_optimized_function(original, candidate, "f")
    assert any("banned call 'eval'" in problem for problem in problems)


def test_validate_rejects_banned_import():
    original = "def f(a):\n    return a\n"
    candidate = "def f(a):\n    import os\n    return os.getcwd()\n"
    problems = validate_optimized_function(original, candidate, "f")
    assert any("not allowed" in problem for problem in problems)


def test_validate_rejects_decorators_and_renames():
    original = "def f(a):\n    return a\n"
    candidate = "@cache\ndef f(a):\n    return a\n"
    problems = validate_optimized_function(original, candidate, "f")
    assert any("decorator" in problem for problem in problems)

    renamed = "def g(a):\n    return a\n"
    problems = validate_optimized_function(original, renamed, "f")
    assert any("name changed" in problem for problem in problems)


def test_validate_rejects_identical_code():
    original = "def f(a):\n    return a\n"
    problems = validate_optimized_function(original, original, "f")
    assert any("identical" in problem for problem in problems)


def test_deterministic_duplicate_match_semantics(make_finding):
    source = textwrap.dedent(
        """\
        def find_common(a, b):
            result = []
            for x in a:
                for y in b:
                    if x == y:
                        result.append(x)
            return result
        """
    )
    finding = make_finding("nested_loops", "find_common")
    proposal = deterministic_proposal(finding, source, "find_common")
    assert proposal is not None
    assert "Counter" in proposal.source

    original = load_function(source, "find_common")
    optimized = load_function(proposal.source, "find_common")
    cases = [
        ([], []),
        ([1, 2, 3], [2, 3, 4]),
        ([1, 1, 2], [1]),
        ([1, 2], [2, 2]),
        ([1, 1, 2], [1, 1]),
        (["a", "b"], ["b"]),
    ]
    for a, b in cases:
        assert optimized(a, b) == original(a, b)


def test_deterministic_duplicate_match_preserves_docstring(make_finding):
    source = textwrap.dedent(
        '''\
        def find_common(a, b):
            """Find common values."""
            result = []
            for x in a:
                for y in b:
                    if x == y:
                        result.append(x)
            return result
        '''
    )
    finding = make_finding("nested_loops", "find_common")
    proposal = deterministic_proposal(finding, source, "find_common")
    assert proposal is not None
    assert '"""Find common values."""' in proposal.source
    assert load_function(proposal.source, "find_common").__doc__ == "Find common values."


def test_deterministic_membership_literal(make_finding):
    source = textwrap.dedent(
        """\
        def check(x):
            if x in [1, 2, 3]:
                return True
            return False
        """
    )
    finding = make_finding("membership_in_loop", "check")
    proposal = deterministic_proposal(finding, source, "check")
    assert proposal is not None
    assert "{" in proposal.source

    original = load_function(source, "check")
    optimized = load_function(proposal.source, "check")
    for value in range(6):
        assert optimized(value) == original(value)


def test_deterministic_string_concat(make_finding):
    source = textwrap.dedent(
        '''\
        def build_message(items):
            """Join the items with commas."""
            message = ""
            for item in items:
                message += item + ","
            return message
        '''
    )
    finding = make_finding("string_concat_in_loop", "build_message")
    proposal = deterministic_proposal(finding, source, "build_message")
    assert proposal is not None
    assert "join" in proposal.source
    assert '"""Join the items with commas."""' in proposal.source

    original = load_function(source, "build_message")
    optimized = load_function(proposal.source, "build_message")
    for items in ([], ["a"], ["a", "b"], ["x", "x", "y"]):
        assert optimized(items) == original(items)


def test_deterministic_string_concat_with_prefix(make_finding):
    source = textwrap.dedent(
        """\
        def build(items):
            out = "prefix:"
            for item in items:
                out += item
            return out
        """
    )
    finding = make_finding("string_concat_in_loop", "build")
    proposal = deterministic_proposal(finding, source, "build")
    assert proposal is not None
    original = load_function(source, "build")
    optimized = load_function(proposal.source, "build")
    for items in ([], ["a"], ["a", "b"]):
        assert optimized(items) == original(items)


def test_deterministic_returns_none_for_unknown_pattern(make_finding):
    source = "def f(x):\n    return x\n"
    finding = make_finding("nested_loops", "f")
    assert deterministic_proposal(finding, source, "f") is None
