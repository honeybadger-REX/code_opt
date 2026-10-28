from __future__ import annotations

import textwrap

from app.analyzer import analyze_python_source, analyze_repository, is_test_file


def test_detects_nested_loops():
    source = textwrap.dedent(
        """
        def find_common(a, b):
            result = []
            for x in a:
                for y in b:
                    if x == y:
                        result.append(x)
            return result
        """
    )
    result = analyze_python_source(source, "mod.py")
    nested = [f for f in result.findings if f.kind == "nested_loops"]
    assert len(nested) == 1
    assert nested[0].function == "find_common"
    assert nested[0].details["pattern"] == "duplicate_match"
    assert nested[0].details["variables"]["outer_iter"] == "a"
    assert nested[0].severity == "high"


def test_detects_membership_on_list():
    source = textwrap.dedent(
        """
        def has_dupes(items):
            seen = []
            for item in items:
                if item in seen:
                    return True
                seen.append(item)
            return False
        """
    )
    result = analyze_python_source(source, "mod.py")
    kinds = {f.kind for f in result.findings}
    assert "membership_in_loop" in kinds
    finding = next(f for f in result.findings if f.kind == "membership_in_loop")
    assert finding.details["container"] == "name_list"


def test_membership_on_set_is_not_flagged():
    source = textwrap.dedent(
        """
        def has_dupes(items):
            seen = set()
            for item in items:
                if item in seen:
                    return True
                seen.add(item)
            return False
        """
    )
    result = analyze_python_source(source, "mod.py")
    assert not [f for f in result.findings if f.kind == "membership_in_loop"]


def test_detects_string_concatenation():
    source = textwrap.dedent(
        """
        def build(items):
            out = ""
            for item in items:
                out += item
            return out
        """
    )
    result = analyze_python_source(source, "mod.py")
    findings = [f for f in result.findings if f.kind == "string_concat_in_loop"]
    assert len(findings) == 1
    assert findings[0].details["variable"] == "out"


def test_detects_dead_code_once():
    source = textwrap.dedent(
        """
        def f(value):
            return value
            value += 1
            return value
        """
    )
    result = analyze_python_source(source, "mod.py")
    dead = [f for f in result.findings if f.kind == "dead_code"]
    assert len(dead) == 1
    assert dead[0].function == "f"


def test_detects_duplicate_functions():
    source = textwrap.dedent(
        """
        def first(items):
            total = 0
            for item in items:
                total += item
            return total

        def second(items):
            total = 0
            for item in items:
                total += item
            return total
        """
    )
    result = analyze_python_source(source, "mod.py")
    duplicates = [f for f in result.findings if f.kind == "duplicate_code"]
    assert len(duplicates) == 1
    assert duplicates[0].details["duplicate_of"] == "first"


def test_function_metadata_and_complexity():
    source = textwrap.dedent(
        """
        class Box:
            def total(self, values):
                total = 0
                for value in values:
                    for inner in values:
                        if inner == value:
                            total += 1
                return total
        """
    )
    result = analyze_python_source(source, "mod.py")
    assert len(result.classes) == 1
    assert result.classes[0].methods == ["total"]
    info = next(f for f in result.functions if f.name == "total")
    assert info.qualname == "Box.total"
    assert info.max_loop_depth == 2
    assert info.complexity == "O(n^2)"
    assert info.args == ["self", "values"]


def test_parse_errors_are_reported(tmp_path):
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    analysis = analyze_repository(tmp_path)
    assert analysis.parse_errors
    assert "SyntaxError" in analysis.parse_errors[0]["error"]


def test_excluded_directories(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "ignored.py").write_text("def f():\n    pass\n", encoding="utf-8")
    analysis = analyze_repository(tmp_path)
    assert analysis.files_analyzed == 1
    assert {f.file for f in analysis.functions} == set()


def test_example_repository_findings():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples"
    analysis = analyze_repository(root)
    nested = [f for f in analysis.findings if f.kind == "nested_loops"]
    assert len(nested) == 1
    assert nested[0].function == "find_common"
    assert any(f.kind == "membership_in_loop" for f in analysis.findings)
    assert any(f.kind == "string_concat_in_loop" for f in analysis.findings)


def test_is_test_file():
    assert is_test_file("tests/test_api.py")
    assert is_test_file("pkg/test_mod.py")
    assert is_test_file("pkg/mod_test.py")
    assert is_test_file("conftest.py")
    assert not is_test_file("pkg/mod.py")
