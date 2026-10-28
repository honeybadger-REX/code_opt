from __future__ import annotations

import textwrap

import pytest

from app.git_manager import create_workspace
from app.tools import ToolError, ToolRegistry


@pytest.fixture
def registry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "readme.txt").write_text("hello world\n", encoding="utf-8")
    workspace = create_workspace(repo)
    tools = ToolRegistry(workspace.root)
    tools.attach_workspace(workspace)
    yield tools
    workspace.cleanup()


def test_read_and_write_file(registry):
    content = registry.read_file("a.py")
    assert "def add" in content["content"]
    registry.write_file("b.py", "x = 1\n")
    assert (registry.root / "b.py").read_text(encoding="utf-8") == "x = 1\n"


def test_path_traversal_is_blocked(registry, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ToolError):
        registry.read_file("../secret.txt")


def test_search_repository(registry):
    result = registry.search_repository("def add")
    assert result["matches"]
    assert result["matches"][0]["file"] == "a.py"
    assert result["matches"][0]["line"] == 1


def test_analyze_python(registry):
    analysis = registry.analyze_python("a.py")
    assert analysis["file"] == "a.py"
    assert analysis["functions"][0]["name"] == "add"


def test_git_diff_and_revert(registry):
    registry.write_file("a.py", "def add(a, b):\n    return b + a\n")
    diff = registry.git_diff(file="a.py")
    assert diff["empty"] is False
    assert "return b + a" in diff["diff"]
    registry.git_revert(file="a.py")
    restored = registry.read_file("a.py")["content"]
    assert "return a + b" in restored
    assert registry.git_diff(file="a.py")["empty"] is True


def test_generic_call_and_unknown_tool(registry):
    assert "def add" in registry.call("read_file", path="a.py")["content"]
    with pytest.raises(ToolError):
        registry.call("does_not_exist")


def test_benchmark_tool(registry):
    baseline = textwrap.dedent(
        """\
        def find_common(a: list[int], b: list[int]) -> list[int]:
            result = []
            for x in a:
                for y in b:
                    if x == y:
                        result.append(x)
            return result
        """
    )
    optimized = textwrap.dedent(
        """\
        def find_common(a: list[int], b: list[int]) -> list[int]:
            from collections import Counter

            counts = Counter(b)
            result = []
            for x in a:
                n = counts.get(x, 0)
                if n:
                    result.extend([x] * n)
            return result
        """
    )
    registry.write_file("bench_mod.py", optimized)
    result = registry.benchmark(
        function="find_common",
        baseline_source=baseline,
        optimized_path="bench_mod.py",
        params=[
            {"name": "a", "annotation": "list[int]"},
            {"name": "b", "annotation": "list[int]"},
        ],
        size=200,
        repeats=2,
    )
    assert result["available"] is True
    assert result["outputs_match"] is True
    assert result["improvement_pct"] > 0
