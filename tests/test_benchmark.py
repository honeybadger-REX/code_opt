from __future__ import annotations

import textwrap

from app.benchmark import compare_functions, compare_sources, infer_params

SLOW_SOURCE = textwrap.dedent(
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

FAST_SOURCE = textwrap.dedent(
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


def test_compare_detects_improvement(tmp_path):
    optimized = tmp_path / "fast.py"
    optimized.write_text(FAST_SOURCE, encoding="utf-8")
    params, error = infer_params(FAST_SOURCE, "find_common")
    assert error is None

    result = compare_functions(
        repo_dir=tmp_path,
        baseline_source=SLOW_SOURCE,
        optimized_path=optimized,
        function_name="find_common",
        params=params,
        size=400,
        repeats=2,
    )
    assert result.available is True
    assert result.outputs_match is True
    assert result.improvement_pct is not None
    assert result.improvement_pct > 50
    assert result.before_seconds > result.after_seconds


def test_compare_detects_wrong_output(tmp_path):
    wrong = textwrap.dedent(
        """\
        def find_common(a: list[int], b: list[int]) -> list[int]:
            return []
        """
    )
    optimized = tmp_path / "wrong.py"
    optimized.write_text(wrong, encoding="utf-8")
    params, _ = infer_params(wrong, "find_common")

    result = compare_functions(
        repo_dir=tmp_path,
        baseline_source=SLOW_SOURCE,
        optimized_path=optimized,
        function_name="find_common",
        params=params,
        size=100,
        repeats=2,
    )
    assert result.available is True
    assert result.outputs_match is False


def test_mutating_function_is_rejected(tmp_path):
    source = textwrap.dedent(
        """\
        def sort_values(values: list[int]) -> list[int]:
            values.sort()
            return values
        """
    )
    optimized = tmp_path / "mutating.py"
    optimized.write_text(source, encoding="utf-8")
    params, _ = infer_params(source, "sort_values")

    result = compare_functions(
        repo_dir=tmp_path,
        baseline_source=source,
        optimized_path=optimized,
        function_name="sort_values",
        params=params,
        size=20,
        repeats=2,
    )
    assert result.available is False
    assert "mutates" in (result.reason or "")


def test_infer_params_requires_annotations():
    source = "def f(a, b):\n    return a + b\n"
    params, error = infer_params(source, "f")
    assert params == []
    assert error is not None
    assert "type annotation" in error


def test_infer_params_uses_literal_defaults():
    source = "def f(a, b=3):\n    return a + b\n"
    params, error = infer_params(source, "f")
    assert error is not None  # 'a' is still unannotated

    source = "def f(a: int, b=3):\n    return a + b\n"
    params, error = infer_params(source, "f")
    assert error is None
    assert [p["annotation"] for p in params] == ["int", "int"]


def test_compare_sources_helper(tmp_path):
    result = compare_sources(
        repo_dir=tmp_path,
        baseline_source=SLOW_SOURCE,
        optimized_source=FAST_SOURCE,
        function_name="find_common",
        params=infer_params(FAST_SOURCE, "find_common")[0],
        size=200,
        repeats=2,
    )
    assert result.available is True
    assert result.outputs_match is True
