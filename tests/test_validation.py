from __future__ import annotations

from pathlib import Path

from app.validation import (
    assess_function_safety,
    choose_validation_strategy,
    discover_repository,
    generate_cases,
    infer_validation_params,
    run_behavioral_validation,
)


def test_discovery_detects_pytest_repository(tmp_path):
    (tmp_path / "module.py").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    (tmp_path / "test_module.py").write_text(
        "from module import f\n\n\ndef test_f():\n    assert f(1) == 1\n",
        encoding="utf-8",
    )
    capabilities = discover_repository(tmp_path)
    assert capabilities.test_framework == "pytest"
    assert capabilities.test_files == 1
    assert capabilities.automated_execution_supported is True
    assert capabilities.test_file_paths == ["test_module.py"]


def test_discovery_detects_tests_directory(tmp_path):
    (tmp_path / "module.py").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_module.py").write_text(
        "from module import f\n\n\ndef test_f():\n    assert f(1) == 1\n",
        encoding="utf-8",
    )
    capabilities = discover_repository(tmp_path)
    assert capabilities.test_framework == "pytest"
    assert capabilities.test_directories == ["tests"]
    assert "tests/test_module.py" in capabilities.test_file_paths


def test_discovery_detects_unittest_without_forcing_pytest(tmp_path):
    (tmp_path / "module.py").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    (tmp_path / "test_module.py").write_text(
        "import unittest\n\n\n"
        "class ModuleTests(unittest.TestCase):\n"
        "    def test_f(self):\n"
        "        self.assertEqual(f(1), 1)\n",
        encoding="utf-8",
    )
    capabilities = discover_repository(tmp_path)
    assert capabilities.test_framework == "unittest"
    assert capabilities.unittest_detected is True


def test_discovery_reports_no_tests(tmp_path):
    (tmp_path / "module.py").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    capabilities = discover_repository(tmp_path)
    assert capabilities.test_framework == "none"
    assert capabilities.test_files == 0
    assert capabilities.automated_execution_supported is False


def test_generated_cases_include_edge_cases():
    source = "def f(items: list[int], n: int) -> int:\n    return len(items) + n\n"
    params, error = infer_validation_params(source, "f")
    assert error is None
    cases, case_error = generate_cases(params)
    assert case_error is None
    assert len(cases) >= 20
    assert {"args": [[], 0], "kwargs": {}} in cases
    assert any(case["args"] == [[1, 1, 2, 2], 0] for case in cases)
    assert any(case["args"][1] == -7 for case in cases)


def test_generated_cases_support_optional_and_keyword_only():
    source = (
        "from typing import Optional\n\n"
        "def f(items: Optional[list[str]] = None, *, limit: int = 2) -> int:\n"
        "    return len(items or []) + limit\n"
    )
    params, error = infer_validation_params(source, "f")
    assert error is None
    cases, case_error = generate_cases(params)
    assert case_error is None
    assert any(case["args"] == [None] for case in cases)
    assert any(case["kwargs"].get("limit") == 0 for case in cases)


def test_unsafe_function_is_rejected_without_execution(tmp_path):
    source = (
        "def send_email(path: str) -> int:\n"
        "    with open(path, 'w') as handle:\n"
        "        handle.write('x')\n"
        "    return 1\n"
    )
    (tmp_path / "unsafe.py").write_text(source, encoding="utf-8")
    assessment = assess_function_safety(source, "send_email")
    assert assessment["safe"] is False
    assert any("open" in reason or "write" in reason for reason in assessment["reasons"])


def test_unsafe_module_level_code_blocks_generated_validation(tmp_path):
    source = (
        "import urllib.request\n\n"
        "DATA = urllib.request.urlopen('https://example.com').read()\n\n"
        "def f(x: int) -> int:\n"
        "    return x\n"
    )
    assessment = assess_function_safety(source, "f")
    assert assessment["safe"] is False
    assert assessment["generated_possible"] is False
    assert any("module-level" in reason for reason in assessment["reasons"])


def test_behavioral_validation_matches_and_detects_mismatch(tmp_path):
    original = (
        "def find_common(a: list[int], b: list[int]) -> list[int]:\n"
        "    result = []\n"
        "    for x in a:\n"
        "        for y in b:\n"
        "            if x == y:\n"
        "                result.append(x)\n"
        "    return result\n"
    )
    good = (
        "from collections import Counter\n\n"
        "def find_common(a: list[int], b: list[int]) -> list[int]:\n"
        "    counts = Counter(b)\n"
        "    result = []\n"
        "    for x in a:\n"
        "        n = counts.get(x, 0)\n"
        "        if n:\n"
        "            result.extend([x] * n)\n"
        "    return result\n"
    )
    bad = "def find_common(a: list[int], b: list[int]) -> list[int]:\n    return []\n"

    params, error = infer_validation_params(original, "find_common")
    assert error is None
    cases, case_error = generate_cases(params)
    assert case_error is None

    good_path = tmp_path / "good.py"
    good_path.write_text(good, encoding="utf-8")
    good_result = run_behavioral_validation(tmp_path, original, good_path, "find_common", cases)
    assert good_result.ran is True
    assert good_result.passed is True
    assert good_result.cases_matched == good_result.cases_total

    bad_path = tmp_path / "bad.py"
    bad_path.write_text(bad, encoding="utf-8")
    bad_result = run_behavioral_validation(tmp_path, original, bad_path, "find_common", cases)
    assert bad_result.ran is True
    assert bad_result.passed is False
    assert bad_result.mismatches
    assert bad_result.mismatches[0]["reason"] == "return value differs"


def test_behavioral_validation_tolerates_float_rounding(tmp_path):
    original = "def f(x: float) -> float:\n    return x / 3.0\n"
    optimized = "def f(x: float) -> float:\n    return (x * 2.0) / 6.0\n"
    params, error = infer_validation_params(original, "f")
    assert error is None
    cases, case_error = generate_cases(params)
    assert case_error is None
    path = tmp_path / "optimized.py"
    path.write_text(optimized, encoding="utf-8")
    result = run_behavioral_validation(tmp_path, original, path, "f", cases)
    assert result.ran is True
    assert result.passed is True


def test_validation_strategy_prefers_existing_tests(tmp_path):
    (tmp_path / "module.py").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    (tmp_path / "test_module.py").write_text(
        "from module import f\n\n\ndef test_f():\n    assert f(1) == 1\n",
        encoding="utf-8",
    )
    capabilities = discover_repository(tmp_path)
    strategy = choose_validation_strategy(
        capabilities,
        {"safe_count": 0, "candidates_found": 0},
    )
    assert strategy["strategy"] == "EXISTING_TESTS"
    assert strategy["confidence"] == "HIGH"


def test_validation_strategy_static_only_when_nothing_is_safe(tmp_path):
    (tmp_path / "module.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    capabilities = discover_repository(tmp_path)
    strategy = choose_validation_strategy(
        capabilities,
        {"safe_count": 0, "candidates_found": 1},
    )
    assert strategy["strategy"] == "STATIC_ONLY"
    assert strategy["confidence"] == "LOW"
    assert "No tests and no safe" in strategy["reason"]
