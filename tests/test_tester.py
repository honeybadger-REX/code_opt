from __future__ import annotations

import sys

import app.tester as tester
from app.tester import run_tests, run_unittest


def test_passing_suite(tmp_path):
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    result = run_tests(tmp_path, timeout=120)
    assert result.ran is True
    assert result.passed is True
    assert result.passed_count == 1
    assert result.failed_count == 0
    assert result.returncode == 0
    assert result.command[:3] == [sys.executable, "-m", "pytest"]
    assert result.cwd == str(tmp_path.resolve())
    assert result.python_executable == sys.executable


def test_inherited_pytest_options_are_ignored(tmp_path, monkeypatch):
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k definitely_not_a_test")
    monkeypatch.setenv("PYTEST_PLUGINS", "does_not_exist")

    result = run_tests(tmp_path, timeout=120)

    assert result.ran is True
    assert result.passed is True
    assert result.passed_count == 1
    assert "PYTEST_ADDOPTS" in result.sanitized_env
    assert "PYTEST_PLUGINS" in result.sanitized_env
    assert "deselected" not in result.output.lower()


def test_missing_pytest_is_reported(tmp_path, monkeypatch):
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tester.importlib.util, "find_spec", lambda name: None)

    result = run_tests(tmp_path, timeout=30)

    assert result.ran is False
    assert result.passed is False
    assert "pytest is not installed" in result.reason
    assert result.command[:3] == [sys.executable, "-m", "pytest"]
    assert result.cwd == str(tmp_path.resolve())


def test_failing_suite(tmp_path):
    (tmp_path / "test_bad.py").write_text(
        "def test_bad():\n    assert 1 + 1 == 3\n",
        encoding="utf-8",
    )
    result = run_tests(tmp_path, timeout=120)
    assert result.ran is True
    assert result.passed is False
    assert result.failed_count == 1
    assert result.returncode != 0
    assert "1 failed" in result.reason


def test_no_tests_collected(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
    result = run_tests(tmp_path, timeout=120)
    assert result.ran is False
    assert result.passed is False
    assert result.reason is not None
    assert "no tests" in result.reason.lower()


def test_unittest_suite_passes(tmp_path):
    (tmp_path / "test_calc.py").write_text(
        "import unittest\n\n\n"
        "class CalcTests(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(1 + 1, 2)\n",
        encoding="utf-8",
    )
    result = run_unittest(tmp_path, timeout=120)
    assert result.runner == "unittest"
    assert result.ran is True
    assert result.passed is True
    assert result.passed_count == 1
    assert result.returncode == 0
    assert result.command[:3] == [sys.executable, "-m", "unittest"]


def test_unittest_failure_is_reported(tmp_path):
    (tmp_path / "test_calc.py").write_text(
        "import unittest\n\n\n"
        "class CalcTests(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(1 + 1, 3)\n",
        encoding="utf-8",
    )
    result = run_unittest(tmp_path, timeout=120)
    assert result.ran is True
    assert result.passed is False
    assert result.failed_count == 1
    assert "unittest exited" in result.reason


def test_unittest_no_tests_is_not_a_pass(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
    result = run_unittest(tmp_path, timeout=120)
    assert result.ran is False
    assert result.passed is False
    assert "no unittest tests" in result.reason
