"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.models import Finding


@pytest.fixture
def make_finding():
    def _make(
        kind: str,
        function: str = "f",
        file: str = "module.py",
        line: int = 1,
        **details,
    ) -> Finding:
        return Finding(
            id=f"{kind}-test",
            kind=kind,
            file=file,
            line=line,
            function=function,
            severity="high",
            message="test finding",
            suggestion="test suggestion",
            confidence=0.9,
            auto_optimizable=True,
            details=details,
        )

    return _make


@pytest.fixture
def sample_repo(tmp_path):
    """A small Python repository with a slow function and tests."""
    source = """\
def find_common(a: list[int], b: list[int]) -> list[int]:
    result = []
    for x in a:
        for y in b:
            if x == y:
                result.append(x)
    return result
"""
    tests = """\
from slow_mod import find_common


def test_basic():
    assert find_common([1, 2, 3], [2, 3, 4]) == [2, 3]


def test_duplicates():
    assert find_common([1, 1], [1, 1]) == [1, 1, 1, 1]
"""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "slow_mod.py").write_text(source, encoding="utf-8")
    (repo / "test_slow_mod.py").write_text(tests, encoding="utf-8")
    return repo, source


@pytest.fixture
def membership_repo(tmp_path):
    """A repository whose finding has no deterministic fallback.

    This is used to test pure revert behaviour: the model's proposal is the only
    candidate, so a failing test or a slower benchmark must revert the change.
    """
    source = """\
def has_duplicates(items: list[int]) -> bool:
    seen = []
    for item in items:
        if item in seen:
            return True
        seen.append(item)
    return False
"""
    tests = """\
from membership_mod import has_duplicates


def test_duplicates():
    assert has_duplicates([1, 2, 2]) is True
    assert has_duplicates([1, 2, 3]) is False
"""
    repo = tmp_path / "membership_repo"
    repo.mkdir()
    (repo / "membership_mod.py").write_text(source, encoding="utf-8")
    (repo / "test_membership_mod.py").write_text(tests, encoding="utf-8")
    return repo, source
