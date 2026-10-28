"""Behavioural tests for examples/inefficient.py.

These tests are the correctness oracle for the optimizer.  They deliberately
cover ordering, duplicates, and empty inputs so that unsafe set-based rewrites
are rejected.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inefficient import build_message, dead_branch, find_common, has_duplicates


def test_find_common_basic():
    assert find_common([1, 2, 3], [2, 3, 4]) == [2, 3]


def test_find_common_preserves_order_of_first_list():
    assert find_common([3, 1, 2], [1, 2, 3]) == [3, 1, 2]


def test_find_common_duplicates_in_second_list():
    assert find_common([1, 2], [2, 2]) == [2, 2]


def test_find_common_duplicates_in_first_list():
    assert find_common([1, 1, 2], [1]) == [1, 1]


def test_find_common_duplicates_in_both_lists():
    assert find_common([1, 1, 2], [1, 1]) == [1, 1, 1, 1]


def test_find_common_empty_inputs():
    assert find_common([], [1, 2]) == []
    assert find_common([1, 2], []) == []


def test_find_common_strings():
    assert find_common(["a", "b", "c"], ["c", "a"]) == ["a", "c"]


def test_has_duplicates():
    assert has_duplicates([1, 2, 3, 2]) is True
    assert has_duplicates([1, 2, 3]) is False
    assert has_duplicates([]) is False


def test_build_message():
    assert build_message(["a", "b"]) == "a,b,"
    assert build_message([]) == ""


def test_dead_branch():
    assert dead_branch(5) == 5
    assert dead_branch(-1) == 0
