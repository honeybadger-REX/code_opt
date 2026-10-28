"""Deliberately inefficient code used by the CodeOptimizer AI demo.

Every function here is correct; the point is that each one contains a real,
measurable performance problem that the agent can find and fix.
"""


def find_common(a: list[int], b: list[int]) -> list[int]:
    """Return the elements of ``a`` that also occur in ``b``.

    Each element is emitted once for every matching element in ``b``, in the
    order it appears in ``a``.  This nested loop is O(len(a) * len(b)).
    """
    result = []

    for x in a:
        for y in b:
            if x == y:
                result.append(x)

    return result


def has_duplicates(items: list[int]) -> bool:
    """Return True when ``items`` contains the same value more than once."""
    seen = []
    for item in items:
        if item in seen:
            return True
        seen.append(item)
    return False


def build_message(items: list[str]) -> str:
    """Join the items with commas (quadratic string concatenation)."""
    message = ""
    for item in items:
        message += item + ","
    return message


def dead_branch(value: int) -> int:
    """A tiny function with unreachable code after the return."""
    if value > 0:
        return value
    return 0
    value += 1
    return value
