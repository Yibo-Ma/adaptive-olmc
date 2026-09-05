"""Pure-logic tests for the cross-chunk context feature.

Covers compression/online/context_window.ContextWindow (the rolling tail contract
that both endpoints must reproduce identically) and evaluation/context_curve's
A/B/C/D decomposition arithmetic.  No torch.

Runnable two ways (the repo ships no pytest dependency):
    python tests/test_context.py
    pytest tests/test_context.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from compression.online.context_window import ContextWindow  # noqa: E402
import context_curve as cc  # noqa: E402


def _close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b}"


# ---- ContextWindow -------------------------------------------------------

def test_disabled_is_exactly_the_old_behaviour():
    w = ContextWindow(0)
    assert not w.enabled
    assert w.tail() is None
    w.extend([[1, 2, 3], [4, 5]])
    assert w.tail() is None            # never conditions -> BOS path, byte-identical


def test_tail_is_none_before_any_chunk():
    w = ContextWindow(4)
    assert w.enabled and w.tail() is None   # first interval has no history yet


def test_tail_accumulates_in_coding_order():
    w = ContextWindow(10)
    w.extend([[1, 2], [3]])
    assert w.tail() == [1, 2, 3]
    w.extend([[4, 5]])
    assert w.tail() == [1, 2, 3, 4, 5]


def test_trims_to_window_keeping_the_most_recent():
    w = ContextWindow(3)
    w.extend([[1, 2, 3, 4, 5]])
    assert w.tail() == [3, 4, 5]
    w.extend([[6]])
    assert w.tail() == [4, 5, 6]


def test_generator_input_and_empty_chunks():
    w = ContextWindow(5)
    w.extend(ids for ids in ([1, 2], [], [3]))    # generators are accepted
    assert w.tail() == [1, 2, 3]


# ---- context_curve decomposition ----------------------------------------

def test_cell_mapping():
    assert cc.cell_of("static", 0) == "A"
    assert cc.cell_of("online", 0) == "B"
    assert cc.cell_of("static", 2048) == "C"
    assert cc.cell_of("online", 2048) == "D"
    assert cc.cell_of("both", 0) is None


def test_decompose_math():
    cells = {"A": {"comp": 1000.0}, "B": {"comp": 900.0},
             "C": {"comp": 950.0}, "D": {"comp": 880.0}}
    d = cc.decompose(cells)
    _close(d["gain_B"], 10.0)      # adaptation alone
    _close(d["gain_C"], 5.0)       # context alone
    _close(d["gain_D"], 12.0)      # both
    _close(d["d_vs_c"], 7.0)       # adaptation still worth 7pp once context is there
    _close(d["d_vs_b"], 2.0)
    _close(d["additive"], 15.0)
    _close(d["overlap"], 3.0)      # 15 expected vs 12 realised -> 3pp redundant


def test_decompose_requires_full_square():
    assert cc.decompose({"A": {"comp": 1.0}, "B": {"comp": 1.0}}) is None


def main() -> int:
    tests = [test_disabled_is_exactly_the_old_behaviour, test_tail_is_none_before_any_chunk,
             test_tail_accumulates_in_coding_order, test_trims_to_window_keeping_the_most_recent,
             test_generator_input_and_empty_chunks, test_cell_mapping,
             test_decompose_math, test_decompose_requires_full_square]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK  {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
