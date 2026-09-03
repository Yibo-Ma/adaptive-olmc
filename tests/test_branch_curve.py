"""Pure-logic tests for evaluation/branch_curve.py (the same-state branching consumer).

No torch, no model, no GPU — exercises the best-action / oracle / switch math on
synthetic per-boundary records shaped exactly like online_compressor._record_branch
emits, so it also pins the --json schema contract.

Runnable two ways (the repo ships no pytest dependency):
    python tests/test_branch_curve.py          # plain script, prints OK / raises
    pytest tests/test_branch_curve.py          # if pytest is installed
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "evaluation"))
import branch_curve as bc  # noqa: E402


def _rec(phase, next_bits, curr_bits, next_base, curr_base, bytes_=400, toks=100):
    lrs = [0.0, 3e-5, 1e-4, 3e-4]
    return {
        "phase": phase, "ref_lr": 1e-4,
        "curr": {"tokens": toks, "bytes": bytes_, "bits_base": curr_base},
        "next": {"tokens": toks, "bytes": bytes_, "bits_base": next_base},
        "candidates": [
            {"lr": a, "next_bits": next_bits[a], "curr_bits": curr_bits[a],
             "nonfinite": False}
            for a in lrs
        ],
    }


def _records():
    """Three boundaries. b0: medium clearly best. b1: small best (medium close).
    b2: medium best by a hair over small (a near-tie)."""
    return [
        _rec(0, {0.0: 100.0, 3e-5: 90.0, 1e-4: 80.0, 3e-4: 95.0},
                {0.0: 200.0, 3e-5: 180.0, 1e-4: 150.0, 3e-4: 140.0}, 105.0, 210.0),
        _rec(1, {0.0: 100.0, 3e-5: 95.0, 1e-4: 96.0, 3e-4: 100.0},
                {0.0: 200.0, 3e-5: 185.0, 1e-4: 170.0, 3e-4: 160.0}, 98.0, 205.0),
        _rec(2, {0.0: 100.0, 3e-5: 100.0, 1e-4: 99.99, 3e-4: 110.0},
                {0.0: 200.0, 3e-5: 190.0, 1e-4: 175.0, 3e-4: 165.0}, 101.0, 205.0),
    ]


def _close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"


def test_per_boundary_best_and_gk():
    bs = bc._boundaries_from_records(_records())
    b0 = bs[0]
    best_lr, best_bits, second_bits, gap, near_tie = b0.best()
    assert best_lr == 1e-4 and best_bits == 80.0 and second_bits == 90.0
    _close(gap, 10.0)
    assert not near_tie
    _close(b0.g_k(1e-4), 20.0)          # skip(100) - medium(80)
    _close(b0.delta_in(1e-4), 50.0)     # skip curr(200) - medium curr(150)
    # b2 is a near-tie: medium 99.99 vs small 100.0
    assert bs[2].best()[4] is True


def test_best_fixed_and_oracle():
    bs = bc._boundaries_from_records(_records())
    totals = bc.action_totals(bs)
    _close(totals[0.0], 300.0)
    _close(totals[3e-5], 285.0)
    _close(totals[1e-4], 275.99)
    _close(totals[3e-4], 305.0)
    assert bc.best_fixed_action(bs) == 1e-4
    _close(bc.oracle_total(bs), 274.99)          # 80 + 95 + 99.99
    _close(bc.oracle_headroom(bs), 1.0 / 275.99)  # (275.99 - 274.99) / 275.99


def test_switches_and_near_tie_stability():
    bs = bc._boundaries_from_records(_records())
    raw = bc.best_action_seq(bs)
    assert raw == [1e-4, 3e-5, 1e-4]             # medium, small, medium
    _close(bc.switch_count(raw), 2)
    # stable: b2's near-tie holds the previous action (small), removing one switch
    stable = bc.best_action_seq(bs, stable=True)
    assert stable == [1e-4, 3e-5, 3e-5]
    _close(bc.switch_count(stable), 1)
    _close(bc.near_tie_fraction(bs), 1.0 / 3.0)
    tc = bc.transition_counts(stable)
    assert tc[(1e-4, 3e-5)] == 1 and tc[(3e-5, 3e-5)] == 1


def test_win_fractions_and_excess():
    bs = bc._boundaries_from_records(_records())
    wins = bc.win_fractions(bs)
    _close(wins[1e-4], 2.0 / 3.0)
    _close(wins[3e-5], 1.0 / 3.0)
    _close(wins[0.0], 0.0)
    ex = bc.per_action_excess(bs)
    _close(ex[1e-4], 1.0)                         # only loses 1 bit at b1
    _close(ex[0.0], 25.01)                        # 20 + 5 + 0.01
    _close(ex[3e-4], 30.01)                       # 15 + 5 + 10.01


def test_domain_assignment():
    bs = bc._boundaries_from_records(_records())   # each next.bytes = 400 -> cum 400/800/1200
    blocks = [{"dataset": "A", "byte_start": 0, "byte_end": 800},
              {"dataset": "B", "byte_start": 800, "byte_end": 2000}]
    doms = bc.assign_domains(bs, blocks)
    assert doms == ["A", "B", "B"]                 # cum 400->A, 800->B, 1200->B


def test_empty():
    assert bc.oracle_headroom([]) == 0.0
    assert bc.near_tie_fraction([]) == 0.0
    assert bc.action_totals([]) == {}


def main() -> int:
    tests = [test_per_boundary_best_and_gk, test_best_fixed_and_oracle,
             test_switches_and_near_tie_stability, test_win_fractions_and_excess,
             test_domain_assignment, test_empty]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK  {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
