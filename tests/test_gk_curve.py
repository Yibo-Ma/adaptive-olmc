"""Pure-logic tests for evaluation/gk_curve.py (the g_k / Δ_in consumer).

No torch, no model, no GPU — exercises the normalisation and aggregate math on
synthetic per-boundary records shaped exactly like online_compressor._record_gk
emits, so it also pins the --json schema contract.

Runnable two ways (the repo ships no pytest dependency):
    python tests/test_gk_curve.py          # plain script, prints OK / raises
    pytest tests/test_gk_curve.py          # if pytest is installed
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "evaluation"))
import gk_curve as gk  # noqa: E402


def _records():
    """Two boundaries: #0 a clean positive-transfer update, #1 the overfit
    signature (large Δ_in, negative g_k)."""
    return [
        {"phase": 0,
         "curr": {"tokens": 100, "bytes": 400, "bits_pre": 800.0, "bits_post": 760.0, "bits_base": 850.0},
         "next": {"tokens": 100, "bytes": 400, "bits_pre": 800.0, "bits_post": 720.0}},
        {"phase": 1,
         "curr": {"tokens": 100, "bytes": 400, "bits_pre": 800.0, "bits_post": 600.0, "bits_base": 700.0},
         "next": {"tokens": 100, "bytes": 400, "bits_pre": 800.0, "bits_post": 840.0}},
    ]


def _close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b}"


def test_per_boundary_g_and_din():
    b0, b1 = gk._boundaries_from_records(_records())
    # boundary 0: g_bits = 800-720 = 80 (helped); din_bits = 800-760 = 40
    _close(b0.g_bits, 80.0)
    _close(b0.din_bits, 40.0)
    _close(b0.g("rel"), 80.0 / 800.0)     # fraction of pre-update cost
    _close(b0.g("bpb"), 80.0 / 400.0)     # Δbpb
    _close(b0.g("bpt"), 80.0 / 100.0)     # bits/token
    _close(b0.din("bpb"), 40.0 / 400.0)
    # boundary 1: g_bits = 800-840 = -40 (hurt = negative transfer)
    _close(b1.g_bits, -40.0)
    _close(b1.din_bits, 200.0)            # memorised hard, but g_k < 0 -> overfit
    assert b1.g_bits < 0 < b1.din_bits


def test_aggregates():
    bs = gk._boundaries_from_records(_records())
    _close(gk.harmful_fraction(bs), 0.5)                 # 1 of 2 updates hurt
    _close(gk.mean_g(bs, "bpb"), (0.2 + (-0.1)) / 2)     # 0.05
    _close(gk.mean_g(bs, "rel"), (0.1 + (-0.05)) / 2)    # 0.025
    _close(gk.stream_g(bs, "bpb"), (80.0 - 40.0) / 800.0)   # size-weighted = 0.05
    _close(gk.stream_g(bs, "rel"), 40.0 / 1600.0)          # 0.025
    _close(gk.mean_din(bs, "rel"), (0.05 + 0.25) / 2)      # 0.15
    # transfer efficiency = total g_bits / total Δ_in bits = (80-40)/(40+200)
    _close(gk.transfer_efficiency(bs), 40.0 / 240.0)


def test_regret():
    bs = gk._boundaries_from_records(_records())
    b0, b1 = bs
    # regret = online cost − base cost:  b0 800-850=-50 (ahead),  b1 800-700=+100 (behind)
    _close(b0.regret_bits, -50.0)
    _close(b1.regret_bits, +100.0)
    assert gk.has_regret(bs)
    _close(gk.worse_than_base_fraction(bs), 0.5)            # only b1 > 0
    assert gk.cumulative_regret_bits(bs) == [-50.0, 50.0]   # ends behind static
    _close(gk.stream_regret(bs, "rel"), 50.0 / 1550.0)      # Σregret / Σbase
    _close(gk.stream_regret(bs, "bpb"), 50.0 / 800.0)
    _close(b0.regret("bpb"), -50.0 / 400.0)


def test_regret_absent_is_backward_compatible():
    # records without a base term -> regret unavailable, everything else still works
    recs = [{"phase": 0,
             "curr": {"tokens": 10, "bytes": 40, "bits_pre": 80.0, "bits_post": 76.0},
             "next": {"tokens": 10, "bytes": 40, "bits_pre": 80.0, "bits_post": 72.0}}]
    bs = gk._boundaries_from_records(recs)
    assert bs[0].curr_bits_base is None
    assert bs[0].regret_bits is None
    assert not gk.has_regret(bs)
    _close(bs[0].g_bits, 8.0)          # g_k still fine


def test_empty_and_zero_denominator():
    assert gk.harmful_fraction([]) == 0.0
    assert gk.mean_g([], "bpb") == 0.0
    assert gk.stream_g([], "rel") == 0.0
    z = gk.Boundary(phase=0, curr_tokens=0, curr_bytes=0, curr_bits_pre=0.0,
                    curr_bits_post=0.0, next_tokens=0, next_bytes=0,
                    next_bits_pre=0.0, next_bits_post=0.0)
    assert z.g("rel") == 0.0 and z.g("bpb") == 0.0 and z.din("bpt") == 0.0


def main() -> int:
    tests = [test_per_boundary_g_and_din, test_aggregates, test_regret,
             test_regret_absent_is_backward_compatible, test_empty_and_zero_denominator]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK  {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
