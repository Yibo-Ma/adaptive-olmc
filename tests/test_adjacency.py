"""Pure-logic tests for the V1-A adjacency probe.

Covers compression/online/adjacency_plan.plan_adjacency (control selection) and
evaluation/adjacency_curve (ratio-of-sums stats, permutation, PASS/FAIL verdict),
on synthetic inputs shaped exactly like online_compressor._record_adjacency emits.
No torch. Runnable as a plain script or under pytest.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from compression.online.adjacency_plan import plan_adjacency  # noqa: E402
import adjacency_curve as ac  # noqa: E402

H = [1, 2, 4, 8]


def _close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"


def _rec(k, g_adj4, g_ctrl4, adj_pre4=400.0, curr_drop=True, domain="A"):
    """One boundary whose H=4 gains are g_adj4 / g_ctrl4 (other horizons scaled)."""
    adj_pre = [adj_pre4 / 4, adj_pre4 / 2, adj_pre4, adj_pre4 * 2]
    adj_post = [p - g_adj4 * (p / adj_pre4) for p in adj_pre]
    ctrl_post = [p - g_ctrl4 * (p / adj_pre4) for p in adj_pre]
    return {"phase": k, "boundary_k": k, "domain": domain, "horizons": H,
            "future_static_bits": adj_pre4, "control_starts": [k + 100, k + 200],
            "curr_pre_bits": 500.0, "curr_post_bits": 480.0 if curr_drop else 520.0,
            "adj_pre": adj_pre, "adj_post": adj_post,
            "ctrl_pre": list(adj_pre), "ctrl_post": ctrl_post}


# ---- plan_adjacency (control selection) ----------------------------------

def test_plan_structure():
    n = 40
    domains = ["A"] * n
    static = [10.0] * n                       # all windows same difficulty -> all match
    plan = plan_adjacency(domains, static, h_max=4, k_controls=4, n_target=3, seed=0)
    assert len(plan) == 3
    ks = [k for k, _ in plan]
    assert all(ks[i + 1] - ks[i] >= 5 for i in range(len(ks) - 1))     # spaced >= h_max+1
    for k, ctrls in plan:
        assert len(ctrls) == 4 and len(set(ctrls)) == 4
        for j in ctrls:
            assert 0 <= j <= n - 4                                     # valid window start
            assert abs(j - k) >= 8                                     # non-adjacent (>= 2*h_max)
            assert (j + 4 <= k) or (j >= k + 5)                        # no overlap with [k, k+4]


def test_plan_domain_and_scarcity():
    # two domains; controls must stay in the future window's domain
    domains = ["A"] * 20 + ["B"] * 20
    static = [10.0] * 40
    plan = plan_adjacency(domains, static, h_max=4, k_controls=4, n_target=10, seed=1)
    for k, ctrls in plan:
        dom = domains[k + 1]
        for j in ctrls:
            assert all(domains[j + i] == dom for i in range(4))
    # impossible request (need 100 controls) -> every boundary dropped
    assert plan_adjacency(["A"] * 40, static, 4, 100, 10, 0) == []


# ---- aggregate statistics (ratio-of-sums) --------------------------------

def test_agg_ratio_of_sums():
    bs = ac.boundaries_from_records([_rec(0, 20.0, 8.0), _rec(1, 16.0, 12.0)], "A")
    a = ac.agg(bs, 4)
    # b0: t=12, b1: t=4 ; g_adj 20+16=36 ; adj_pre 400+400=800
    _close(a["T_pct"], 100.0 * 16.0 / 800.0)      # 2.0
    _close(a["G_adj_pct"], 100.0 * 36.0 / 800.0)  # 4.5
    _close(a["T_share"], 16.0 / 36.0)             # 0.444...
    _close(a["t_pos_frac"], 1.0)
    _close(a["mean_t"], 8.0)


def test_permutation_and_ci_directions():
    bs = ac.boundaries_from_records([_rec(i, 20.0, 8.0) for i in range(8)], "A")
    p = ac.permutation_p(bs, 4, n_perm=2000, seed=0)
    assert 0.0 < p < 0.05                          # 8 all-positive t -> null rarely exceeds
    lo, hi = ac.block_bootstrap_ci(bs, 4, n_boot=500, seed=0)
    assert lo > 0                                  # constant positive T_pct


def test_verdict_pass():
    bs = ac.boundaries_from_records([_rec(i, 20.0, 8.0) for i in range(8)], "A")
    v = ac.verdict(bs, {"A": ac._t_pct(bs, 4)}, seed=0)
    assert v["label"] == "PASS", v["reasons"]


def test_verdict_significant_but_small():
    bs = ac.boundaries_from_records([_rec(i, 20.0, 19.0) for i in range(8)], "A")
    v = ac.verdict(bs, {"A": ac._t_pct(bs, 4)}, seed=0)   # t=1 -> tiny T_pct/share
    assert v["label"] == "SIGNIFICANT-BUT-SMALL", v["reasons"]


def test_verdict_fail_negative():
    bs = ac.boundaries_from_records([_rec(i, 8.0, 20.0) for i in range(8)], "A")
    v = ac.verdict(bs, {"A": ac._t_pct(bs, 4)}, seed=0)   # t<0
    assert v["label"] == "FAIL"


def test_verdict_fail_direction_inconsistent():
    # pooled is significant + material, but one dataset's T_pct is negative
    bs = ac.boundaries_from_records([_rec(i, 20.0, 8.0) for i in range(8)], "A")
    per = {"A": ac._t_pct(bs, 4), "B": -1.0}
    v = ac.verdict(bs, per, seed=0)
    assert v["label"] == "FAIL" and any("direction" in r for r in v["reasons"]), v


def main() -> int:
    tests = [test_plan_structure, test_plan_domain_and_scarcity, test_agg_ratio_of_sums,
             test_permutation_and_ci_directions, test_verdict_pass,
             test_verdict_significant_but_small, test_verdict_fail_negative,
             test_verdict_fail_direction_inconsistent]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK  {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
