"""Analyse the V1-A adjacency probe from eval_online --adjacency-probe results.

V1-A asks one question: does a single OSOA update help the TRUE adjacent future
significantly and materially more than a matched non-adjacent same-domain future?

    T(t,H) = G_adj(t,H) − G_ctrl(t,H)
    G_adj  = L(future | θ_t) − L(future | θ_t^+)      (update's help on the true successor)
    G_ctrl = mean over K matched controls of the same

Aggregates use ratio-of-sums (Σ T / Σ adj_pre), block bootstrap CIs, and a sign-flip
permutation null, then emit a PASS / SIGNIFICANT-BUT-SMALL / FAIL verdict against the
project-investment thresholds (T_pct ≥ 0.5%, T_share ≥ 20%, p < 0.05, CI lower > 0).

    eval_online.py ... --adjacency-probe --json enwik9.json
    python evaluation/adjacency_curve.py enwik9.json pile.json bean.json --out-dir figs

Pass one JSON per dataset; the cross-dataset direction check and the pooled verdict
use all of them.  Pure statistics are separated from I/O so tests/test_adjacency.py
can pin the schema and the math without torch (mirrors gk_curve / branch_curve).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

MAIN_H = 4                 # pre-registered main horizon (avoids horizon cherry-picking)
T_PCT_MIN = 0.5            # project-investment magnitude gate (% of pre-update cost)
T_SHARE_MIN = 0.20        # adjacency share of the true-future gain
P_MAX = 0.05


@dataclass
class Boundary:
    dataset: str
    domain: str
    k: int
    horizons: List[int]
    adj_pre: List[float]
    adj_post: List[float]
    ctrl_pre: List[float]      # mean over the K controls, cumulative per horizon
    ctrl_post: List[float]
    curr_pre: float
    curr_post: float
    controls: List[int]

    def _i(self, H: int) -> int:
        return self.horizons.index(H)

    def g_adj(self, H: int) -> float:
        i = self._i(H)
        return self.adj_pre[i] - self.adj_post[i]

    def g_ctrl(self, H: int) -> float:
        i = self._i(H)
        return self.ctrl_pre[i] - self.ctrl_post[i]

    def t(self, H: int) -> float:
        return self.g_adj(H) - self.g_ctrl(H)

    def adj_pre_bits(self, H: int) -> float:
        return self.adj_pre[self._i(H)]


def boundaries_from_records(records: List[Dict], dataset: str) -> List[Boundary]:
    out = []
    for r in records:
        out.append(Boundary(
            dataset=dataset, domain=str(r.get("domain", "stream")), k=r["boundary_k"],
            horizons=r["horizons"], adj_pre=r["adj_pre"], adj_post=r["adj_post"],
            ctrl_pre=r["ctrl_pre"], ctrl_post=r["ctrl_post"],
            curr_pre=r["curr_pre_bits"], curr_post=r["curr_post_bits"],
            controls=r.get("control_starts", []),
        ))
    return out


# ---------------------------------------------------------------------------
# Aggregate statistics (ratio-of-sums, as V1-A requires)
# ---------------------------------------------------------------------------

def agg(bs: Sequence[Boundary], H: int) -> Dict[str, float]:
    if not bs:
        return dict(n=0, T_pct=0.0, G_adj_pct=0.0, T_share=float("nan"),
                    t_pos_frac=0.0, mean_t=0.0, median_t=0.0, sum_t=0.0, sum_adj=0.0)
    St = sum(b.t(H) for b in bs)
    Sadj = sum(b.g_adj(H) for b in bs)
    Spre = sum(b.adj_pre_bits(H) for b in bs)
    ts = [b.t(H) for b in bs]
    return dict(
        n=len(bs),
        T_pct=100.0 * St / Spre if Spre else 0.0,
        G_adj_pct=100.0 * Sadj / Spre if Spre else 0.0,
        T_share=(St / Sadj) if Sadj > 0 else float("nan"),
        t_pos_frac=sum(1 for v in ts if v > 0) / len(ts),
        mean_t=St / len(ts), median_t=median(ts), sum_t=St, sum_adj=Sadj,
    )


def _t_pct(bs: Sequence[Boundary], H: int) -> float:
    Spre = sum(b.adj_pre_bits(H) for b in bs)
    return 100.0 * sum(b.t(H) for b in bs) / Spre if Spre else 0.0


def block_bootstrap_ci(bs: Sequence[Boundary], H: int, n_boot: int = 10000,
                       n_blocks: int = 10, seed: int = 0,
                       alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI on T_pct via contiguous-block resampling (respects autocorrelation;
    do NOT treat chunks as independent)."""
    if len(bs) < 2:
        return (float("nan"), float("nan"))
    nb = min(n_blocks, len(bs))
    edges = [round(i * len(bs) / nb) for i in range(nb + 1)]
    # Pre-aggregate each contiguous block's (Σt, Σadj_pre) so a resample is O(#blocks),
    # not O(#boundaries) — 10k bootstraps stay instant even on a full-stream run.
    block_sums = []
    for i in range(nb):
        blk = bs[edges[i]:edges[i + 1]]
        if not blk:
            continue
        block_sums.append((sum(b.t(H) for b in blk),
                           sum(b.adj_pre_bits(H) for b in blk)))
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        chosen = [rng.choice(block_sums) for _ in range(len(block_sums))]
        st = sum(c[0] for c in chosen)
        sp = sum(c[1] for c in chosen)
        stats.append(100.0 * st / sp if sp else 0.0)
    stats.sort()
    lo = stats[int(alpha / 2 * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return (lo, hi)


def permutation_p(bs: Sequence[Boundary], H: int, n_perm: int = 1000,
                  seed: int = 0) -> float:
    """One-sided sign-flip randomisation test for Σ T > 0.  Under the null (the true
    future is exchangeable with matched controls) each boundary's T is symmetric about
    0, so random sign flips give the null distribution of Σ T."""
    ts = [b.t(H) for b in bs]
    obs = sum(ts)
    rng = random.Random(seed)
    ge = 1                                       # +1 for the observed (conservative)
    for _ in range(n_perm):
        s = sum(v if rng.random() < 0.5 else -v for v in ts)
        if s >= obs:
            ge += 1
    return ge / (n_perm + 1)


def leave_one_block_min_tpct(bs: Sequence[Boundary], H: int, n_blocks: int = 10) -> float:
    """Smallest T_pct when dropping any one contiguous block — a proxy for 'not driven
    by a single region/file' when per-source labels are unavailable."""
    if len(bs) < 2:
        return _t_pct(bs, H)
    nb = min(n_blocks, len(bs))
    edges = [round(i * len(bs) / nb) for i in range(nb + 1)]
    worst = float("inf")
    for i in range(nb):
        kept = list(bs[:edges[i]]) + list(bs[edges[i + 1]:])
        if kept:
            worst = min(worst, _t_pct(kept, H))
    return worst


def verdict(pooled: List[Boundary], per_dataset_tpct: Dict[str, float],
            H: int = MAIN_H, seed: int = 0) -> Dict:
    """PASS / SIGNIFICANT-BUT-SMALL / FAIL per the V1-A rules."""
    a = agg(pooled, H)
    lo, hi = block_bootstrap_ci(pooled, H, seed=seed)
    p = permutation_p(pooled, H, seed=seed)
    lobo = leave_one_block_min_tpct(pooled, H)
    reasons = []

    directions_ok = bool(per_dataset_tpct) and all(v > 0 for v in per_dataset_tpct.values())
    significant = (a["T_pct"] > 0 and lo > 0 and p < P_MAX and a["G_adj_pct"] > 0)
    material = (a["T_pct"] >= T_PCT_MIN and
                (a["T_share"] >= T_SHARE_MIN if a["T_share"] == a["T_share"] else False))
    robust = directions_ok and lobo > 0

    if not significant:
        label = "FAIL"
        if a["T_pct"] <= 0:
            reasons.append("pooled T_pct <= 0")
        if not (lo > 0):
            reasons.append(f"bootstrap CI includes 0 ({lo:+.3f}, {hi:+.3f})")
        if not (p < P_MAX):
            reasons.append(f"permutation p={p:.3f} not < {P_MAX}")
        if not (a["G_adj_pct"] > 0):
            reasons.append("G_adj_pct <= 0 (update does not help true future)")
    elif not directions_ok:
        label = "FAIL"
        reasons.append("cross-dataset direction inconsistent: "
                       + ", ".join(f"{d}:{v:+.3f}%" for d, v in per_dataset_tpct.items()))
    elif not robust:
        label = "FAIL"
        reasons.append(f"driven by one region (leave-one-block min T_pct={lobo:+.3f}%)")
    elif not material:
        label = "SIGNIFICANT-BUT-SMALL"
        reasons.append(f"significant but T_pct={a['T_pct']:.3f}% < {T_PCT_MIN}% "
                       f"or T_share={a['T_share']:.3f} < {T_SHARE_MIN}")
    else:
        label = "PASS"
        reasons.append("significant, material and cross-source consistent")

    return dict(label=label, reasons=reasons, H=H, t_pct=a["T_pct"], g_adj_pct=a["G_adj_pct"],
                t_share=a["T_share"], t_pos_frac=a["t_pos_frac"], ci=[lo, hi], p=p,
                leave_one_block_min_tpct=lobo, n=a["n"])


# ---------------------------------------------------------------------------
# I/O + reporting
# ---------------------------------------------------------------------------

def load_run(path: str) -> Tuple[List[Dict], Dict, Dict, str]:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    rec = next((r.get("adjacency") for r in blob.get("results", [])
                if r.get("mode") == "online" and r.get("adjacency")), None)
    return rec, blob.get("args", {}), blob.get("config", {}), _dataset_of(blob.get("args", {}))


def _dataset_of(args: Dict) -> str:
    d = args.get("data", "")
    return os.path.basename(str(d).rstrip("/\\")) or "stream"


def write_per_boundary_csv(path: str, all_bs: List[Boundary]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "domain", "boundary_k", "horizon", "adj_pre_bits",
                    "adj_post_bits", "g_adj_bits", "ctrl_pre_bits_mean",
                    "ctrl_post_bits_mean", "g_ctrl_bits", "t_bits", "t_pct",
                    "curr_pre_bits", "curr_post_bits", "control_starts"])
        for b in all_bs:
            for H in b.horizons:
                i = b._i(H)
                adj_pre, ctrl_pre = b.adj_pre[i], b.ctrl_pre[i]
                w.writerow([b.dataset, b.domain, b.k, H, f"{adj_pre:.6f}",
                            f"{b.adj_post[i]:.6f}", f"{b.g_adj(H):.6f}", f"{ctrl_pre:.6f}",
                            f"{b.ctrl_post[i]:.6f}", f"{b.g_ctrl(H):.6f}", f"{b.t(H):.6f}",
                            f"{100.0 * b.t(H) / adj_pre if adj_pre else 0.0:.4f}",
                            f"{b.curr_pre:.6f}", f"{b.curr_post:.6f}",
                            " ".join(map(str, b.controls))])


def summarize(per_ds: Dict[str, List[Boundary]], pooled: List[Boundary],
              horizons: List[int], main_h: int) -> Dict:
    print(f"\n{'=' * 72}\n  V1-A adjacency-specific transfer   ({len(pooled)} boundaries, "
          f"{len(per_ds)} dataset(s))\n{'=' * 72}")
    print(f"  main horizon H={main_h}  ·  thresholds: T_pct>={T_PCT_MIN}%  "
          f"T_share>={T_SHARE_MIN}  p<{P_MAX}")
    # sanity: the update should reduce the current chunk's own NLL
    dropped = sum(1 for b in pooled if b.curr_post < b.curr_pre)
    print(f"  sanity: current-chunk NLL dropped after update on {dropped}/{len(pooled)} "
          f"boundaries")

    print("  " + "-" * 68)
    print(f"  {'H':>3} {'T_pct%':>9} {'G_adj%':>9} {'T_share':>9} {'T>0 frac':>9} {'mean_t':>10}")
    for H in horizons:
        a = agg(pooled, H)
        share = f"{a['T_share']:.3f}" if a["T_share"] == a["T_share"] else "  n/a"
        print(f"  {H:>3} {a['T_pct']:>9.3f} {a['G_adj_pct']:>9.3f} {share:>9} "
              f"{a['t_pos_frac']:>9.2f} {a['mean_t']:>10.2f}")

    print("  " + "-" * 68)
    print(f"  per-dataset T_pct(H={main_h}):")
    per_ds_tpct = {}
    for ds, bs in per_ds.items():
        per_ds_tpct[ds] = _t_pct(bs, main_h)
        print(f"      {ds:>26}  (n={len(bs):>4})  T_pct={per_ds_tpct[ds]:+.3f}%")

    v = verdict(pooled, per_ds_tpct, H=main_h)
    print("  " + "-" * 68)
    print(f"  bootstrap 95% CI (T_pct H={main_h}) = [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] %"
          f"   permutation p = {v['p']:.4f}")
    print(f"  leave-one-block min T_pct = {v['leave_one_block_min_tpct']:+.3f} %")
    print(f"\n  VERDICT: {v['label']}")
    for r in v["reasons"]:
        print(f"    - {r}")
    print("=" * 72)
    return dict(verdict=v, per_dataset_tpct=per_ds_tpct,
                per_horizon={H: agg(pooled, H) for H in horizons})


def plot(per_ds: Dict[str, List[Boundary]], pooled: List[Boundary],
         horizons: List[int], out_path: str, main_h: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("V1-A adjacency-specific transfer", fontweight="bold")

    # (1) T_pct vs horizon
    axes[0, 0].plot(horizons, [agg(pooled, H)["T_pct"] for H in horizons], "o-")
    axes[0, 0].axhline(0, color="k", lw=0.8)
    axes[0, 0].axhline(T_PCT_MIN, color="#d62728", ls="--", lw=0.8, label=f"{T_PCT_MIN}% gate")
    axes[0, 0].set_xlabel("horizon H"); axes[0, 0].set_ylabel("T_pct  [%]")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.25)

    # (2) T distribution at main H
    ts = [b.t(main_h) for b in pooled]
    axes[0, 1].hist(ts, bins=40, color="#4c78a8")
    axes[0, 1].axvline(0, color="k", lw=0.8)
    axes[0, 1].set_xlabel(f"T(t, H={main_h})  [bits]"); axes[0, 1].set_ylabel("boundaries")
    axes[0, 1].set_title(f"T>0 on {100 * agg(pooled, main_h)['t_pos_frac']:.0f}%", fontsize=9)

    # (3) observed Σt vs sign-flip null
    rng = random.Random(0)
    obs = sum(ts)
    null = [sum(v if rng.random() < 0.5 else -v for v in ts) for _ in range(2000)]
    axes[1, 0].hist(null, bins=40, color="#bbbbbb")
    axes[1, 0].axvline(obs, color="#d62728", lw=1.5, label="observed")
    axes[1, 0].set_xlabel("Σ T (permutation null)"); axes[1, 0].legend(fontsize=8)

    # (4) per-dataset T_pct
    ds = list(per_ds); vals = [_t_pct(per_ds[d], main_h) for d in ds]
    axes[1, 1].bar(range(len(ds)), vals,
                   color=["#2ca02c" if v > 0 else "#d62728" for v in vals])
    axes[1, 1].axhline(0, color="k", lw=0.8)
    axes[1, 1].set_xticks(range(len(ds))); axes[1, 1].set_xticklabels(ds, rotation=30, ha="right", fontsize=8)
    axes[1, 1].set_ylabel(f"T_pct(H={main_h})  [%]")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(
        description="Analyse the V1-A adjacency probe (one JSON per dataset)")
    p.add_argument("results", nargs="+", metavar="RESULT_JSON")
    p.add_argument("--out-dir", default=None,
                   help="directory for v1a_per_boundary.csv, v1a_summary.json and the plot")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    per_ds: Dict[str, List[Boundary]] = {}
    all_bs: List[Boundary] = []
    horizons: Optional[List[int]] = None
    for path in args.results:
        try:
            rec, run_args, config, ds = load_run(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [skip] {path}: {e}"); continue
        if not rec:
            print(f"  [skip] {path}: no online adjacency records "
                  f"(run eval_online with --adjacency-probe)"); continue
        bs = boundaries_from_records(rec, ds)
        per_ds.setdefault(ds, []).extend(bs)
        all_bs.extend(bs)
        horizons = horizons or bs[0].horizons
    if not all_bs:
        return 1

    main_h = MAIN_H if MAIN_H in horizons else max(horizons)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results[0]))
    summary = summarize(per_ds, all_bs, horizons, main_h)
    write_per_boundary_csv(os.path.join(out_dir, "v1a_per_boundary.csv"), all_bs)
    with open(os.path.join(out_dir, "v1a_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda o: str(o))
    print(f"  wrote {os.path.join(out_dir, 'v1a_per_boundary.csv')}")
    print(f"  wrote {os.path.join(out_dir, 'v1a_summary.json')}")
    if not args.no_plot:
        plot(per_ds, all_bs, horizons, os.path.join(out_dir, "v1a_adjacency.png"), main_h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
