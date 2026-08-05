"""The cross-knob 'collapse' plot — the payoff of the gk_rank / gk_lr / gk_epochs study.

Tests the central claim: rank, lr and epochs are REDUNDANT controls of a single latent
'adaptation strength', measured by the in-sample gain Δ_in.  For each dataset it plots

      x = mean Δ_in  (realised adaptation strength)
      y = an outcome (transfer efficiency / delta_pct / stream g_k / regret)

with one series per knob.  If the rank-, lr- and epochs-generated series trace ONE curve,
adaptation strength IS one axis, independent of the knob — so a controller only has to
regulate that one thing.  Where the curve turns (efficiency drops, delta_pct peaks, regret
crosses 0) is the OVERFIT THRESHOLD; it should move with the dataset's transferability.

Pure consumer: reads what the runs already recorded (reuses evaluation/gk_curve.py's parsing),
never loads a model or touches a GPU.  matplotlib optional.

    python evaluation/collapse_curve.py \
        --rank   results/sweeps/text/<ts>_gk_rank \
        --lr     results/sweeps/text/<ts>_gk_lr \
        --epochs results/sweeps/text/<ts>_gk_epochs \
        --metric delta

Each --rank/--lr/--epochs is a sweep group dir (its runs/*/result.json are read and tagged
with that knob).  --epochs is optional (collapse shows with 2 knobs; epochs confirms the 3rd).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_curve as gk  # noqa: E402

# marker/colour per knob, and the knob's varying parameter in the run config
KNOB_STYLE = {
    "rank":   dict(color="#1f77b4", marker="o", param="lora_r"),
    "lr":     dict(color="#d62728", marker="s", param="lr"),
    "epochs": dict(color="#2ca02c", marker="^", param="epochs"),
}

# selectable y-metric: (axis label, per-run extractor, want-line-at-zero?)
METRICS = {
    "efficiency": ("transfer efficiency  g_k/Δ_in  [%]", lambda p: p.efficiency, False),
    "delta":      ("delta_pct  (online vs static)  [%]", lambda p: p.delta_pct, True),
    "gk":         ("stream g_k  [Δbpb]",               lambda p: p.stream_g_bpb, True),
    "regret":     ("stream regret  (online−base)  [%]", lambda p: p.regret_rel, True),
}


# ---------------------------------------------------------------------------
# Input layer  (one sweep run -> one RunPoint)
# ---------------------------------------------------------------------------

@dataclass
class RunPoint:
    dataset: str
    knob: str
    knob_value: float
    mean_din: float              # x-axis: mean Δ_in [%] = realised adaptation strength
    efficiency: float            # y options
    stream_g_bpb: float
    delta_pct: Optional[float]
    regret_rel: Optional[float]


def _delta_pct(results: List[dict]) -> Optional[float]:
    s = next((r for r in results if r.get("mode") == "static"), None)
    o = next((r for r in results if r.get("mode") == "online"), None)
    if s and o and s.get("bpsp"):
        return (s["bpsp"] - o["bpsp"]) / s["bpsp"] * 100.0
    return None


def load_point(path: str, knob: str) -> Optional[RunPoint]:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    results = blob.get("results") or []
    recs = next((r["gk"] for r in results if r.get("mode") == "online" and r.get("gk")), None)
    if not recs:
        return None
    bs = gk._boundaries_from_records(recs)
    params = gk._extract_params(blob)
    val = params.get(KNOB_STYLE[knob]["param"])
    return RunPoint(
        dataset=params.get("data") or "?",
        knob=knob,
        knob_value=float(val) if val is not None else float("nan"),
        mean_din=gk.mean_din(bs, "rel") * 100.0,
        efficiency=gk.transfer_efficiency(bs) * 100.0,
        stream_g_bpb=gk.stream_g(bs, "bpb"),
        delta_pct=_delta_pct(results),
        regret_rel=(gk.stream_regret(bs, "rel") * 100.0 if gk.has_regret(bs) else None),
    )


def load_group(group: str, knob: str) -> List[RunPoint]:
    files = sorted(glob.glob(os.path.join(group, "runs", "*", "result.json"))
                   or glob.glob(group))                 # allow a direct glob too
    pts = []
    for p in files:
        try:
            pt = load_point(p, knob)
        except (OSError, ValueError, KeyError):
            pt = None
        if pt is not None:
            pts.append(pt)
    return pts


def by_dataset(points: List[RunPoint]) -> Dict[str, List[RunPoint]]:
    out: Dict[str, List[RunPoint]] = {}
    for p in points:
        out.setdefault(p.dataset, []).append(p)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(points: List[RunPoint], metric: str) -> None:
    label, extract, _ = METRICS[metric]
    print(f"\n{'=' * 74}\n  COLLAPSE — {label}   (x = mean Δ_in %)\n{'=' * 74}")
    for ds, pts in by_dataset(points).items():
        print(f"\n  {ds}")
        for knob in ("rank", "lr", "epochs"):
            row = sorted((p for p in pts if p.knob == knob), key=lambda p: p.mean_din)
            if not row:
                continue
            cells = "  ".join(f"{KNOB_STYLE[knob]['param'][:4]}={p.knob_value:g}"
                              f"(Δin={p.mean_din:.1f},y={extract(p):+.2f})"
                              for p in row if extract(p) is not None)
            print(f"    {knob:<7} {cells}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Render (matplotlib optional)
# ---------------------------------------------------------------------------

def plot(points: List[RunPoint], out_path: str, metric: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plot skipped: matplotlib not installed (pip install matplotlib).")
        return False

    label, extract, zero_line = METRICS[metric]
    groups = by_dataset(points)
    datasets = list(groups)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.6),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for knob in ("rank", "lr", "epochs"):
            row = sorted((p for p in groups[ds] if p.knob == knob and extract(p) is not None),
                         key=lambda p: p.mean_din)
            if not row:
                continue
            st = KNOB_STYLE[knob]
            xs = [p.mean_din for p in row]
            ys = [extract(p) for p in row]
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], ms=6, lw=1.4,
                    label=knob, zorder=3)
            for p in row:                                # annotate each point's knob value
                ax.annotate(f"{p.knob_value:g}", (p.mean_din, extract(p)),
                            textcoords="offset points", xytext=(4, 4), fontsize=7,
                            color=st["color"])
        if zero_line:
            ax.axhline(0, color="0.5", lw=1.0, ls=":")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("mean Δ_in  [%]   (adaptation strength →)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel(label)
    fig.suptitle("Cross-knob collapse: rank / lr / epochs vs adaptation strength (Δ_in)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Cross-knob collapse plot (gk_rank/gk_lr/gk_epochs)")
    p.add_argument("--rank", help="sweep group dir for the rank axis (or a result.json glob)")
    p.add_argument("--lr", help="sweep group dir for the lr axis")
    p.add_argument("--epochs", help="sweep group dir for the epochs axis (optional)")
    p.add_argument("--metric", choices=list(METRICS), default="delta",
                   help="y-axis outcome (default: delta = online-vs-static gain)")
    p.add_argument("--out", default="collapse.png", help="output PNG path")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    points: List[RunPoint] = []
    for knob, group in (("rank", args.rank), ("lr", args.lr), ("epochs", args.epochs)):
        if group:
            pts = load_group(group, knob)
            print(f"  {knob:<7}: {len(pts)} runs from {group}")
            points += pts
    if not points:
        p.error("no runs loaded — pass at least --rank and --lr (each a sweep group dir)")

    knobs = {p_.knob for p_ in points}
    if len(knobs) < 2:
        print("  [warn] only one knob present — the collapse needs >=2 knobs to compare.")

    print_summary(points, args.metric)
    if not args.no_plot:
        plot(points, args.out, args.metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
