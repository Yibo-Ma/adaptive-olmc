"""Decompose online-compression gain into cross-chunk CONTEXT vs weight ADAPTATION.

The coder normally starts every chunk from a bare BOS, so nothing crosses a chunk
boundary except through the adapted weights.  ``eval_online.py --ctx-tokens N``
lifts that restriction for BOTH modes, which turns the question "how much of OSOA's
gain is real adaptation and how much is it standing in for the truncated context?"
into a square, one per context length N:

    A = static, ctx 0     B = online, ctx 0      (the historical setup)
    C = static, ctx N     D = online, ctx N

    C vs A : what plain context is worth
    D vs C : what adaptation still adds once context is available
    overlap: (gain_B + gain_C) - gain_D  -> 0 means the two are complementary,
             large means they were capturing the same information

Several N may be swept at once (e.g. 0 / 2048 / 6144).  A and B come from the ctx-0
runs and are shared; every non-zero N gets its own square, so the ladder shows
whether adaptation's residual value (D - C) shrinks as context grows.

    eval_online.py --mode both --ctx-tokens 0    --json runs/enwik9_ctx0.json
    eval_online.py --mode both --ctx-tokens 2048 --json runs/enwik9_ctx2048.json
    eval_online.py --mode both --ctx-tokens 6144 --json runs/enwik9_ctx6144.json
    python evaluation/context_curve.py runs/*.json --out-dir figs

Cells are identified from each JSON's own args (mode x ctx_tokens) and grouped by
dataset.  Pure arithmetic is separated from I/O so tests can pin it (mirrors
gk_curve / branch_curve).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


def gain_pct(base_comp: float, comp: float) -> float:
    """Compressed-size reduction vs the A cell, in % (higher = better)."""
    return 100.0 * (base_comp - comp) / base_comp if base_comp else 0.0


def decompose(a_comp: float, b_comp: float, c_comp: float,
              d_comp: float) -> Dict[str, float]:
    """The three decisive numbers plus the overlap, from one square's four sizes."""
    g_b = gain_pct(a_comp, b_comp)                 # adaptation alone
    g_c = gain_pct(a_comp, c_comp)                 # context alone
    g_d = gain_pct(a_comp, d_comp)                 # both
    return dict(
        gain_B=g_b, gain_C=g_c, gain_D=g_d,
        d_vs_c=g_d - g_c,                          # adaptation's residual value given context
        d_vs_b=g_d - g_b,                          # context's residual value given adaptation
        additive=g_b + g_c,
        overlap=g_b + g_c - g_d,                   # 0 = complementary, large = redundant
    )


def squares(ladder: Dict[int, Dict[str, Dict]]) -> Dict[int, Dict[str, float]]:
    """One decomposition per non-zero ctx, sharing the ctx-0 A/B cells.

    ``ladder`` is {ctx_tokens: {mode: result_row}}.  Contexts whose square is
    incomplete are skipped rather than half-reported."""
    base = ladder.get(0, {})
    if "static" not in base or "online" not in base:
        return {}
    a, b = base["static"]["comp"], base["online"]["comp"]
    out = {}
    for ctx, cells in sorted(ladder.items()):
        if ctx == 0 or "static" not in cells or "online" not in cells:
            continue
        out[ctx] = decompose(a, b, cells["static"]["comp"], cells["online"]["comp"])
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_run(path: str) -> List[Tuple[str, int, str, Dict]]:
    """Return [(dataset, ctx_tokens, mode, result_row), ...] for one JSON."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    args = blob.get("args", {})
    ds = os.path.basename(str(args.get("data", "")).rstrip("/\\")) or "stream"
    ctx = int(args.get("ctx_tokens", 0) or 0)
    return [(ds, ctx, r["mode"], dict(r, source=path))
            for r in blob.get("results", [])
            if r.get("mode") in ("static", "online")]


def summarize(by_ds: Dict[str, Dict[int, Dict[str, Dict]]]) -> Dict:
    print(f"\n{'=' * 78}\n  Context vs adaptation decomposition  "
          f"({len(by_ds)} dataset(s))\n{'=' * 78}")
    summary: Dict[str, Dict] = {}
    for ds, ladder in sorted(by_ds.items()):
        print(f"\n  {ds}")
        base = ladder.get(0, {})
        a = base.get("static", {}).get("comp")
        print(f"  {'ctx':>7}{'static B':>12}{'online B':>12}"
              f"{'C vs A':>10}{'D vs A':>10}{'D vs C':>10}")
        print("  " + "-" * 61)
        for ctx, cells in sorted(ladder.items()):
            s = cells.get("static", {}).get("comp")
            o = cells.get("online", {}).get("comp")
            gc = f"{gain_pct(a, s):>9.2f}%" if (a and s) else f"{'-':>10}"
            gd = f"{gain_pct(a, o):>9.2f}%" if (a and o) else f"{'-':>10}"
            dvc = (f"{gain_pct(a, o) - gain_pct(a, s):>9.2f}%"
                   if (a and s and o) else f"{'-':>10}")
            print(f"  {ctx:>7}{(s if s else '-'):>12}{(o if o else '-'):>12}"
                  f"{gc}{gd}{dvc}")
        sq = squares(ladder)
        summary[ds] = sq
        if not sq:
            print("    (no complete square - need --mode both at ctx 0 and at some N)")
            continue
        for ctx, d in sq.items():
            print(f"\n    ctx {ctx}:  B (adapt alone) = {d['gain_B']:+.2f} %"
                  f"   C (context alone) = {d['gain_C']:+.2f} %"
                  f"   D (both) = {d['gain_D']:+.2f} %")
            print(f"              D-C adaptation GIVEN context = {d['d_vs_c']:+.2f} %"
                  f"   <- does OSOA survive?")
            print(f"              D-B context GIVEN adaptation = {d['d_vs_b']:+.2f} %"
                  f"   overlap (B+C-D) = {d['overlap']:+.2f} %")
    print("=" * 78)
    return summary


def plot(by_ds: Dict[str, Dict[int, Dict[str, Dict]]], out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    usable = {ds: squares(l) for ds, l in by_ds.items()}
    usable = {ds: sq for ds, sq in usable.items() if sq}
    if not usable:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cross-chunk context vs weight adaptation", fontweight="bold")
    for ds, sq in sorted(usable.items()):
        ctxs = sorted(sq)
        ax1.plot(ctxs, [sq[c]["gain_C"] for c in ctxs], "o--", label=f"{ds}  C (context)")
        ax1.plot(ctxs, [sq[c]["gain_D"] for c in ctxs], "o-", label=f"{ds}  D (both)")
        ax2.plot(ctxs, [sq[c]["d_vs_c"] for c in ctxs], "o-", label=ds)
        # adaptation alone is context-independent: one flat reference per dataset
        ax2.axhline(sq[ctxs[0]]["gain_B"], lw=0.6, ls=":", alpha=0.5)
    ax1.set_xlabel("ctx tokens"); ax1.set_ylabel("gain vs A  [%]")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.25)
    ax2.set_xlabel("ctx tokens")
    ax2.set_ylabel("D - C: adaptation given context  [%]")
    ax2.set_title("dotted = B (adaptation alone, ctx 0)", fontsize=8)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
        description="Decompose gain into context vs adaptation (A/B/C/D per ctx level)")
    p.add_argument("results", nargs="+", metavar="RESULT_JSON")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    by_ds: Dict[str, Dict[int, Dict[str, Dict]]] = {}
    for path in args.results:
        try:
            rows = load_run(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [skip] {path}: {e}")
            continue
        for ds, ctx, mode, row in rows:
            by_ds.setdefault(ds, {}).setdefault(ctx, {})[mode] = row
    if not by_ds:
        return 1

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results[0]))
    summary = summarize(by_ds)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "context_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  wrote {os.path.join(out_dir, 'context_summary.json')}")
    if not args.no_plot:
        plot(by_ds, os.path.join(out_dir, "context_decomposition.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
