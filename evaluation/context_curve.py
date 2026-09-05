"""Decompose online-compression gain into cross-chunk CONTEXT vs weight ADAPTATION.

The coder normally starts every chunk from a bare BOS, so nothing crosses a chunk
boundary except through the adapted weights.  ``eval_online.py --ctx-tokens N``
lifts that restriction for BOTH modes, which turns the question "how much of OSOA's
gain is real adaptation and how much is it standing in for the truncated context?"
into a clean 2x2:

    A = static, ctx 0     B = online, ctx 0      (the historical setup)
    C = static, ctx N     D = online, ctx N

    C vs A : what plain context is worth
    D vs C : what adaptation still adds once context is available
    overlap: (gain_B + gain_C) - gain_D  -> 0 means the two are complementary,
             large means they were capturing the same information

    eval_online.py --mode both --ctx-tokens 0    --json runs/enwik9_ctx0.json
    eval_online.py --mode both --ctx-tokens 2048 --json runs/enwik9_ctx2048.json
    python evaluation/context_curve.py runs/*.json --out-dir figs

Pass every run; cells are identified from each JSON's own args (mode x ctx_tokens)
and grouped by dataset.  Pure arithmetic is separated from I/O so tests can pin it
(mirrors gk_curve / branch_curve).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

CELLS = {(False, False): "A", (True, False): "B", (False, True): "C", (True, True): "D"}
CELL_DESC = {"A": "static  ctx0", "B": "online  ctx0",
             "C": "static  ctxN", "D": "online  ctxN"}


def cell_of(mode: str, ctx_tokens: int) -> Optional[str]:
    """A/B/C/D from (mode, ctx_tokens); None for modes outside the 2x2."""
    if mode not in ("static", "online"):
        return None
    return CELLS[(mode == "online", int(ctx_tokens or 0) > 0)]


def gain_pct(base_comp: float, comp: float) -> float:
    """Compressed-size reduction vs the A cell, in % (higher = better)."""
    return 100.0 * (base_comp - comp) / base_comp if base_comp else 0.0


def decompose(cells: Dict[str, Dict]) -> Optional[Dict[str, float]]:
    """The three decisive numbers plus the overlap, from one stream's A/B/C/D cells."""
    if not {"A", "B", "C", "D"} <= set(cells):
        return None
    a = cells["A"]["comp"]
    g_b = gain_pct(a, cells["B"]["comp"])          # adaptation alone
    g_c = gain_pct(a, cells["C"]["comp"])          # context alone
    g_d = gain_pct(a, cells["D"]["comp"])          # both
    return dict(
        gain_B=g_b, gain_C=g_c, gain_D=g_d,
        d_vs_c=g_d - g_c,                          # adaptation's residual value given context
        d_vs_b=g_d - g_b,                          # context's residual value given adaptation
        additive=g_b + g_c,
        overlap=g_b + g_c - g_d,                   # 0 = complementary, large = redundant
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_run(path: str) -> List[Tuple[str, str, Dict]]:
    """Return [(dataset, cell, result_row), ...] for every mode present in one JSON."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    args = blob.get("args", {})
    ds = os.path.basename(str(args.get("data", "")).rstrip("/\\")) or "stream"
    ctx = int(args.get("ctx_tokens", 0) or 0)
    out = []
    for r in blob.get("results", []):
        c = cell_of(r.get("mode", ""), ctx)
        if c:
            out.append((ds, c, dict(r, ctx_tokens=ctx, source=path)))
    return out


def summarize(by_ds: Dict[str, Dict[str, Dict]]) -> Dict:
    print(f"\n{'=' * 78}\n  Context vs adaptation decomposition  "
          f"({len(by_ds)} dataset(s))\n{'=' * 78}")
    summary = {}
    for ds, cells in by_ds.items():
        ctxn = next((c["ctx_tokens"] for c in cells.values() if c["ctx_tokens"]), 0)
        print(f"\n  {ds}   (ctx N = {ctxn} tokens)")
        print(f"  {'cell':<6}{'setup':<16}{'bytes':>12}{'bpb':>10}{'gain vs A':>12}")
        print("  " + "-" * 56)
        a = cells.get("A", {}).get("comp")
        for c in ("A", "B", "C", "D"):
            if c not in cells:
                print(f"  {c:<6}{CELL_DESC[c]:<16}{'(missing)':>12}")
                continue
            r = cells[c]
            g = gain_pct(a, r["comp"]) if a else float("nan")
            print(f"  {c:<6}{CELL_DESC[c]:<16}{r['comp']:>12}{r['bpb']:>10.4f}{g:>11.2f}%")
        d = decompose(cells)
        summary[ds] = d
        if d is None:
            print("  (incomplete 2x2 - run --mode both at ctx 0 and ctx N)")
            continue
        print("  " + "-" * 56)
        print(f"    C vs A  context alone                = {d['gain_C']:+.2f} %")
        print(f"    B vs A  adaptation alone             = {d['gain_B']:+.2f} %")
        print(f"    D vs A  both                         = {d['gain_D']:+.2f} %")
        print(f"    D vs C  adaptation GIVEN context     = {d['d_vs_c']:+.2f} %"
              f"   <- does OSOA survive?")
        print(f"    D vs B  context GIVEN adaptation     = {d['d_vs_b']:+.2f} %")
        print(f"    overlap (B+C-D)                      = {d['overlap']:+.2f} %"
              f"   (0 = complementary)")
    print("=" * 78)
    return summary


def plot(by_ds: Dict[str, Dict[str, Dict]], out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    dss = [d for d in by_ds if decompose(by_ds[d])]
    if not dss:
        return False
    dec = {d: decompose(by_ds[d]) for d in dss}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cross-chunk context vs weight adaptation", fontweight="bold")
    x = list(range(len(dss)))
    w = 0.27
    ax1.bar([i - w for i in x], [dec[d]["gain_C"] for d in dss], w, label="C: context only")
    ax1.bar(x, [dec[d]["gain_B"] for d in dss], w, label="B: adaptation only")
    ax1.bar([i + w for i in x], [dec[d]["gain_D"] for d in dss], w, label="D: both")
    ax1.set_xticks(x); ax1.set_xticklabels(dss, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("gain vs A  [%]"); ax1.legend(fontsize=8); ax1.grid(alpha=0.25, axis="y")

    ax2.bar([i - w / 2 for i in x], [dec[d]["d_vs_c"] for d in dss], w,
            label="D-C: adaptation given context", color="#2ca02c")
    ax2.bar([i + w / 2 for i in x], [dec[d]["overlap"] for d in dss], w,
            label="overlap (B+C-D)", color="#d62728")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(dss, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("[%]"); ax2.legend(fontsize=8); ax2.grid(alpha=0.25, axis="y")

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
        description="Decompose gain into context vs adaptation (A/B/C/D)")
    p.add_argument("results", nargs="+", metavar="RESULT_JSON")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    by_ds: Dict[str, Dict[str, Dict]] = {}
    for path in args.results:
        try:
            rows = load_run(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [skip] {path}: {e}")
            continue
        for ds, cell, row in rows:
            by_ds.setdefault(ds, {})[cell] = row
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
