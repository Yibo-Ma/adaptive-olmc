"""Analyse the same-state branching probe from eval_online --branch-lrs results.

At each training boundary the encoder scored the NEXT interval under several
candidate learning rates from the identical parent state S_k (see
online_compressor._record_branch).  This consumer answers the two decisive
questions of "Dynamic Online Compression V1":

    1. does the best update strength differ across intervals?
    2. how much could a per-interval oracle beat the best fixed strength?

plus the switch structure that says whether that oracle is *trackable*.

    eval_online.py ... --branch-lrs 0,3e-5,1e-4,3e-4 --json run.json
    python evaluation/branch_curve.py run.json                 # one run
    python evaluation/branch_curve.py results/.../runs/*/result.json --out-dir figs

The authoritative per-candidate code length is the PDF-NLL (measure_interval_bits,
exact -Σ log2 p) the producer stored — framing-free, identical to the g_k metric,
so best-action / oracle comparisons carry no per-chunk coder overhead.

Pure aggregate math is separated from I/O so tests/test_branch_curve.py can pin the
schema and the statistics without torch (mirrors evaluation/gk_curve.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

NEAR_TIE_FRAC = 0.001          # 2nd-vs-1st gap < 0.1% of best -> a numerical tie, not a real switch


# ---------------------------------------------------------------------------
# Per-boundary model
# ---------------------------------------------------------------------------

@dataclass
class Boundary:
    phase: int
    next_bytes: int
    next_tokens: int
    next_base_bits: Optional[float]           # L(next|S_0) = Static
    curr_base_bits: Optional[float]
    lrs: List[float]                          # candidate order as recorded
    next_bits: Dict[float, float]             # lr -> L(next | S_k^lr)
    curr_bits: Dict[float, float]             # lr -> L(curr | S_k^lr)
    nonfinite: bool = False

    # -- per-boundary derived quantities -----------------------------------
    def best(self) -> Tuple[float, float, float, float, bool]:
        """(best_lr, best_bits, second_bits, gap_bits, near_tie) over candidates,
        ranked by next-interval code length (lower = better)."""
        ranked = sorted(self.lrs, key=lambda a: self.next_bits[a])
        best_lr = ranked[0]
        best_bits = self.next_bits[best_lr]
        second_bits = self.next_bits[ranked[1]] if len(ranked) > 1 else best_bits
        gap = second_bits - best_bits
        near_tie = len(ranked) > 1 and gap < NEAR_TIE_FRAC * abs(best_bits)
        return best_lr, best_bits, second_bits, gap, near_tie

    def skip_next(self) -> Optional[float]:
        return self.next_bits.get(0.0)

    def skip_curr(self) -> Optional[float]:
        return self.curr_bits.get(0.0)

    def g_k(self, lr: float) -> Optional[float]:
        """Transfer of candidate ``lr`` vs no-update (skip): L(next|S_k) - L(next|S_k^lr)."""
        s = self.skip_next()
        return None if s is None else s - self.next_bits[lr]

    def delta_in(self, lr: float) -> Optional[float]:
        """In-sample fit of candidate ``lr`` vs skip: L(curr|S_k) - L(curr|S_k^lr)."""
        s = self.skip_curr()
        return None if s is None else s - self.curr_bits[lr]


def _boundaries_from_records(records: List[Dict]) -> List[Boundary]:
    out: List[Boundary] = []
    for r in records:
        cands = r["candidates"]
        lrs = [float(c["lr"]) for c in cands]
        out.append(Boundary(
            phase=r["phase"],
            next_bytes=r["next"]["bytes"], next_tokens=r["next"]["tokens"],
            next_base_bits=r["next"].get("bits_base"),
            curr_base_bits=r["curr"].get("bits_base"),
            lrs=lrs,
            next_bits={float(c["lr"]): c["next_bits"] for c in cands},
            curr_bits={float(c["lr"]): c["curr_bits"] for c in cands},
            nonfinite=any(c.get("nonfinite") for c in cands),
        ))
    return out


# ---------------------------------------------------------------------------
# Aggregate statistics (the decisive numbers)
# ---------------------------------------------------------------------------

def action_totals(bs: List[Boundary]) -> Dict[float, float]:
    """lr -> Σ_k L(next | S_k^lr): the total code length of each *fixed* policy."""
    lrs = bs[0].lrs if bs else []
    return {a: sum(b.next_bits[a] for b in bs) for a in lrs}


def best_fixed_action(bs: List[Boundary]) -> float:
    totals = action_totals(bs)
    return min(totals, key=totals.get)


def oracle_total(bs: List[Boundary]) -> float:
    """Σ_k min_a L(next | S_k^lr): the per-boundary (local) counterfactual oracle.
    NOTE this is an UPPER BOUND on any deployable dynamic policy — each boundary
    branches from the same reference parent, so it is not a coherent trajectory."""
    return sum(b.best()[1] for b in bs)


def oracle_headroom(bs: List[Boundary]) -> float:
    """(best-fixed total − local-oracle total) / best-fixed total.  The dynamic
    selection space: how much a per-interval strength choice could save over the
    single best fixed strength (fraction of the best-fixed code length)."""
    if not bs:
        return 0.0
    tot = action_totals(bs)[best_fixed_action(bs)]
    return (tot - oracle_total(bs)) / tot if tot else 0.0


def best_action_seq(bs: List[Boundary], stable: bool = False) -> List[float]:
    """Best lr per boundary.  ``stable`` keeps the previous boundary's action on a
    near-tie (gap < 0.1% of best), so numerical noise is not counted as a switch."""
    seq: List[float] = []
    for b in bs:
        best_lr, best_bits, second_bits, gap, near_tie = b.best()
        if stable and near_tie and seq:
            prev = seq[-1]
            # keep prev only if it is itself within the near-tie band of the winner
            if abs(b.next_bits[prev] - best_bits) < NEAR_TIE_FRAC * abs(best_bits):
                best_lr = prev
        seq.append(best_lr)
    return seq


def win_fractions(bs: List[Boundary]) -> Dict[float, float]:
    seq = best_action_seq(bs)
    n = len(seq) or 1
    return {a: seq.count(a) / n for a in (bs[0].lrs if bs else [])}


def switch_count(seq: List[float]) -> int:
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])


def same_rate(seq: List[float]) -> float:
    if len(seq) < 2:
        return 0.0
    same = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
    return same / (len(seq) - 1)


def transition_counts(seq: List[float]) -> Dict[Tuple[float, float], int]:
    out: Dict[Tuple[float, float], int] = {}
    for i in range(1, len(seq)):
        key = (seq[i - 1], seq[i])
        out[key] = out.get(key, 0) + 1
    return out


def per_action_excess(bs: List[Boundary]) -> Dict[float, float]:
    """lr -> Σ_k (L(next|S_k^lr) − min_a L(next|S_k^a)): total bits each fixed action
    loses relative to the per-boundary best (0 = never beaten)."""
    lrs = bs[0].lrs if bs else []
    out = {a: 0.0 for a in lrs}
    for b in bs:
        best_bits = b.best()[1]
        for a in lrs:
            out[a] += b.next_bits[a] - best_bits
    return out


def near_tie_fraction(bs: List[Boundary]) -> float:
    if not bs:
        return 0.0
    return sum(1 for b in bs if b.best()[4]) / len(bs)


def mean_g_k(bs: List[Boundary], lr: float) -> Optional[float]:
    gs = [b.g_k(lr) for b in bs if b.g_k(lr) is not None]
    return sum(gs) / len(gs) if gs else None


def mean_delta_in(bs: List[Boundary], lr: float) -> Optional[float]:
    ds = [b.delta_in(lr) for b in bs if b.delta_in(lr) is not None]
    return sum(ds) / len(ds) if ds else None


# ---------------------------------------------------------------------------
# Optional per-domain analysis (needs the mixed-stream provenance sidecar)
# ---------------------------------------------------------------------------

def assign_domains(bs: List[Boundary], blocks: List[Dict]) -> List[Optional[str]]:
    """Label each boundary by the domain block its trained interval falls in, using
    cumulative curr-interval bytes vs the byte ranges in mix_manifest.json's blocks.
    Approximate (byte<->boundary), but enough for per-source action distributions."""
    domains: List[Optional[str]] = []
    cum = 0
    for b in bs:
        cum += b.next_bytes            # bytes coded through this boundary's interval
        dom = None
        for blk in blocks:
            if blk["byte_start"] <= cum < blk["byte_end"]:
                dom = blk["dataset"]
                break
        domains.append(dom if dom is not None else (blocks[-1]["dataset"] if blocks else None))
    return domains


def _load_manifest_blocks(data_path: Optional[str]) -> Optional[List[Dict]]:
    if not data_path:
        return None
    p = os.path.join(data_path, "mix_manifest.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("blocks")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# I/O + reporting
# ---------------------------------------------------------------------------

def _fmt_lr(lr: float) -> str:
    return "skip" if lr == 0.0 else f"{lr:g}"


def _config_label(args: Dict, config: Dict) -> str:
    parts = []
    if args.get("model"):
        parts.append(os.path.basename(str(args["model"]).rstrip("/\\")))
    if args.get("data"):
        parts.append(os.path.basename(str(args["data"]).rstrip("/\\")))
    r = config.get("lora_r")
    if r is not None:
        parts.append(f"r{r}")
    if args.get("chunk_size"):
        parts.append(f"chunk {args['chunk_size']}")
    if args.get("branch_ref_lr") is not None:
        parts.append(f"ref_lr {args['branch_ref_lr']:g}")
    return "  ·  ".join(parts)


def load_run(path: str) -> Tuple[List[Dict], Dict, Dict, str]:
    """Return (branch_records, args, config, run_tag) from an eval_online --json file."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    results = blob.get("results", [])
    branch = next((r.get("branch") for r in results
                   if r.get("mode") == "online" and r.get("branch")), None)
    return branch, blob.get("args", {}), blob.get("config", {}), _run_tag(path)


def _run_tag(path: str) -> str:
    """``runs/<id>/result.json`` -> ``<id>``; otherwise the file's own stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem == "result":
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if parent:
            return parent
    return stem


def summarize(bs: List[Boundary], label: str, tag: str, source: str,
              domains: Optional[List[Optional[str]]]) -> None:
    print(f"\n{'=' * 72}\n  {tag}   (text, {len(bs)} boundaries)\n{'=' * 72}")
    print(f"  config: {label}")
    print(f"  source: {source}")
    lrs = bs[0].lrs if bs else []
    nonfin = sum(1 for b in bs if b.nonfinite)
    if nonfin:
        print(f"  WARNING: {nonfin} boundaries had a non-finite candidate")

    # --- the two decisive numbers ---
    a_fixed = best_fixed_action(bs)
    headroom = oracle_headroom(bs)
    print("  " + "-" * 68)
    print(f"  best fixed action        = lr {_fmt_lr(a_fixed)}")
    print(f"  local-oracle headroom    = {headroom * 100:+.3f} %  "
          f"(dynamic vs best-fixed; UPPER BOUND, not a coherent trajectory)")

    # --- per-action picture ---
    wins = win_fractions(bs)
    totals = action_totals(bs)
    excess = per_action_excess(bs)
    base_tot = totals[a_fixed]
    print("  " + "-" * 68)
    print(f"  {'action':>8} {'win%':>7} {'excess_vs_best':>16} {'mean g_k%':>10} {'mean Δin%':>10}")
    for a in lrs:
        g = mean_g_k(bs, a)
        d = mean_delta_in(bs, a)
        # normalise g_k/Δ_in to the mean skip (pre-update) cost for a % reading
        pre = sum(b.skip_next() for b in bs if b.skip_next() is not None)
        pre_in = sum(b.skip_curr() for b in bs if b.skip_curr() is not None)
        gp = (mean_g_k(bs, a) * len(bs) / pre * 100) if (g is not None and pre) else float("nan")
        dp = (mean_delta_in(bs, a) * len(bs) / pre_in * 100) if (d is not None and pre_in) else float("nan")
        print(f"  {_fmt_lr(a):>8} {wins[a] * 100:>6.1f}% {excess[a] / base_tot * 100:>15.3f}% "
              f"{gp:>9.2f} {dp:>9.2f}")

    # --- switch structure (is the oracle trackable?) ---
    seq = best_action_seq(bs)
    seq_stable = best_action_seq(bs, stable=True)
    print("  " + "-" * 68)
    print(f"  best-action switches     = {switch_count(seq)}  "
          f"(stable, near-ties held = {switch_count(seq_stable)})")
    print(f"  adjacent same-action     = {same_rate(seq) * 100:.1f} %  "
          f"(stable {same_rate(seq_stable) * 100:.1f} %)")
    print(f"  near-tie boundaries      = {near_tie_fraction(bs) * 100:.1f} %  "
          f"(2nd within 0.1% of best)")
    tc = transition_counts(seq_stable)
    if tc:
        print("  transition matrix (stable, from -> to : count):")
        for a in lrs:
            row = "  ".join(f"{_fmt_lr(b)}:{tc.get((a, b), 0)}" for b in lrs)
            print(f"      {_fmt_lr(a):>6} -> {row}")

    # --- per-domain best-action distribution (mixed stream only) ---
    if domains and any(domains):
        print("  " + "-" * 68)
        uniq = []
        for d in domains:
            if d not in uniq:
                uniq.append(d)
        print("  best-action distribution by source:")
        for dom in uniq:
            idx = [i for i, d in enumerate(domains) if d == dom]
            n = len(idx) or 1
            dist = "  ".join(
                f"{_fmt_lr(a)}:{sum(1 for i in idx if seq[i] == a) / n * 100:.0f}%"
                for a in lrs)
            print(f"      {str(dom):>22}  (n={len(idx):>4})  {dist}")
    print("=" * 72)


def plot_run(bs: List[Boundary], domains: Optional[List[Optional[str]]],
             out_path: str, label: str, tag: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    lrs = bs[0].lrs
    x = list(range(len(bs)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[1, 1])
    fig.suptitle(f"Dynamic-strength branching — {tag}", fontweight="bold")

    # (top) best action per boundary
    seq = best_action_seq(bs, stable=True)
    y = [lrs.index(a) for a in seq]
    ax1.scatter(x, y, s=6, c=y, cmap="viridis")
    ax1.set_yticks(range(len(lrs)))
    ax1.set_yticklabels([_fmt_lr(a) for a in lrs])
    ax1.set_ylabel("best strength")
    ax1.set_title(label, fontsize=9)
    ax1.grid(alpha=0.25, lw=0.5)
    _shade_domains(ax1, domains)

    # (bottom) cumulative excess of each fixed action over the per-boundary oracle
    for a in lrs:
        cum, running = [], 0.0
        for b in bs:
            running += b.next_bits[a] - b.best()[1]
            cum.append(running)
        ax2.plot(x, cum, lw=1.2, label=f"lr {_fmt_lr(a)}")
    ax2.set_xlabel("training boundary (phase)")
    ax2.set_ylabel("cumulative excess\nover local oracle [bits]")
    ax2.legend(fontsize=8, ncol=len(lrs))
    ax2.grid(alpha=0.25, lw=0.5)
    _shade_domains(ax2, domains)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


def _shade_domains(ax, domains: Optional[List[Optional[str]]]) -> None:
    if not domains or not any(domains):
        return
    start = 0
    palette = ["#f4f4f4", "#e9f1fb", "#fdecea"]      # light, cycling per domain
    order, seen = [], set()
    for d in domains:
        if d not in seen:
            seen.add(d); order.append(d)
    for i in range(1, len(domains) + 1):
        if i == len(domains) or domains[i] != domains[start]:
            dom = domains[start]
            ax.axvspan(start, i - 1, color=palette[order.index(dom) % len(palette)], alpha=0.5, zorder=0)
            start = i


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(
        description="Analyse the same-state branching probe from eval_online --branch-lrs")
    p.add_argument("results", nargs="+", metavar="RESULT_JSON")
    p.add_argument("--out-dir", default=None,
                   help="directory for the .branch.png plots (default: next to each JSON)")
    p.add_argument("--no-plot", action="store_true", help="print summaries only")
    args = p.parse_args()

    n_ok = 0
    for path in args.results:
        try:
            records, run_args, config, tag = load_run(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [skip] {path}: {e}")
            continue
        if not records:
            print(f"  [skip] {path}: no online branch records "
                  f"(run eval_online with --branch-lrs)")
            continue
        bs = _boundaries_from_records(records)
        label = _config_label(run_args, config)
        blocks = _load_manifest_blocks(run_args.get("data"))
        domains = assign_domains(bs, blocks) if blocks else None
        summarize(bs, label, tag, path, domains)
        if not args.no_plot:
            out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
            plot_run(bs, domains, os.path.join(out_dir, f"{tag}.branch.png"), label, tag)
        n_ok += 1
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
