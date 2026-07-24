"""Plot per-chunk coding-rate curves from eval_online result JSONs.

Pure analysis + rendering: this tool reads what a run already recorded and never
loads a model, compresses anything, or touches a GPU.  Run experiments once with
``eval_online.py --json ...`` (or a whole ``experiments/sweep.py`` grid, which
writes one ``result.json`` per run), then plot whichever ones you care about.

    eval_online.py --json run.json   ──▶  run.json  {args, config, results[]}
                                              │  results[i].chunk_bits / chunk_lengths
    rate_curve.py run.json           ──▶  run.png

What the rate means
-------------------
Per chunk, rate = chunk_bits / chunk_length: the model's coding cost per coded
symbol (a *token* for text, a *byte* for the bGPT modalities).  Static and online
code the same chunks, so at every index both series share a denominator — their
difference, ratio, and cumulative gap are exact regardless of which symbol that
denominator counts.

The figure
----------
    top     per-chunk rate, static vs online, with LoRA update boundaries marked
    bottom  cumulative saved bytes, with the break-even chunk annotated

Usage
-----
    # one run (writes run.png next to run.json)
    python evaluation/rate_curve.py results/my_run.json

    # a whole sweep, smoothed, collected into one folder
    python evaluation/rate_curve.py results/sweeps/text/*/runs/*/result.json \
        --rolling 5 --out-dir results/figures

Requires ``--mode both`` runs for the comparison panels; a single-mode run plots
its one curve.  matplotlib is optional — without it, --no-plot still prints the
summary (including time-to-positive-gain).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# What one unit of ``chunk_lengths`` counts, per modality: text chunks carry token
# ids, bGPT chunks carry raw bytes (see the backends' ChunkUnit).
SYMBOL = {"text": "token", "image": "byte", "audio": "byte"}

MAX_BOUNDARY_LINES = 40      # beyond this, drawing every update just greys out the plot


# ---------------------------------------------------------------------------
# Pure series math
# ---------------------------------------------------------------------------

def rates(bits: List[int], lengths: List[int]) -> List[float]:
    return [b / L if L else 0.0 for b, L in zip(bits, lengths)]


def update_boundaries(n_chunks: int, train_interval: int) -> List[int]:
    """Chunk indices at which freshly trained weights first apply.

    Mirrors OnlineCompressor's schedule: training runs after every *full* interval
    and never after the partial tail, so boundaries land at train_interval,
    2*train_interval, …  A boundary at n_chunks is dropped — that update trains on
    the last group and no chunk is ever coded with it.
    """
    if train_interval < 1:
        raise ValueError("train_interval must be >= 1")
    return list(range(train_interval, n_chunks, train_interval))


def rolling_mean(values: List[float], window: int) -> List[float]:
    """Trailing rolling mean, partial at the start so the curve spans every chunk."""
    if window <= 1:
        return list(values)
    out: List[float] = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def cumulative_savings(static_bits: List[int], online_bits: List[int]) -> List[int]:
    """C(t) = bits static spent on chunks 0..t  −  bits online spent on the same."""
    out, running = [], 0
    for s, o in zip(static_bits, online_bits):
        running += s - o
        out.append(running)
    return out


def time_to_positive_gain(cumulative: List[int]) -> Optional[int]:
    """First chunk index where online is cumulatively ahead (None if never)."""
    return next((i for i, c in enumerate(cumulative) if c > 0), None)


# ---------------------------------------------------------------------------
# Input layer  (eval_online --json  ->  Run)
# ---------------------------------------------------------------------------

@dataclass
class Mode:
    """One mode's recorded series from a run."""
    bits: List[int]
    lengths: List[int]
    archive_bytes: int

    @property
    def rates(self) -> List[float]:
        return rates(self.bits, self.lengths)

    @property
    def mean_rate(self) -> float:
        """Stream rate = total payload bits / total symbols (not a mean of ratios)."""
        n = sum(self.lengths)
        return sum(self.bits) / n if n else 0.0


@dataclass
class Run:
    tag: str
    modality: str
    train_interval: int
    source: str
    modes: Dict[str, Mode] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return SYMBOL.get(self.modality, "symbol")

    @property
    def n_chunks(self) -> int:
        return len(next(iter(self.modes.values())).bits)

    @property
    def boundaries(self) -> List[int]:
        return (update_boundaries(self.n_chunks, self.train_interval)
                if "online" in self.modes else [])


def _default_tag(path: str) -> str:
    """``runs/<id>/result.json`` -> ``<id>``; otherwise the file's own stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem == "result":
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if parent:
            return parent
    return stem


def load_run(path: str) -> Run:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    results = blob.get("results")
    if not results:
        raise ValueError(f"{path}: no 'results' — not an eval_online --json file?")

    modes: Dict[str, Mode] = {}
    for r in results:
        if "chunk_bits" not in r:
            raise ValueError(
                f"{path}: run has no per-chunk record. It predates that being logged — "
                f"re-run eval_online.py with --json to capture it.")
        modes[r["mode"]] = Mode(bits=r["chunk_bits"], lengths=r["chunk_lengths"],
                                archive_bytes=r["comp"])

    # Static and online must have coded an identical chunk decomposition, or the
    # per-index comparison is meaningless. Catch a mismatch loudly.
    if "static" in modes and "online" in modes:
        if modes["static"].lengths != modes["online"].lengths:
            raise ValueError(
                f"{path}: static and online coded different chunk layouts "
                f"({len(modes['static'].lengths)} vs {len(modes['online'].lengths)} chunks). "
                f"A curve comparing them would be meaningless.")

    args, cfg = blob.get("args", {}), blob.get("config", {})
    return Run(
        tag=_default_tag(path),
        modality=args.get("modality", "text"),
        train_interval=cfg.get("train_interval", 1),
        source=path,
        modes=modes,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(run: Run) -> None:
    print(f"\n{'=' * 72}\n  {run.tag}   ({run.modality}, {run.n_chunks} chunks)\n{'=' * 72}")
    print(f"  source: {run.source}")
    print(f"  train_interval: {run.train_interval} | updates: {len(run.boundaries)}")
    for name, m in run.modes.items():
        payload = sum(m.bits) // 8
        print(f"  {name:<8} mean rate = {m.mean_rate:.4f} bits/{run.symbol}"
              f" | payload {payload} B + header {m.archive_bytes - payload} B"
              f" = archive {m.archive_bytes} B")

    if len(run.modes) == 2:
        s, o = run.modes["static"], run.modes["online"]
        cum = cumulative_savings(s.bits, o.bits)
        delta = s.mean_rate - o.mean_rate
        pct = (delta / s.mean_rate * 100) if s.mean_rate else 0.0
        ttpg = time_to_positive_gain(cum)
        print("  " + "-" * 68)
        print(f"  delta = {delta:+.4f} bits/{run.symbol}  ({pct:+.2f}%)"
              f" | total saved = {cum[-1] / 8:.0f} B")
        print(f"  time-to-positive-gain: "
              + (f"chunk {ttpg}" if ttpg is not None else "never (online never pulls ahead)"))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Render layer  (matplotlib is optional and imported lazily)
# ---------------------------------------------------------------------------

def plot_run(run: Run, out_path: str, rolling: int = 1) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")                     # headless / cluster-safe
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plot skipped: matplotlib not installed (pip install matplotlib).")
        return False

    has_both = len(run.modes) == 2
    x = list(range(run.n_chunks))

    fig, axes = plt.subplots(
        2 if has_both else 1, 1, figsize=(9, 6.5 if has_both else 3.8),
        sharex=True, gridspec_kw={"height_ratios": [2, 1]} if has_both else None,
    )
    ax0 = axes[0] if has_both else axes

    bounds = run.boundaries
    note = ""
    if bounds and len(bounds) <= MAX_BOUNDARY_LINES:
        for i, b in enumerate(bounds):
            ax0.axvline(b, color="0.85", lw=0.8, zorder=0,
                        label="LoRA update" if i == 0 else None)
    elif bounds:
        note = f"  (updates every {run.train_interval} chunks; lines omitted)"

    style = {"static": dict(color="#888888", lw=1.6, ls="--"),
             "online": dict(color="#1f77b4", lw=1.8)}
    for name in ("static", "online"):
        if name in run.modes:
            ax0.plot(x, rolling_mean(run.modes[name].rates, rolling),
                     label=name, **style[name])

    ax0.set_ylabel(f"bits / {run.symbol}" + (f"\n(rolling {rolling})" if rolling > 1 else ""))
    ax0.set_title(f"Per-chunk coding rate — {run.tag}{note}")
    ax0.legend(loc="upper right", frameon=False)
    ax0.grid(alpha=0.25, lw=0.5)

    if has_both:
        ax1 = axes[1]
        cum = cumulative_savings(run.modes["static"].bits, run.modes["online"].bits)
        cum_bytes = [c / 8 for c in cum]
        ax1.plot(x, cum_bytes, color="#2ca02c", lw=1.8)
        ax1.axhline(0, color="0.5", lw=1.0, ls=":")
        ax1.fill_between(x, 0, cum_bytes, where=[c >= 0 for c in cum_bytes],
                         color="#2ca02c", alpha=0.12, interpolate=True)
        ax1.fill_between(x, 0, cum_bytes, where=[c < 0 for c in cum_bytes],
                         color="#d62728", alpha=0.12, interpolate=True)
        ttpg = time_to_positive_gain(cum)
        if ttpg is not None:
            ax1.axvline(ttpg, color="#d62728", lw=1.0, ls="--")
            ax1.annotate(f"break-even @ {ttpg}", xy=(ttpg, 0), xytext=(4, 8),
                         textcoords="offset points", fontsize=8, color="#d62728")
        ax1.set_ylabel("cumulative\nsaved bytes")
        ax1.grid(alpha=0.25, lw=0.5)

    (axes[-1] if has_both else ax0).set_xlabel("chunk index")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=160)
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

    p = argparse.ArgumentParser(
        description="Plot per-chunk coding-rate curves from eval_online --json results")
    p.add_argument("results", nargs="+", metavar="RESULT_JSON",
                   help="one or more eval_online --json files (globs work)")
    p.add_argument("--out-dir", default=None,
                   help="where to write PNGs (default: beside each input JSON)")
    p.add_argument("--rolling", type=int, default=1,
                   help="trailing window for the plotted rate (default 1 = raw per-chunk)")
    p.add_argument("--no-plot", action="store_true", help="print summaries only")
    args = p.parse_args()

    if args.rolling < 1:
        p.error("--rolling must be >= 1")

    failed = 0
    for path in args.results:
        try:
            run = load_run(path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"  [skip] {path}: {exc}")
            failed += 1
            continue

        print_summary(run)
        if not args.no_plot:
            out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
            plot_run(run, os.path.join(out_dir, f"{run.tag}.png"), args.rolling)

    if failed:
        print(f"\n  {failed}/{len(args.results)} input(s) skipped")
    return 1 if failed == len(args.results) else 0


if __name__ == "__main__":
    sys.exit(main())
