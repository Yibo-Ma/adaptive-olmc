"""Plot the prequential g_k / Δ_in instrument from eval_online result JSONs.

Pure analysis + rendering, like rate_curve.py: this tool reads what a run already
recorded with ``--measure-gk`` and never loads a model, compresses anything, or
touches a GPU.  Run once, plot whichever runs you care about.

    eval_online.py --mode online --measure-gk --json run.json
                                     │  results[online].gk[i] = {phase, curr, next}
    gk_curve.py run.json         ──▶ run.gk.png  (+ printed summary)

The instrument
--------------
At each training boundary k the encoder measures, for the interval it trained on
(``curr`` = I_k) and the next interval (``next`` = I_{k+1}), the model code length
before and after that one update (``bits_pre`` under S_k, ``bits_post`` under
S_{k+1}).  From those:

    g_k     = next.bits_pre − next.bits_post     bits saved on the FUTURE interval
    Δ_in(k) = curr.bits_pre − curr.bits_post     bits saved on the interval trained on

g_k > 0 the update helped; g_k < 0 it hurt (negative transfer / online overfitting,
isolated — chunk difficulty cancels because both terms code the same bytes).  The
overfit signature is Δ_in large with g_k ≤ 0 (memorised I_k, didn't transfer).

Normalisation (``--unit``): the run stores raw bits plus each interval's original
byte and token counts, so the tokenizer-independent field standard **Δbpb**
(bits/byte) and the scale-free **relative %** are both derivable; ``bits/token`` is
offered too but is tokenizer-dependent, so it is never the headline.

The figure
----------
    top     per-boundary g_k (green ≥0 / red <0) with Δ_in overlaid, zero line
    bottom  Δ_in vs g_k scatter — the overfit quadrant (Δ_in>0, g_k<0) is bottom-right

Usage
-----
    python evaluation/gk_curve.py results/run.json                 # PNG beside the JSON
    python evaluation/gk_curve.py results/sweeps/*/result.json \
        --unit bpb --out-dir results/figures                       # a whole sweep at once

matplotlib is optional — without it, ``--no-plot`` still prints the summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

# Per-boundary g_k normalisations.  "rel" is a fraction shown as a percent; the
# other two are already in their stated unit.  bits/token is tokenizer-dependent
# (hence not a default) — kept only as an internal cross-check.
UNIT_LABEL = {"rel": "% of next interval's pre-update cost",
              "bpb": "bits/byte", "bpt": "bits/token"}
UNIT_SCALE = {"rel": 100.0, "bpb": 1.0, "bpt": 1.0}


# ---------------------------------------------------------------------------
# Pure series math
# ---------------------------------------------------------------------------

@dataclass
class Boundary:
    """One training boundary's raw measurements (bits + sizes, both intervals)."""
    phase: int
    curr_tokens: int
    curr_bytes: int
    curr_bits_pre: float
    curr_bits_post: float
    next_tokens: int
    next_bytes: int
    next_bits_pre: float
    next_bits_post: float

    @property
    def g_bits(self) -> float:
        """Bits this update saved on the NEXT interval (g_k). <0 = it hurt."""
        return self.next_bits_pre - self.next_bits_post

    @property
    def din_bits(self) -> float:
        """Bits this update saved on the interval it trained on (Δ_in, in-sample)."""
        return self.curr_bits_pre - self.curr_bits_post

    def g(self, unit: str) -> float:
        """g_k normalised: fraction of pre-update cost ('rel'), Δbpb, or bits/token."""
        if unit == "bpb":
            return self.g_bits / self.next_bytes if self.next_bytes else 0.0
        if unit == "bpt":
            return self.g_bits / self.next_tokens if self.next_tokens else 0.0
        return self.g_bits / self.next_bits_pre if self.next_bits_pre else 0.0

    def din(self, unit: str) -> float:
        """Δ_in normalised the same way, over the interval it trained on."""
        if unit == "bpb":
            return self.din_bits / self.curr_bytes if self.curr_bytes else 0.0
        if unit == "bpt":
            return self.din_bits / self.curr_tokens if self.curr_tokens else 0.0
        return self.din_bits / self.curr_bits_pre if self.curr_bits_pre else 0.0


def harmful_fraction(bs: List[Boundary]) -> float:
    """Fraction of updates that made the next interval more expensive (g_k < 0).

    The number an aggregate delta hides: even a stream whose total delta is
    positive can carry many harmful updates washed out by the mean.
    """
    return sum(1 for b in bs if b.g_bits < 0) / len(bs) if bs else 0.0


def mean_g(bs: List[Boundary], unit: str) -> float:
    """Per-boundary mean of the normalised g_k (each boundary weighted equally)."""
    return sum(b.g(unit) for b in bs) / len(bs) if bs else 0.0


def mean_din(bs: List[Boundary], unit: str) -> float:
    return sum(b.din(unit) for b in bs) / len(bs) if bs else 0.0


def stream_g(bs: List[Boundary], unit: str) -> float:
    """Stream-level g_k: total bits saved over total denominator (size-weighted).

    = the transferability scalar for the whole stream in the chosen unit.
    """
    num = sum(b.g_bits for b in bs)
    if unit == "bpb":
        den = sum(b.next_bytes for b in bs)
    elif unit == "bpt":
        den = sum(b.next_tokens for b in bs)
    else:
        den = sum(b.next_bits_pre for b in bs)
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# Input layer  (eval_online --json  ->  GkRun)
# ---------------------------------------------------------------------------

@dataclass
class GkRun:
    tag: str
    modality: str
    source: str
    boundaries: List[Boundary]


def _boundaries_from_records(records: List[dict]) -> List[Boundary]:
    out: List[Boundary] = []
    for i, r in enumerate(records):
        c, n = r["curr"], r["next"]
        out.append(Boundary(
            phase=r.get("phase", i),
            curr_tokens=c["tokens"], curr_bytes=c["bytes"],
            curr_bits_pre=c["bits_pre"], curr_bits_post=c["bits_post"],
            next_tokens=n["tokens"], next_bytes=n["bytes"],
            next_bits_pre=n["bits_pre"], next_bits_post=n["bits_post"],
        ))
    return out


def _default_tag(path: str) -> str:
    """``runs/<id>/result.json`` -> ``<id>``; otherwise the file's own stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem == "result":
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if parent:
            return parent
    return stem


def load_gk_run(path: str) -> GkRun:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    results = blob.get("results")
    if not results:
        raise ValueError(f"{path}: no 'results' — not an eval_online --json file?")

    gk = next((r["gk"] for r in results if r.get("mode") == "online" and r.get("gk")), None)
    if not gk:
        raise ValueError(
            f"{path}: no g_k records. Re-run the online mode with --measure-gk "
            f"(only that flag records the prequential instrument).")

    args = blob.get("args", {})
    return GkRun(
        tag=_default_tag(path),
        modality=args.get("modality", "text"),
        source=path,
        boundaries=_boundaries_from_records(gk),
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(run: GkRun, unit: str) -> None:
    bs = run.boundaries
    scale, label = UNIT_SCALE[unit], UNIT_LABEL[unit]
    print(f"\n{'=' * 72}\n  {run.tag}   ({run.modality}, {len(bs)} boundaries)\n{'=' * 72}")
    print(f"  source: {run.source}")
    print(f"  harmful updates (g_k < 0): {harmful_fraction(bs) * 100:.1f}%"
          f"  ({sum(1 for b in bs if b.g_bits < 0)}/{len(bs)})")
    print("  " + "-" * 68)
    print(f"  g_k   mean (per-boundary) = {mean_g(bs, unit) * scale:+.4f} {label}")
    print(f"  g_k   stream (size-weighted, = transferability)"
          f" = {stream_g(bs, unit) * scale:+.4f} {label}")
    print(f"  Δ_in  mean (per-boundary) = {mean_din(bs, unit) * scale:+.4f} {label}")
    # bits/byte is the cross-model headline regardless of the chosen display unit.
    if unit != "bpb":
        print(f"  g_k   stream Δbpb          = {stream_g(bs, 'bpb'):+.4f} bits/byte")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Render layer  (matplotlib is optional and imported lazily)
# ---------------------------------------------------------------------------

def plot_run(run: GkRun, out_path: str, unit: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")                     # headless / cluster-safe
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plot skipped: matplotlib not installed (pip install matplotlib).")
        return False

    bs = run.boundaries
    scale, label = UNIT_SCALE[unit], UNIT_LABEL[unit]
    x = [b.phase for b in bs]
    g = [b.g(unit) * scale for b in bs]
    din = [b.din(unit) * scale for b in bs]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in g]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7),
                                   gridspec_kw={"height_ratios": [2, 1]})

    # top: per-boundary g_k (green helped / red hurt) with Δ_in overlaid
    ax0.bar(x, g, color=colors, width=0.8, zorder=2,
            label="g_k  (green ≥0 / red <0)")
    ax0.plot(x, din, color="#7f7f7f", lw=1.3, marker="o", ms=3, zorder=3,
             label="Δ_in (in-sample)")
    ax0.axhline(0, color="0.4", lw=1.0)
    ax0.set_ylabel(f"g_k  [{label}]")
    ax0.set_xlabel("training boundary (phase)")
    ax0.set_title(f"Prequential adaptation gain — {run.tag}"
                  f"   |   harmful {harmful_fraction(bs) * 100:.0f}%")
    ax0.legend(loc="upper right", frameon=False)
    ax0.grid(alpha=0.25, lw=0.5)

    # bottom: Δ_in vs g_k — overfit signature is the bottom-right quadrant
    ax1.scatter(din, g, c=colors, s=18, zorder=2)
    ax1.axhline(0, color="0.4", lw=1.0)
    ax1.axvline(0, color="0.4", lw=1.0)
    ax1.set_xlabel(f"Δ_in (memorisation)  [{label}]")
    ax1.set_ylabel(f"g_k (transfer)  [{label}]")
    if din and g and min(din) < 0 < max(din) and min(g) < 0:
        ax1.annotate("overfit\n(learns, doesn't transfer)",
                     xy=(max(din) * 0.5, min(g) * 0.7), fontsize=8, color="#d62728",
                     ha="center", va="center")
    ax1.grid(alpha=0.25, lw=0.5)

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
        description="Plot the prequential g_k / Δ_in instrument from eval_online --json results")
    p.add_argument("results", nargs="+", metavar="RESULT_JSON",
                   help="one or more eval_online --json files written with --measure-gk (globs work)")
    p.add_argument("--unit", choices=["rel", "bpb", "bpt"], default="rel",
                   help="g_k normalisation: rel = %% of pre-update cost (default), "
                        "bpb = bits/byte (Δbpb), bpt = bits/token (tokenizer-dependent)")
    p.add_argument("--out-dir", default=None,
                   help="where to write PNGs (default: beside each input JSON)")
    p.add_argument("--no-plot", action="store_true", help="print summaries only")
    args = p.parse_args()

    failed = 0
    for path in args.results:
        try:
            run = load_gk_run(path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"  [skip] {path}: {exc}")
            failed += 1
            continue

        print_summary(run, args.unit)
        if not args.no_plot:
            out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
            plot_run(run, os.path.join(out_dir, f"{run.tag}.gk.png"), args.unit)

    if failed:
        print(f"\n  {failed}/{len(args.results)} input(s) skipped")
    return 1 if failed == len(args.results) else 0


if __name__ == "__main__":
    sys.exit(main())
