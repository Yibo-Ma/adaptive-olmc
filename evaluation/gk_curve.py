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
UNIT_LABEL = {"rel": "% of pre-update cost",
              "bpb": "bits/byte", "bpt": "bits/token"}
UNIT_SCALE = {"rel": 100.0, "bpb": 1.0, "bpt": 1.0}
UNIT_SUFFIX = {"rel": "%", "bpb": " bpb", "bpt": " b/tok"}   # compact label for on-figure stats


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
    curr_bits_base: Optional[float] = None   # L(curr|S_0); None if run predates --measure-gk regret

    @property
    def g_bits(self) -> float:
        """Bits this update saved on the NEXT interval (g_k). <0 = it hurt."""
        return self.next_bits_pre - self.next_bits_post

    @property
    def din_bits(self) -> float:
        """Bits this update saved on the interval it trained on (Δ_in, in-sample)."""
        return self.curr_bits_pre - self.curr_bits_post

    @property
    def regret_bits(self) -> Optional[float]:
        """Online cost − frozen-base cost on the interval it's coding (regret-vs-base).
        >0 = the adapted model is WORSE than doing nothing = cumulative overfitting.
        None if the run didn't record the base term."""
        if self.curr_bits_base is None:
            return None
        return self.curr_bits_pre - self.curr_bits_base

    def regret(self, unit: str) -> Optional[float]:
        """regret_bits normalised the same way as g/din (fraction of base cost / Δbpb / b-tok)."""
        r = self.regret_bits
        if r is None:
            return None
        if unit == "bpb":
            return r / self.curr_bytes if self.curr_bytes else 0.0
        if unit == "bpt":
            return r / self.curr_tokens if self.curr_tokens else 0.0
        return r / self.curr_bits_base if self.curr_bits_base else 0.0

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


def transfer_efficiency(bs: List[Boundary]) -> float:
    """Stream-level g_k / Δ_in — the fraction of the in-sample gain that carries
    forward to the next interval (the prequential generalization efficiency; ~1/8
    in the first enwik9 read).  Size-weighted (total transfer bits / total in-sample
    bits), so it is unit-independent."""
    din = sum(b.din_bits for b in bs)
    return sum(b.g_bits for b in bs) / din if din else 0.0


# --- regret-vs-base (cumulative overfitting; needs the base term) -----------

def has_regret(bs: List[Boundary]) -> bool:
    """True iff every boundary recorded the base term (run used the newer instrument)."""
    return bool(bs) and all(b.curr_bits_base is not None for b in bs)


def worse_than_base_fraction(bs: List[Boundary]) -> float:
    """Fraction of intervals where the adapted model is worse than the frozen base
    (regret > 0) — the per-interval 'adaptation is hurting right now' rate."""
    rs = [b.regret_bits for b in bs if b.regret_bits is not None]
    return sum(1 for r in rs if r > 0) / len(rs) if rs else 0.0


def cumulative_regret_bits(bs: List[Boundary]) -> List[float]:
    """Running Σ regret_bits = online-total − base-total so far.  >0 = online has
    fallen BEHIND static; the boundary where it crosses 0 is when adaptation turns net-harmful."""
    out, acc = [], 0.0
    for b in bs:
        acc += b.regret_bits or 0.0
        out.append(acc)
    return out


def stream_regret(bs: List[Boundary], unit: str) -> float:
    """Stream-level regret: total (online − base) over total base cost (rel) / bytes / tokens.
    In 'rel' this is ≈ −delta_pct (base ≈ static), a free cross-check against summary.csv."""
    num = sum(b.regret_bits for b in bs if b.regret_bits is not None)
    if unit == "bpb":
        den = sum(b.curr_bytes for b in bs)
    elif unit == "bpt":
        den = sum(b.curr_tokens for b in bs)
    else:
        den = sum(b.curr_bits_base for b in bs if b.curr_bits_base is not None)
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
    params: dict            # defining run params (model/data/rank/…) for self-labelling


def _extract_params(blob: dict) -> dict:
    """The run's defining parameters, from the eval_online --json header (args +
    config), so a summary or figure identifies itself without relying on the
    filename."""
    a, c = blob.get("args", {}), blob.get("config", {})

    def _base(p):
        return os.path.basename(str(p).rstrip("/\\")) if p else None
    return {
        "model": _base(a.get("model")),
        "data": _base(a.get("data")),
        "lora_r": c.get("lora_r", a.get("lora_r")),
        "lora_alpha": c.get("lora_alpha", a.get("lora_alpha")),
        "chunk_size": a.get("chunk_size"),
        "lr": c.get("learning_rate", a.get("lr")),
        "epochs": c.get("epochs_per_train", a.get("epochs_per_train")),
        "train_interval": c.get("train_interval", a.get("train_interval")),
        "max_bytes": a.get("max_bytes"),
    }


def _human_size(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    return f"{n / 1e6:.2f}MB" if n >= 1e6 else f"{n / 1e3:.0f}KB"


def _format_config(p: dict, n_boundaries: int) -> str:
    """One-line, human-readable run signature (model · data · r · chunk · lr · …)."""
    parts = [str(p[k]) for k in ("model", "data") if p.get(k)]
    if p.get("lora_r") is not None:
        r = f"r{p['lora_r']}"
        if p.get("lora_alpha") is not None:
            r += f"(α{p['lora_alpha']})"
        parts.append(r)
    if p.get("chunk_size") is not None:
        parts.append(f"chunk {p['chunk_size']}")
    if p.get("lr") is not None:
        parts.append(f"lr {p['lr']:g}")
    if p.get("epochs") is not None:
        parts.append(f"ep {p['epochs']}")
    if p.get("train_interval") is not None:
        parts.append(f"ti {p['train_interval']}")
    if p.get("max_bytes") is not None:
        parts.append(_human_size(p["max_bytes"]))
    parts.append(f"{n_boundaries} bnd")
    return "  ·  ".join(parts)


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
            curr_bits_base=c.get("bits_base"),
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
        params=_extract_params(blob),
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(run: GkRun, unit: str) -> None:
    bs = run.boundaries
    scale, label = UNIT_SCALE[unit], UNIT_LABEL[unit]
    print(f"\n{'=' * 72}\n  {run.tag}   ({run.modality}, {len(bs)} boundaries)\n{'=' * 72}")
    print(f"  config: {_format_config(run.params, len(bs))}")
    print(f"  source: {run.source}")
    print(f"  harmful updates (g_k < 0): {harmful_fraction(bs) * 100:.1f}%"
          f"  ({sum(1 for b in bs if b.g_bits < 0)}/{len(bs)})")
    print("  " + "-" * 68)
    print(f"  g_k   mean (per-boundary) = {mean_g(bs, unit) * scale:+.4f} {label}")
    print(f"  g_k   stream (size-weighted, = transferability)"
          f" = {stream_g(bs, unit) * scale:+.4f} {label}")
    print(f"  Δ_in  mean (per-boundary) = {mean_din(bs, unit) * scale:+.4f} {label}")
    print(f"  transfer efficiency (g_k/Δ_in) = {transfer_efficiency(bs) * 100:.1f}%")
    # bits/byte is the cross-model headline regardless of the chosen display unit.
    if unit != "bpb":
        print(f"  g_k   stream Δbpb          = {stream_g(bs, 'bpb'):+.4f} bits/byte")
    if has_regret(bs):
        cum = cumulative_regret_bits(bs)
        behind = cum[-1] > 0
        print("  " + "-" * 68)
        print(f"  regret>0 (online worse than base): {worse_than_base_fraction(bs) * 100:.1f}%")
        print(f"  regret stream (online−base)      = {stream_regret(bs, unit) * scale:+.4f} {label}"
              f"   (≈ −delta_pct)")
        print(f"  cumulative regret (final)        = {cum[-1] / 8:+.0f} B  "
              f"({'online BEHIND static — adaptation net-harmful' if behind else 'online ahead of static'})")
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
    scale, label, suffix = UNIT_SCALE[unit], UNIT_LABEL[unit], UNIT_SUFFIX[unit]
    x = [b.phase for b in bs]
    g = [b.g(unit) * scale for b in bs]
    din = [b.din(unit) * scale for b in bs]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in g]
    regret_avail = has_regret(bs)

    # A regret (cumulative-drift) panel is inserted between g_k and the scatter when
    # the run recorded the base term; older runs fall back to the original 2 panels.
    n = 3 if regret_avail else 2
    ratios = [2, 1, 1] if regret_avail else [2, 1]
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 9.6 if regret_avail else 7.8),
                             gridspec_kw={"height_ratios": ratios})
    ax0, ax_reg, ax1 = axes[0], (axes[1] if regret_avail else None), axes[-1]

    # Header: bold phenomenon + the run's full parameter signature + headline stats,
    # so the figure identifies itself without the filename (see _format_config).
    stats_line = (f"harmful {harmful_fraction(bs) * 100:.0f}%      "
                  f"mean g_k {stream_g(bs, unit) * scale:+.3g}{suffix}      "
                  f"transfer eff {transfer_efficiency(bs) * 100:.0f}%")
    if regret_avail:
        stats_line += f"      regret>0 {worse_than_base_fraction(bs) * 100:.0f}%"
    fig.suptitle(f"Prequential adaptation gain — {run.tag}", fontsize=13,
                 fontweight="bold", y=0.995)
    ax0.set_title(_format_config(run.params, len(bs)) + "\n" + stats_line,
                  fontsize=8.5, color="0.30", pad=8)

    # top: per-boundary g_k (green helped / red hurt) with Δ_in overlaid
    ax0.bar(x, g, color=colors, width=0.8, zorder=2,
            label="g_k  (green ≥0 / red <0)")
    ax0.plot(x, din, color="#7f7f7f", lw=1.3, marker="o", ms=3, zorder=3,
             label="Δ_in (in-sample)")
    ax0.axhline(0, color="0.4", lw=1.0)
    ax0.set_ylabel(f"g_k  [{label}]")
    if not regret_avail:
        ax0.set_xlabel("training boundary (phase)")
    ax0.legend(loc="upper right", frameon=False)
    ax0.grid(alpha=0.25, lw=0.5)

    # middle: cumulative regret-vs-base (%) — the drift curve.  <0 online ahead of
    # static (green), >0 online behind (red); the crossover = when adaptation turns
    # net-harmful.  This is what g_k alone cannot show (see regret_bits).
    if regret_avail:
        cum, base_cum, acc, bacc = [], [], 0.0, 0.0
        for b in bs:
            acc += b.regret_bits or 0.0
            bacc += b.curr_bits_base or 0.0
            cum.append(acc)
            base_cum.append(bacc)
        cum_rel = [100.0 * c / bc if bc else 0.0 for c, bc in zip(cum, base_cum)]
        ax_reg.plot(x, cum_rel, color="#555555", lw=1.5, zorder=3)
        ax_reg.axhline(0, color="0.4", lw=1.0)
        ax_reg.fill_between(x, 0, cum_rel, where=[v <= 0 for v in cum_rel],
                            color="#2ca02c", alpha=0.15, interpolate=True)
        ax_reg.fill_between(x, 0, cum_rel, where=[v > 0 for v in cum_rel],
                            color="#d62728", alpha=0.15, interpolate=True)
        cross = next((xi for xi, v in zip(x, cum_rel) if v > 0), None)
        if cross is not None:
            ax_reg.axvline(cross, color="#d62728", lw=1.0, ls="--")
            ax_reg.annotate("online falls behind static", xy=(cross, 0), xytext=(4, 6),
                            textcoords="offset points", fontsize=8, color="#d62728")
        ax_reg.set_ylabel("cumulative regret\n[% vs base]  (>0 worse)")
        ax_reg.set_xlabel("training boundary (phase)")
        ax_reg.grid(alpha=0.25, lw=0.5)

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

    fig.tight_layout(rect=[0, 0, 1, 0.96])       # leave room for the suptitle
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")   # never clip long axis labels
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
