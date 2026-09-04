"""Pure boundary-sampling / matched-control selection for the V1-A adjacency probe.

No torch here — it decides *which* boundaries to probe and *which* non-adjacent
windows to use as difficulty-matched controls, from per-interval domain labels and
static (frozen-base) NLLs alone.  online_compressor._record_adjacency then scores
the chosen windows under the pre/post-update model; evaluation/adjacency_curve.py
consumes the raw bits.  Keeping the selection here (side-effect-free, deterministic)
lets tests pin it without a model and keeps the producer thin.

A boundary k probes the true future window of ``h_max`` intervals [k+1 .. k+h_max].
A control is another window of the same length that is: same domain, same static-NLL
decile (difficulty-matched on the window total), non-overlapping with [k .. k+h_max],
and at least ``2*h_max`` intervals away from k.  Sampled boundaries are spread across
the stream and spaced at least ``h_max+1`` apart so their future windows never overlap.
"""
from __future__ import annotations

import bisect
import random
from typing import Dict, List, Optional, Sequence, Tuple


def _window_single_domain(domains: Sequence, start: int, h: int) -> Optional[object]:
    """Domain of the window [start, start+h), or None if it spans >1 domain / overruns."""
    if start < 0 or start + h > len(domains):
        return None
    d = domains[start]
    return d if all(domains[start + i] == d for i in range(h)) else None


def _decile_index(sorted_vals: List[float], v: float, n_deciles: int) -> int:
    r = bisect.bisect_right(sorted_vals, v) - 1
    r = min(len(sorted_vals) - 1, max(0, r))
    return min(n_deciles - 1, int(r * n_deciles / max(len(sorted_vals), 1)))


def plan_adjacency(
    domains: Sequence,
    static_nll: Sequence[float],
    h_max: int,
    k_controls: int,
    n_target: int,
    seed: int,
    n_deciles: int = 10,
) -> List[Tuple[int, List[int]]]:
    """Return ``[(boundary_k, [control_start, ...]), ...]`` deterministically.

    Boundaries with no in-domain future window, or without ``k_controls`` qualifying
    matched controls, are skipped (never relaxed).  The result is spread across the
    stream (strided) and spaced >= ``h_max+1`` apart."""
    n = len(static_nll)
    min_gap = h_max + 1

    # Per-window (start -> domain, static total) for every fully in-domain window.
    win_dom: Dict[int, object] = {}
    win_static: Dict[int, float] = {}
    for j in range(0, n - h_max + 1):
        d = _window_single_domain(domains, j, h_max)
        if d is not None:
            win_dom[j] = d
            win_static[j] = float(sum(static_nll[j:j + h_max]))

    # Per-domain sorted window totals -> decile lookup.
    per_dom_sorted: Dict[object, List[float]] = {}
    for j, s in win_static.items():
        per_dom_sorted.setdefault(win_dom[j], []).append(s)
    for d in per_dom_sorted:
        per_dom_sorted[d].sort()

    # Valid boundaries: the future window [k+1, k+h_max] is a single-domain window.
    valid_k = [k for k in range(0, n - 1) if (k + 1) in win_dom]
    if not valid_k:
        return []
    step = max(min_gap, len(valid_k) // max(n_target, 1))

    plan: List[Tuple[int, List[int]]] = []
    last_k = -(10 ** 9)
    idx = 0
    while idx < len(valid_k) and len(plan) < n_target:
        k = valid_k[idx]
        idx += step
        if k - last_k < min_gap:
            continue
        fut_start = k + 1
        d = win_dom[fut_start]
        target_dec = _decile_index(per_dom_sorted[d], win_static[fut_start], n_deciles)

        # Qualifying controls: same domain, same decile, non-adjacent, non-overlapping.
        cands = [
            j for j in win_dom
            if win_dom[j] == d
            and j != fut_start
            and abs(j - k) >= 2 * h_max
            and (j + h_max <= k or j >= k + h_max + 1)          # no overlap with [k, k+h_max]
            and _decile_index(per_dom_sorted[d], win_static[j], n_deciles) == target_dec
        ]
        if len(cands) < k_controls:
            continue
        rng = random.Random((seed << 20) ^ k)                    # per-boundary, reproducible
        controls = sorted(rng.sample(cands, k_controls))
        plan.append((k, controls))
        last_k = k
    return plan
