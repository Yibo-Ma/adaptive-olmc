#!/usr/bin/env python3
"""Build a heterogeneous (multi-domain) text stream for the overfit-vs-static test.

Concatenates a contiguous byte-prefix of each of several *raw* datasets into a single
``part-00000.txt``, so the stream contains hard domain switches.  This is the stream
used to check whether vanilla online adaptation over-fits one domain and then loses to
Static across the switch (cumulative regret going positive).

It reads from and writes into ``data/text/raw/`` (the base path the current pipeline
uses), so the result is a drop-in dataset directory ``eval_online.py`` consumes:

    data/text/raw/<key>/part-*.txt          (sources, one per domain)
        -> data/text/raw/<out>/             part-00000.txt
                                            manifest.jsonl      (shard sizes)
                                            mix_manifest.json   (provenance)

Each block is a *prefix* of its source (coherent in-domain, giving a clean switch at
each boundary), cut on a UTF-8 char boundary and backed off to the nearest newline.
The per-block byte ranges are recorded in ``mix_manifest.json`` so the switch offsets
can be mapped onto eval_online's chunk / regret curve.

Raw text is *not* tokenizer-normalized, so a byte-exact round-trip is not guaranteed
(that needs ``data/text/normalized/``); run the compress-only rate test with
``--no-decompress`` (as the rank/lr/epoch sweeps do).

Run from the repo root:

    # default: the intended 3 domains (present in raw/ on the cluster)
    python scripts/build_mixed_stream.py

    # explicit / local subset / custom size-weights-name:
    python scripts/build_mixed_stream.py --datasets enwik9,medal,codesearchnet
    python scripts/build_mixed_stream.py --datasets enwik9,beancounter,pile_of_law_eurlex \
        --total-bytes 10MB --weights 1,1,1 --out mix_enwik_bean_law

NOTE: this dev machine only syncs a subset of ``data/text/raw/`` (no beancounter /
pile_of_law_eurlex locally); the script fail-fasts and lists what is available.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DATA_ROOT, human_bytes, parse_size  # noqa: E402

RAW = DATA_ROOT / "text" / "raw"
DEFAULT_DATASETS = "enwik9,beancounter,pile_of_law_eurlex"
DEFAULT_TOTAL = 10_000_000
# How far back to look for a newline when trimming a block to a clean document edge.
NEWLINE_WINDOW = 8192


def _short(key: str) -> str:
    """Compact tag for the auto output name (``pile_of_law_eurlex`` -> ``pile``)."""
    return key.split("_", 1)[0]


def _available():
    """Raw datasets that actually carry part-*.txt data."""
    if not RAW.is_dir():
        return []
    return [d.name for d in sorted(RAW.iterdir())
            if d.is_dir() and glob.glob(str(d / "part-*.txt"))]


def _load_source_bytes(key: str) -> bytes:
    """Concatenate a source's part-*.txt shards in order (matches eval_online)."""
    parts = sorted(glob.glob(str(RAW / key / "part-*.txt")))
    if not parts:
        avail = ", ".join(_available()) or "(none)"
        sys.exit(f"[build_mixed_stream] ERROR: no part-*.txt for '{key}' under "
                 f"{RAW / key}\n                   available under data/text/raw: {avail}")
    raw = b""
    for p in parts:
        with open(p, "rb") as f:
            raw += f.read()
    return raw


def _safe_cut(raw: bytes, target: int) -> int:
    """Largest cut <= target that leaves valid UTF-8, preferring a newline boundary.

    First back off any partial multi-byte char, then, if a newline sits within
    ``NEWLINE_WINDOW`` bytes before the cut, end the block right after it so each
    domain block terminates on a clean document edge.
    """
    if target >= len(raw):
        return len(raw)
    cut = target
    # 0x80-0xBF are UTF-8 continuation bytes; step back to a char start.
    while cut > 0 and (raw[cut] & 0xC0) == 0x80:
        cut -= 1
    nl = raw.rfind(b"\n", max(0, cut - NEWLINE_WINDOW), cut)
    if nl != -1:
        cut = nl + 1
    return cut


def _block_targets(total: int, n: int, weights):
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        sys.exit(f"[build_mixed_stream] ERROR: --weights has {len(weights)} values "
                 f"but {n} datasets")
    s = sum(weights)
    return [int(round(total * w / s)) for w in weights]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=DEFAULT_DATASETS,
                    help="comma-separated raw dataset keys, in stream order "
                         f"(default: {DEFAULT_DATASETS})")
    ap.add_argument("--total-bytes", default=str(DEFAULT_TOTAL), metavar="N",
                    help="total stream size, accepts 10MB / 10000000 (default: 10MB)")
    ap.add_argument("--weights", default=None,
                    help="comma-separated block weights (default: equal)")
    ap.add_argument("--out", default=None,
                    help="output dataset dir name (default: mix_<d0>_<d1>_...)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing non-empty output dir")
    args = ap.parse_args()

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    if len(keys) < 2:
        sys.exit("[build_mixed_stream] ERROR: need >= 2 datasets for a mixed stream")
    total = parse_size(args.total_bytes)
    if not total or total <= 0:
        sys.exit(f"[build_mixed_stream] ERROR: bad --total-bytes {args.total_bytes!r}")
    weights = ([float(w) for w in args.weights.split(",")] if args.weights else None)
    targets = _block_targets(total, len(keys), weights)

    out_key = args.out or ("mix_" + "_".join(_short(k) for k in keys))
    out_dir = RAW / out_key
    if out_dir.is_dir() and glob.glob(str(out_dir / "part-*.txt")) and not args.force:
        sys.exit(f"[build_mixed_stream] ERROR: {out_dir} already has data; pass --force")

    print(f"[build_mixed_stream] target={human_bytes(total)}  "
          f"out=data/text/raw/{out_key}\n")

    # --- assemble blocks (fail-fast on any missing / too-small source) ---
    blocks, buf, offset = [], bytearray(), 0
    for key, want in zip(keys, targets):
        raw = _load_source_bytes(key)
        if len(raw) < want:
            sys.exit(f"[build_mixed_stream] ERROR: '{key}' has {human_bytes(len(raw))}"
                     f" < requested {human_bytes(want)} for its block")
        cut = _safe_cut(raw, want)
        chunk = raw[:cut]
        buf += chunk
        blocks.append(dict(order=len(blocks), dataset=key, byte_start=offset,
                           byte_end=offset + len(chunk), bytes=len(chunk)))
        offset += len(chunk)
    data = bytes(buf)

    # --- verify before writing (byte-exact provenance) ---
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as e:
        sys.exit(f"[build_mixed_stream] ERROR: assembled stream is not valid UTF-8: {e}")
    assert len(data) == sum(b["bytes"] for b in blocks), "block byte accounting mismatch"
    for b in blocks:
        src = _load_source_bytes(b["dataset"])[: b["bytes"]]
        assert data[b["byte_start"]:b["byte_end"]] == src, \
            f"block '{b['dataset']}' not a byte-exact prefix of its source"

    # --- write dataset dir (part + manifest + provenance) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in glob.glob(str(out_dir / "part-*.txt")):
        os.remove(stale)
    (out_dir / "part-00000.txt").write_bytes(data)
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"shard": 0, "bytes": len(data)}) + "\n")
    mix_manifest = dict(out_key=out_key, source_root="data/text/raw",
                        created_by="scripts/build_mixed_stream.py",
                        total_bytes=len(data), target_total_bytes=total,
                        datasets=keys, weights=weights or [1.0] * len(keys),
                        blocks=blocks)
    with open(out_dir / "mix_manifest.json", "w", encoding="utf-8") as f:
        json.dump(mix_manifest, f, indent=2, ensure_ascii=False)

    # --- report (aligned) ---
    print(f"  {'#':>1}  {'domain':<22} {'bytes':>9}  {'share':>6}  byte range")
    for b in blocks:
        share = 100.0 * b["bytes"] / len(data)
        print(f"  {b['order']:>1}  {b['dataset']:<22} {human_bytes(b['bytes']):>9}  "
              f"{share:5.1f}%  {b['byte_start']:,} - {b['byte_end']:,}")
    print(f"     {'-' * 46}")
    print(f"  {'':>1}  {'total':<22} {human_bytes(len(data)):>9}"
          f"          ({len(blocks) - 1} domain switch{'es' if len(blocks) > 2 else ''})\n")
    print("[verify] utf-8 decode OK - block prefixes byte-identical to sources - total matches")
    print(f"[write]  {out_dir / 'part-00000.txt'}")
    print(f"         + manifest.jsonl + mix_manifest.json")


if __name__ == "__main__":
    main()
