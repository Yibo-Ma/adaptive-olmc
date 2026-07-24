#!/usr/bin/env python3
"""Normalize raw text so it survives a tokenizer round-trip losslessly.

The compressor tokenizes text then reconstructs it via ``tokenizer.decode``; a few
characters a tokenizer cannot represent would otherwise break the byte-exact
round-trip.  This script applies that ``encode -> decode`` normalization ahead of
time.

Normalization is **tokenizer-specific**, so this produces *one normalized copy per
tokenizer*.  By default it normalizes with both the Qwen2.5 and Qwen3 tokenizers:

    data/text/raw/<dataset>/            ->  data/text/normalized/<tag>/<dataset>/
                                            tag = qwen2.5 | qwen3

Compress with the matching copy: a Qwen2.5 model -> normalized/qwen2.5/...,
a Qwen3 model -> normalized/qwen3/...  (eval_online picks this automatically).

Each run records, per dataset, exactly which characters the round-trip changed
(``char_changes`` in ``normalize_summary.json``) and prints a compact breakdown.

Run from the repo root:

    python scripts/normalize_text.py                                   # all datasets, both tokenizers
    python scripts/normalize_text.py --dataset pile_of_law_eurlex medal
    python scripts/normalize_text.py --models checkpoints/Qwen3-1.7B-Base --force
    python scripts/normalize_text.py --models checkpoints/Llama-3.2-1B checkpoints/SmolLM2-1.7B
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: utils.*

from _common import DATA_ROOT, human_bytes  # noqa: E402

RAW_ROOT = DATA_ROOT / "text" / "raw"
NORM_ROOT = DATA_ROOT / "text" / "normalized"
DEFAULT_MODELS = ["checkpoints/Qwen2.5-0.5B", "checkpoints/Qwen3-1.7B-Base"]
SHARD_BYTES = 50 << 20
CHUNK_BYTES = 2048              # reporting granularity (matches compressor segment size)


def tokenizer_tag(model_path: str) -> str:
    """Short folder tag for a tokenizer, e.g. Qwen2.5-0.5B -> 'qwen2.5'."""
    base = os.path.basename(str(model_path).rstrip("/\\")).lower()
    m = re.match(r"(qwen[0-9.]+)", base)
    return m.group(1) if m else re.sub(r"[^a-z0-9.]+", "-", base).strip("-")


def roundtrip(text: str, tok) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def char_label(c: str) -> str:
    """Unambiguous, legible label for one character, e.g. '’' -> "U+2019 '’'".
    Non-printable characters show the codepoint alone."""
    cp = f"U+{ord(c):04X}"
    return f"{cp} {c!r}" if c.isprintable() else cp


def char_diff(orig: str, norm: str) -> dict:
    """Which characters the round-trip changed.

    Length-preserving edits are reported as exact old->new ``substitutions``;
    otherwise as multiset deltas (``removed`` / ``added``), since a changed length
    has no unambiguous 1:1 alignment.  Returns ``{"changed": False}`` when identical.
    """
    if orig == norm:
        return {"changed": False}
    if len(orig) == len(norm):
        subs = Counter((a, b) for a, b in zip(orig, norm) if a != b)
        return {
            "changed": True, "mode": "substitution",
            "edits": sum(subs.values()), "distinct": len(subs),
            "substitutions": [{"from": char_label(a), "to": char_label(b), "count": n}
                              for (a, b), n in subs.most_common()],
        }
    removed, added = Counter(orig) - Counter(norm), Counter(norm) - Counter(orig)
    return {
        "changed": True, "mode": "length-changed",
        "edits": sum(removed.values()) + sum(added.values()),
        "distinct": len(removed) + len(added),
        "removed": {char_label(c): n for c, n in removed.most_common()},
        "added": {char_label(c): n for c, n in added.most_common()},
    }


def print_char_changes(cc: dict, top: int = 8, indent: str = "        ") -> None:
    """Compact, capped console breakdown of a :func:`char_diff` result.

    Codepoints only, so it renders in any terminal / log; the glyph-annotated
    labels live in ``normalize_summary.json``.
    """
    cp = lambda label: label.split(" ", 1)[0]        # "U+2019 '’'" -> "U+2019"
    if cc["mode"] == "substitution":
        rows = [f"{cp(s['from'])} -> {cp(s['to'])}  x{s['count']}" for s in cc["substitutions"]]
    else:
        rows = ([f"- {cp(lbl)}  x{n}" for lbl, n in cc["removed"].items()]
                + [f"+ {cp(lbl)}  x{n}" for lbl, n in cc["added"].items()])
    for r in rows[:top]:
        print(indent + r)
    if len(rows) > top:
        print(f"{indent}... and {len(rows) - top} more (see normalize_summary.json)")


def read_raw(ds_dir: Path, max_bytes) -> str:
    """Concatenate part-*.txt (in order), capped at max_bytes."""
    buf, total = [], 0
    for part in sorted(ds_dir.glob("part-*.txt")):
        data = part.read_bytes()
        if max_bytes is not None and total + len(data) > max_bytes:
            data = data[: max_bytes - total]
        buf.append(data)
        total += len(data)
        if max_bytes is not None and total >= max_bytes:
            break
    return b"".join(buf).decode("utf-8", errors="ignore")


def write_shards(text: str, out_dir: Path) -> int:
    for old in out_dir.glob("part-*.txt"):
        old.unlink()
    data = text.encode("utf-8")
    n = max(1, (len(data) + SHARD_BYTES - 1) // SHARD_BYTES)
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as man:
        for s in range(n):
            chunk = data[s * SHARD_BYTES:(s + 1) * SHARD_BYTES]
            (out_dir / f"part-{s:05d}.txt").write_bytes(chunk)
            man.write(json.dumps({"shard": s, "bytes": len(chunk)}) + "\n")
    return n


def normalize_dataset(key: str, tok, tag: str, max_bytes, force: bool) -> dict:
    src = RAW_ROOT / key
    out = NORM_ROOT / tag / key
    summary_path = out / "normalize_summary.json"
    if summary_path.exists() and not force:
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"    [{key}] already normalized for {tag} "
              f"({human_bytes(prev.get('output_bytes', 0))}) — use --force to redo")
        return prev
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    text = read_raw(src, max_bytes)
    in_bytes = len(text.encode("utf-8"))
    norm = roundtrip(text, tok)
    if roundtrip(norm, tok) != norm:                 # idempotency guard
        norm = roundtrip(norm, tok)
    out_bytes = len(norm.encode("utf-8"))
    changed = sum(
        1 for i in range(0, len(text), CHUNK_BYTES)
        if text[i:i + CHUNK_BYTES] != norm[i:i + CHUNK_BYTES]
    )
    chars = char_diff(text, norm)
    shards = write_shards(norm, out)

    summary = {
        "dataset": key, "tokenizer_tag": tag, "model": getattr(tok, "name_or_path", "?"),
        "input_dir": str(src), "output_dir": str(out),
        "shard_bytes": SHARD_BYTES, "roundtrip_chunk_bytes": CHUNK_BYTES,
        "max_input_bytes": max_bytes, "input_bytes": in_bytes, "output_bytes": out_bytes,
        "approx_changed_chunks": changed, "shards": shards,
        "char_changes": chars,
        "seconds": round(time.time() - t0, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    flag = ("byte-identical" if not chars["changed"]
            else f"{chars['edits']} edits / {chars['distinct']} char-type(s)")
    print(f"    [{key}] -> {out}  ({human_bytes(out_bytes)}, {flag}, {summary['seconds']}s)")
    if chars["changed"]:
        print_char_changes(chars)
    return summary


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Normalize raw text per tokenizer (Qwen2.5 + Qwen3)")
    p.add_argument("--dataset", nargs="*", help="dataset key(s); default: all under data/text/raw")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                   help="tokenizer model path(s); one normalized copy per tokenizer")
    p.add_argument("--max-bytes", type=int, default=None, help="cap input bytes per dataset")
    p.add_argument("--force", action="store_true", help="re-normalize even if a summary exists")
    args = p.parse_args()

    if not RAW_ROOT.exists():
        print(f"No raw text at {RAW_ROOT}. Run scripts/download_text.py first."); return 1
    keys = args.dataset or sorted(d.name for d in RAW_ROOT.iterdir() if d.is_dir())
    if not keys:
        print(f"No datasets under {RAW_ROOT}."); return 1

    # Same loader as the compressor backend (slow-first, fast fallback), so the
    # normalized copy is a fixed point of the exact tokenizer eval_online will use.
    from utils.text_utils import load_lm_tokenizer
    for model in args.models:
        tag = tokenizer_tag(model)
        print(f"\n=== tokenizer '{tag}'  ({model}) ===")
        try:
            tok = load_lm_tokenizer(model)
        except Exception as e:
            print(f"  could not load tokenizer {model}: {type(e).__name__}: {e}")
            print(f"  (download it first: python scripts/download_models.py --model <key>)")
            continue
        out_root = NORM_ROOT / tag
        out_root.mkdir(parents=True, exist_ok=True)
        index = []
        for k in keys:
            if not (RAW_ROOT / k).is_dir():
                print(f"    [{k}] no raw data at {RAW_ROOT / k} — download it first "
                      f"(python scripts/download_text.py --dataset {k}); skipped")
                continue
            index.append(normalize_dataset(k, tok, tag, args.max_bytes, args.force))
        (out_root / "normalize_summary.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in index), encoding="utf-8")
        touched = [s["dataset"] for s in index if s.get("char_changes", {}).get("changed")]
        print(f"  normalized {len(index)} dataset(s) -> {out_root}"
              + (f"; {len(touched)} changed: {', '.join(touched)}" if touched
                 else "; all byte-identical"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
