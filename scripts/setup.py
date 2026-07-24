#!/usr/bin/env python3
"""One-command setup, split into two clear phases so a fresh clone is ready to run.

    python scripts/setup.py models     # weights              -> checkpoints/
    python scripts/setup.py data       # ALL datasets + normalize -> data/
    python scripts/setup.py all        # models then data  (default if omitted)

It orchestrates the other scripts in the right order, passing HF-endpoint flags
through.  ``data`` normalizes text, which needs the tokenizer — so run ``models``
first (``all`` does this for you).

By default ``data`` downloads **every** text/image/audio dataset (each capped by
``--*-limit``), keeps going past any failure, and prints a summary of what did /
did not download at the end.  Restrict with e.g. ``--text-datasets enwik9 medal``.

Common flags (any phase): --dry-run, --pip, --no-mirror, --hf-endpoint, --keep-going.

  models   qwen2.5-0.5b, qwen2.5-7b, qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b, bgpt  (~50 GB —
           restrict with e.g. --models qwen2.5-0.5b qwen3-1.7b bgpt)
  text     enwik8/9, text8, silesia, loghub, enron, pile_of_law_eurlex,
           atticus_contracts, edgar_corpus, codesearchnet, medal, hupd
  image    kodak, div2k, clic2024, usc_textures, dtd, eurosat, cifar10, oxford_pet
  audio    librispeech, ljspeech, maestro [~120GB!], peoples_speech, nsynth,
           speech_commands, esc50
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import REPO_ROOT, DATA_ROOT, read_download_status  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

DEFAULT_MODELS = ["qwen2.5-0.5b", "qwen2.5-7b", "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "bgpt"]
DEFAULT_NORM_MODELS = ["checkpoints/Qwen2.5-0.5B", "checkpoints/Qwen3-1.7B-Base"]


def _sel(datasets):
    """None -> download all; non-empty list -> those; [] -> skip the modality."""
    if datasets is None:
        return ["--all"]
    return ["--dataset", *datasets] if datasets else None


# ---------------------------------------------------------------------------
# Phase builders — each returns a list of (title, argv, soft) steps
# (soft=True: a failure is summarized at the end instead of aborting)
# ---------------------------------------------------------------------------

def model_steps(args, mirror) -> list:
    return [("download models",
             [PY, f"{SCRIPTS}/download_models.py", "--model", *args.models, *mirror], False)]


def data_steps(args, mirror) -> list:
    steps = []
    for mod, script, sel, limit in (
        ("text",  "download_text.py",  _sel(args.text_datasets),  args.text_limit),
        ("image", "download_image.py", _sel(args.image_datasets), args.image_limit),
        ("audio", "download_audio.py", _sel(args.audio_datasets), args.audio_limit),
    ):
        if sel is not None:
            steps.append((f"download {mod}",
                          [PY, f"{SCRIPTS}/{script}", *sel, "--limit", str(limit), *mirror], True))
    if args.text_datasets is None or args.text_datasets:   # normalize all downloaded text
        steps.append(("normalize text",
                      [PY, f"{SCRIPTS}/normalize_text.py", "--models", *args.normalize_models], False))
    return steps


# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Setup: model + data download (two phases)")
    p.add_argument("phase", nargs="?", choices=["models", "data", "all"], default="all")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="model keys (phase models)")
    # data: default None => download ALL datasets of that modality
    p.add_argument("--text-datasets", nargs="*", default=None)
    p.add_argument("--image-datasets", nargs="*", default=None)
    p.add_argument("--audio-datasets", nargs="*", default=None)
    p.add_argument("--text-limit", default="20MB", help="bytes per text dataset")
    p.add_argument("--image-limit", default="50", help="images per image dataset")
    p.add_argument("--audio-limit", default="50", help="clips per audio dataset")
    p.add_argument("--normalize-models", nargs="*", default=DEFAULT_NORM_MODELS)
    p.add_argument("--pip", action="store_true", help="also pip install -r requirements.txt")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--keep-going", action="store_true", help="continue past a failing HARD step")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    return p


def _report(phase) -> None:
    """Consolidated download summary read from each modality's status file."""
    print("\n" + "=" * 64)
    print("  DOWNLOAD SUMMARY")
    print("=" * 64)
    any_fail = False
    for mod in ("text", "image", "audio"):
        st = read_download_status(DATA_ROOT / mod)
        if not st:
            continue
        ok, failed, skipped = st.get("ok", []), st.get("failed", []), st.get("skipped", [])
        print(f"  {mod:<6} ok={ok or '[]'}")
        if skipped:
            print(f"         skipped (already present): {skipped}")
        if failed:
            any_fail = True
            print(f"         ❌ FAILED: {failed}")
    print("-" * 64)
    print("  ⚠️  有数据集未下载成功(见上 ❌ FAILED)。" if any_fail
          else "  ✅  所有数据集下载成功。")
    print("=" * 64)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = _build_parser().parse_args()
    mirror = (["--no-mirror"] if args.no_mirror else []) + \
             (["--hf-endpoint", args.hf_endpoint] if args.hf_endpoint else [])

    steps: list = []
    if args.pip:
        steps.append(("pip install",
                      [PY, "-m", "pip", "install", "-r", os.path.join(REPO_ROOT, "requirements.txt")], False))
    if args.phase in ("models", "all"):
        steps += model_steps(args, mirror)
    if args.phase in ("data", "all"):
        steps += data_steps(args, mirror)

    print("=" * 64)
    print(f"  OnlineLMCompress setup — phase: {args.phase}")
    print("=" * 64)
    for i, (title, argv, soft) in enumerate(steps, 1):
        shown = " ".join(a if " " not in a else f'"{a}"' for a in argv[1:])
        print(f"  {i}. {title}{'  (best-effort)' if soft else ''}\n       python {shown}")
    print("=" * 64)
    if args.dry_run:
        print("  (dry-run: nothing executed)")
        return 0

    for i, (title, argv, soft) in enumerate(steps, 1):
        print(f"\n>>> step {i}/{len(steps)}: {title}")
        rc = subprocess.run(argv, cwd=REPO_ROOT).returncode
        if rc != 0:
            print(f"  step '{title}' exited {rc}.")
            if soft:
                print("  (best-effort step — continuing; summarized at the end)")
            elif not args.keep_going:
                print("  aborting (use --keep-going to continue past failures).")
                return rc

    if args.phase in ("data", "all"):
        _report(args.phase)

    print("\n" + "=" * 64)
    if args.phase == "models":
        print("  Models ready. Next: python scripts/setup.py data")
    else:
        print("  Setup done. Try:")
        print("    python evaluation/eval_online.py --modality text  --mode both")
        print("    python evaluation/eval_online.py --modality image --mode both")
        print("    python evaluation/eval_online.py --modality audio --mode both --data data/audio/ljspeech --audio-clips 8")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
