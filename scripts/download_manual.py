#!/usr/bin/env python3
"""Helpers for the few datasets that the three auto-downloaders can't fully script.

Everything that downloads cleanly lives in download_text/image/audio.py.  What's
left needs a special step:

  chestxray14  image — NIH images are split across Box tarballs (Box UI / batch script)
  musdb18      audio — ships as multitrack .stem.mp4; needs the `musdb` decoder
  icbhi        audio — respiratory sounds, behind a registration form

(CLIC is now auto-downloadable — see `scripts/download_image.py --dataset clic2024`.)

Run from the repo root:

    python scripts/download_manual.py musdb18 --limit 20     # auto if `musdb` is installed
    python scripts/download_manual.py chestxray14            # prints instructions
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DATA_ROOT  # noqa: E402


def do_chestxray14(args) -> None:
    print(
        "  [chestxray14] MANUAL — NIH ChestX-ray14 (medical, high online gain).\n"
        "    Images are 12 tarballs on NIH Box:\n"
        "      https://nihcc.app.box.com/v/ChestXray-NIHCC\n"
        "    Use NIH's batch_download_zips.py (links in the Box folder) and extract\n"
        "    the PNGs into  data/image/chestxray14/ .  License: research use.")


def do_icbhi(args) -> None:
    print(
        "  [icbhi] MANUAL — ICBHI respiratory sounds (bioacoustic, high online gain).\n"
        "    Register / download from  https://bhichallenge.med.auth.gr/  and put the\n"
        "    .wav files in  data/audio/icbhi/ .  License: research use (registration).")


def do_musdb18(args) -> None:
    out = DATA_ROOT / "audio" / "musdb18"
    try:
        import musdb
        import soundfile as sf
    except ImportError:
        print(
            "  [musdb18] needs the decoder package:\n"
            "      pip install musdb soundfile      # (+ ffmpeg on PATH)\n"
            "    then re-run to fetch the free 7-second STEMS sample.\n"
            "    Full HQ set (academic / non-commercial): musdb18-hq from\n"
            "      https://sigsep.github.io/datasets/musdb.html (Zenodo).")
        return
    out.mkdir(parents=True, exist_ok=True)
    print("  [musdb18] fetching the free 7s STEMS sample via `musdb` ...")
    db = musdb.DB(root=str(out / "_stems"), download=True)
    n = 0
    for track in db[: args.limit]:
        sf.write(out / f"clip_{n:05d}.wav", track.audio, track.rate)
        n += 1
    print(f"  [musdb18] -> {out}  ({n} mixture wavs)")


HANDLERS = {"chestxray14": do_chestxray14, "icbhi": do_icbhi, "musdb18": do_musdb18}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Manual-step datasets (chestxray14 / musdb18 / icbhi)")
    p.add_argument("dataset", choices=list(HANDLERS))
    p.add_argument("--limit", type=int, default=30, help="musdb18: # clips to extract")
    args = p.parse_args()
    HANDLERS[args.dataset](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
