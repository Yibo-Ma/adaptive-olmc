#!/usr/bin/env python3
"""Download AUDIO datasets into ``data/audio/<key>/``.

Only license-clean, verified-downloadable datasets.  tar(.gz/.bz2) sources come from
direct URLs; People's Speech streams through hf-mirror as raw-byte parquet.  Run from
the repo root.

    python scripts/download_audio.py --list
    python scripts/download_audio.py --dataset librispeech --limit 50
    python scripts/download_audio.py --dataset ljspeech nsynth --limit 30
    python scripts/download_audio.py --all

``--limit`` = number of clips per dataset.  bGPT eval reads these as WAV/FLAC blobs
(see eval_online --modality audio).

tar(.gz/.bz2) sources (librispeech / ljspeech / nsynth / esc50 / speech_commands) are
**streamed**: the connection closes once ``--limit`` clips are extracted, so ``--limit``
also caps the download.  zip sources (MAESTRO) keep their index at the end, so they must
download fully before extracting ``--limit`` clips.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    DATA_ROOT, extract_media, http_download, is_streamable_tar, run_download_cli,
    save_progress, stream_extract_tar,
)

AUD = DATA_ROOT / "audio"
AUDIO_EXTS = (".flac", ".wav")

DATASETS = {
    "librispeech": dict(kind="url_tar", url="https://www.openslr.org/resources/12/test-clean.tar.gz",
                        license="CC-BY 4.0", gain="med",
                        note="OpenSLR direct (avoids HF librosa dep); single reader per chapter"),
    "ljspeech":    dict(kind="url_tar", url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
                        license="public domain", gain="high", note="single female speaker (2.6GB)"),
    "maestro":     dict(kind="hf_audio", hf_id="ddPn08/maestro-v3.0.0", revision="main",
                        manifest="maestro-v3.0.0.csv", audio_col="audio_filename",
                        license="CC-BY-NC-SA (non-commercial)", gain="high",
                        note="solo piano, ~9 min/clip (long-form + homogeneous -> ideal for online). "
                             "The mirror stores raw year-folder .wav files (no parquet). We DIRECT-download "
                             "the first --limit clips over the mirror's /resolve/ URLs, driven by the repo's "
                             "CSV manifest — bypassing datasets.load_dataset(streaming), whose /api tree "
                             "listing ignores HF_ENDPOINT and hits huggingface.co. Fetches a few clips, not 129GB."),
    "peoples_speech": dict(kind="hf_parquet", hf_id="MLCommons/peoples_speech", config="test",
                           split="test", audio_key="audio", license="CC-BY 4.0 / CC0", gain="med"),
    # homogeneous / single-source -> strong online gain
    "nsynth": dict(kind="url_tar",
                   url="http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
                   license="CC-BY 4.0", gain="high",
                   note="single instrument notes — very homogeneous (great for online)"),
    "speech_commands": dict(kind="url_tar",
                            url="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                            license="CC-BY 4.0", gain="med",
                            note="short spoken words; many near-identical utterances"),
    "esc50": dict(kind="url_tar",
                  url="https://github.com/karoldvl/ESC-50/archive/master.tar.gz",
                  license="CC BY-NC 3.0 (non-commercial)", gain="high",
                  note="environmental sounds; stationary texture (bioacoustic-like)"),
}


def dl_url_tar(key, spec, out, limit):
    """tar(.gz/.bz2): stream and close the connection after ``limit`` clips, so ``--limit``
    caps the download.  zip (MAESTRO) keeps its index at the end, so it must download fully."""
    out.mkdir(parents=True, exist_ok=True)
    url = spec["url"]
    if is_streamable_tar(url):
        print(f"  [{key}] {url}  (streaming; stops after {limit} clips)")
        n = stream_extract_tar(url, out, limit, AUDIO_EXTS, "clip", desc=key)
    else:
        big = " (MAESTRO is ~120GB)" if key == "maestro" else ""
        print(f"  [{key}] {url}{big}  (zip: downloads fully, then extracts {limit} clips)")
        src = out / ("_source" + "".join(Path(url).suffixes[-2:]))
        http_download(url, src, desc=f"{key} src")
        n = extract_media(src, out, limit, AUDIO_EXTS, "clip")
    save_progress(out, clips=n, source=url, gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({n} clips)")


def dl_hf_parquet(key, spec, out, limit):
    """Stream the HF parquet and stop after ``limit`` clips so --limit caps the download.
    Audio stays raw bytes (decode=False -> no librosa).  The written parquet keeps the
    {audio:{bytes,path}, ...} schema the loader expects (utils/audio_utils)."""
    from datasets import load_dataset, Audio, Dataset
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{spec.get('config', 'data')}-{spec['split']}.parquet"
    if dst.exists():
        print(f"  [{key}] {dst.name} exists; skip"); return
    print(f"  [{key}] {spec['hf_id']} ({spec.get('config')}) -> parquet "
          f"(streaming raw bytes; stops after {limit} clips)")
    ds = load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"], streaming=True)
    ds = ds.cast_column(spec.get("audio_key", "audio"), Audio(decode=False))
    rows = []
    for row in ds:
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    Dataset.from_list(rows).to_parquet(str(dst))
    save_progress(out, clips=len(rows), source=spec["hf_id"], gain=spec.get("gain"))
    print(f"  [{key}] -> {dst}  ({len(rows)} clips)")


def dl_hf_audio(key, spec, out, limit):
    """Fetch the first ``limit`` audio clips from a MAESTRO-style HF mirror by DIRECT
    file download over the mirror's ``/resolve/`` URLs, driven by the repo's CSV
    manifest.  This deliberately avoids ``datasets.load_dataset(streaming=True)``,
    whose ``/api/.../tree`` repo listing ignores ``HF_ENDPOINT`` and hits
    ``huggingface.co`` — which fails on mirror-only / air-gapped clusters.

    Repo layout expected (ddPn08/maestro-v3.0.0): year folders of ``.wav``/``.midi``
    plus ``maestro-v3.0.0.csv`` whose ``audio_filename`` column holds each clip's
    path relative to the repo root.  The csv is the first (small) file fetched, so a
    failure here is a clean connectivity test before any large audio pull.
    """
    import csv as _csv
    from urllib.parse import quote

    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob("clip_*.*")))
    if have >= limit:
        print(f"  [{key}] already {have} clips (>= limit)"); return

    repo = spec["hf_id"]
    rev = spec.get("revision", "main")
    manifest = spec.get("manifest", "maestro-v3.0.0.csv")
    audio_col = spec.get("audio_col", "audio_filename")
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

    def resolve(path: str) -> str:
        return f"{endpoint}/datasets/{repo}/resolve/{rev}/{quote(path, safe='/')}"

    print(f"  [{key}] direct /resolve download from {repo} (via {manifest}, no /api tree)")
    csv_local = out / "_manifest.csv"
    http_download(resolve(manifest), csv_local, desc="manifest")
    with open(csv_local, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if not rows or audio_col not in rows[0]:
        raise ValueError(
            f"[{key}] column '{audio_col}' not in {manifest}; columns = "
            f"{list(rows[0].keys()) if rows else '(empty csv)'}. Adjust audio_col/manifest.")
    audio_files = [r[audio_col] for r in rows if r.get(audio_col)]

    idx = have
    for rel in audio_files[have:limit]:
        ext = (os.path.splitext(rel)[1] or ".wav").lower()
        http_download(resolve(rel), out / f"clip_{idx:05d}{ext}", desc=f"clip {idx}")
        idx += 1
        if idx >= limit:
            break

    save_progress(out, clips=idx, source=repo, gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({idx} clips)")


HANDLERS = {"url_tar": dl_url_tar, "hf_parquet": dl_hf_parquet, "hf_audio": dl_hf_audio}


def main() -> int:
    return run_download_cli("audio", AUD, AUD, DATASETS, HANDLERS,
                            default_limit=50, limit_kind="count")


if __name__ == "__main__":
    raise SystemExit(main())
