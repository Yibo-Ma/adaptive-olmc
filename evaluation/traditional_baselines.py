"""Traditional (non-neural) lossless compression baselines — the standard-codec column
of the paper's main tables.

Nine codecs, exactly the committed set:

    text   gzip · brotli · LZMA2 · cmix
    image  PNG · WebP · JPEG-XL
    audio  FLAC · OptimFROG

Design
------
* **Same denominators as eval_online.py** so the numbers are directly comparable to the
  static/online results:
      text   -> UTF-8 bytes            (rate label: bpc)
      image  -> W*H*3 RGB sub-pixels    (rate label: bpsp)
      audio  -> 8 kHz / 8-bit PCM samples (rate label: bps)
  Traditional codecs compress the WHOLE medium (unchunked — their real, standard use;
  matches how the base paper reports them); the model path chunks. The reported
  content-relative rate is over the identical content unit either way.
* **torch-free and HF-free**: reads local files only (normalized part-*.txt for text; the
  torch-free utils.img_utils / utils.audio_utils loaders for image/audio). None of these
  codecs touch Hugging Face, so the hf-mirror constraint does not apply here.
* **Pure-Python where possible** (gzip, LZMA2, PNG, WebP, FLAC) so it runs out-of-the-box
  in the `olmc` env; the three specialised binaries (cjxl, ofr, cmix) are shelled out to
  and gracefully skipped with an install hint if not on PATH.

Run from the repo root (flags mirror eval_online.py so you can reuse the same --data):

    python evaluation/traditional_baselines.py --modality text  --data data/text/normalized/qwen3/pile_of_law_eurlex --max-bytes 4000000
    python evaluation/traditional_baselines.py --modality image --data data/image/clic2024 --image-count 8 --image-crop 256
    python evaluation/traditional_baselines.py --modality audio --data data/audio/ljspeech --audio-clips 8
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import lzma
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_WAV_HEADER_BYTES = 44                       # canonical PCM WAV header (8-bit mono => 1 byte == 1 sample)
CONTENT_UNIT = {                             # modality -> (unit, rate label); matches eval_online
    "text":  ("byte",     "bpc"),
    "image": ("subpixel", "bpsp"),
    "audio": ("sample",   "bps"),
}
DEFAULT_DATA = {
    "text":  "data/text/normalized/qwen3/pile_of_law_eurlex",
    "image": "data/image/clic2024",
    "audio": "data/audio/ljspeech",
}
INSTALL_HINT = {
    "cmix":      "build from github.com/byronknoll/cmix (very slow — test on a small --max-bytes first)",
    "jpeg-xl":   "conda install -c conda-forge libjxl   (provides the `cjxl` binary)",
    "optimfrog": "download the free CLI from losslessaudio.org and put `ofr` on PATH",
    "brotli":    "pip install brotli   (else this falls back to the `brotli` CLI)",
}


# ---------------------------------------------------------------------------
# Shell helper for the binary-only codecs (cjxl / ofr / cmix / brotli-CLI)
# ---------------------------------------------------------------------------

def _cli_size(exe, argv, data, in_ext, out_ext):
    """Write ``data`` to a temp input, run ``argv`` (``@in`` / ``@out`` substituted),
    return the compressed-file size in bytes.  ``None`` if the tool is not on PATH or the
    run produced no output (so the caller can skip it cleanly)."""
    if shutil.which(exe) is None:
        return None
    with TemporaryDirectory() as d:
        pin, pout = Path(d) / ("in" + in_ext), Path(d) / ("out" + out_ext)
        pin.write_bytes(data)
        cmd = [str(pout) if a == "@out" else str(pin) if a == "@in" else a for a in argv]
        try:
            subprocess.run(cmd, capture_output=True, check=False)
        except Exception:
            return None
        if pout.exists() and pout.stat().st_size:
            return pout.stat().st_size
        sib = pin.with_suffix(out_ext)          # tools that emit <input>.<ext> beside the input (ofr)
        if sib.exists() and sib.stat().st_size:
            return sib.stat().st_size
        return None


# ---------------------------------------------------------------------------
# Codecs — each fn(payload) -> compressed size in bytes, or None if unavailable
# ---------------------------------------------------------------------------

def _gzip(b):   return len(zlib.compress(b, 9))
def _lzma2(b):  return len(lzma.compress(b, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME))


def _brotli(b):
    try:
        import brotli
        return len(brotli.compress(b, quality=11))
    except ImportError:
        return _cli_size("brotli", ["brotli", "-q", "11", "-f", "-o", "@out", "@in"], b, ".bin", ".br")


def _cmix(b):
    return _cli_size("cmix", ["cmix", "-c", "@in", "@out"], b, ".bin", ".cmix")


def _png(im):
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True, compress_level=9); return buf.tell()


def _webp(im):
    buf = io.BytesIO(); im.save(buf, format="WEBP", lossless=True, quality=100, method=6); return buf.tell()


def _jpegxl(im):
    """Lossless JPEG-XL.  Prefer the Pillow plugin (pure-pip, in-memory, no binary on PATH);
    fall back to the `cjxl` CLI on a PNG if the plugin isn't installed."""
    try:
        import pillow_jxl  # noqa: F401  — pip install pillow-jxl-plugin; registers JXL with Pillow
        buf = io.BytesIO(); im.save(buf, format="JXL", lossless=True, effort=9); return buf.tell()
    except ImportError:
        buf = io.BytesIO(); im.save(buf, format="PNG")     # cjxl reads PNG; -d 0 = mathematically lossless
        return _cli_size("cjxl", ["cjxl", "@in", "@out", "-d", "0", "-e", "9"], buf.getvalue(), ".png", ".jxl")


def _flac(wav):
    """FLAC of the exact 8 kHz / 8-bit mono PCM (pure-Python via soundfile; fair 8-bit).

    Read the WAV back as float (soundfile handles the header and normalises the 8-bit
    levels to [-1, 1)); then re-encode at 8-bit FLAC (subtype PCM_S8) to a REAL temp file.
    Two footguns avoided: (a) don't hand a small-magnitude int array to sf.write —
    libsndfile rescales integer input by its full dtype range and would zero the signal;
    (b) don't write FLAC to a BytesIO — libsndfile's FLAC encoder must seek back to
    finalise the STREAMINFO header, which virtual (in-memory) IO can't do, silently
    yielding an 18-byte empty stream.
    """
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(wav), dtype="float32")
    with TemporaryDirectory() as d:
        p = os.path.join(d, "a.flac")
        sf.write(p, data, sr, format="FLAC", subtype="PCM_S8")
        return os.path.getsize(p)


def _optimfrog(wav):
    return _cli_size("ofr", ["ofr", "--encode", "--preset", "high", "--quiet", "@in"], wav, ".wav", ".ofr")


CODECS = {
    "text":  [("gzip", _gzip), ("brotli", _brotli), ("lzma2", _lzma2), ("cmix", _cmix)],
    "image": [("png", _png), ("webp", _webp), ("jpeg-xl", _jpegxl)],
    "audio": [("flac", _flac), ("optimfrog", _optimfrog)],
}


# ---------------------------------------------------------------------------
# Data loading  (identical items + denominators to eval_online.py)
# ---------------------------------------------------------------------------

def _load(modality, path, args):
    """Return ``(payloads, content_units)``.  ``payloads`` is the list of items each codec
    compresses (one whole-text blob / per-image PIL / per-clip 8k-8bit WAV bytes);
    ``content_units`` is the shared denominator (utf-8 bytes / sub-pixels / samples)."""
    if modality == "text":
        if os.path.isdir(path):
            raw = b"".join(Path(p).read_bytes()
                           for p in sorted(glob.glob(os.path.join(path, "part-*.txt"))))
        else:
            raw = Path(path).read_bytes()
        if args.max_bytes:
            raw = raw[:args.max_bytes]
        b = raw.decode("utf-8", errors="ignore").encode("utf-8")   # canonical bytes
        return [b], len(b)

    if modality == "image":
        from PIL import Image
        from utils.img_utils import _load_image_dir
        files = _load_image_dir(path, args.image_count) if os.path.isdir(path) else [path]
        imgs, subpx = [], 0
        for fp in files:
            im = Image.open(fp).convert("RGB")
            if args.image_crop:
                c = args.image_crop
                im = im.crop((0, 0, min(c, im.width), min(c, im.height)))
            subpx += im.width * im.height * 3
            imgs.append(im)
        return imgs, subpx

    if modality == "audio":
        from utils.audio_utils import audio_to_pydub_seg
        if os.path.isdir(path) or path.endswith(".parquet"):
            from utils.audio_utils import load_audio_samples
            samples = load_audio_samples(path, n=args.audio_clips)
            segs = [audio_to_pydub_seg(s) for s in samples[:args.audio_clips]]
        else:
            segs = [audio_to_pydub_seg(path)]
        wavs, n_samples = [], 0
        for seg in segs:
            seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(1)
            buf = io.BytesIO(); seg.export(buf, format="wav"); wav = buf.getvalue()
            wavs.append(wav)
            n_samples += len(wav) - _WAV_HEADER_BYTES         # 8-bit mono => 1 PCM byte == 1 sample
        return wavs, n_samples

    raise NotImplementedError(modality)


# ---------------------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------------------

def _run(modality, payloads, content_units):
    unit, rate_label = CONTENT_UNIT[modality]
    print(f"\n  {'codec':<12}{'size':>13}{'ratio':>10}{rate_label:>10}")
    print("  " + "-" * 43)
    results = []
    for name, fn in CODECS[modality]:
        try:
            comp, ok = 0, True
            for item in payloads:
                s = fn(item)
                if s is None:
                    ok = False
                    break
                comp += s
        except Exception as e:                       # one broken codec must not kill the rest
            print(f"  {name:<12}{'error':>13}   ({type(e).__name__}: {e})")
            results.append({"codec": name, "status": "error", "error": f"{type(e).__name__}: {e}"})
            continue
        if not ok or content_units == 0:
            print(f"  {name:<12}{'skipped':>13}   ({INSTALL_HINT.get(name, 'unavailable')})")
            results.append({"codec": name, "status": "skipped"})
            continue
        ratio = content_units / comp
        rate = comp * 8 / content_units
        print(f"  {name:<12}{comp:>12}B{ratio:>9.3f}x{rate:>10.4f}")
        results.append({"codec": name, "comp": comp, "ratio": round(ratio, 4),
                        rate_label: round(rate, 4)})
    print("  " + "-" * 43)
    print(f"  (ratio = raw {unit}s / compressed;  {rate_label} = compressed bits / {unit})")
    return results


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Traditional lossless compression baselines")
    p.add_argument("--modality", required=True, choices=["text", "image", "audio"])
    p.add_argument("--data", default=None, help="default: per-modality (see DEFAULT_DATA)")
    p.add_argument("--max-bytes", type=int, default=None, help="text: cap input bytes")
    p.add_argument("--image-count", type=int, default=1, help="image: # images from a dir")
    p.add_argument("--image-crop", type=int, default=0, help="image: crop to NxN (0=full)")
    p.add_argument("--audio-clips", type=int, default=1, help="audio: # clips from a dataset")
    p.add_argument("--chunk-ms", type=int, default=1000, help="(accepted for CLI parity; unused)")
    p.add_argument("--json", default=None, metavar="PATH", help="also write results as JSON")
    args = p.parse_args()

    if args.data is None:
        args.data = DEFAULT_DATA[args.modality]
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"--data not found: {args.data}")

    print(f"{'=' * 45}\n  TRADITIONAL BASELINES  ({args.modality})\n{'=' * 45}")
    print(f"  data: {args.data}")
    payloads, content_units = _load(args.modality, args.data, args)
    unit = CONTENT_UNIT[args.modality][0]
    n_items = len(payloads) if args.modality != "text" else 1
    print(f"  original: {content_units} {unit}s" +
          ("" if args.modality == "text" else f"  ({n_items} items)"))
    results = _run(args.modality, payloads, content_units)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "content_units": content_units, "results": results},
                      f, indent=2, ensure_ascii=False)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
