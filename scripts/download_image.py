#!/usr/bin/env python3
"""Download IMAGE datasets into ``data/image/<key>/`` as ``img_*.{png,tiff,...}``.

Only license-clean, verified-downloadable datasets.  Direct-URL archives (DIV2K,
CLIC, USC-SIPI, ...) need no Hugging Face; Kodak / EuroSAT / CIFAR-10 stream through
hf-mirror.  Run from the repo root.

    python scripts/download_image.py --list
    python scripts/download_image.py --dataset kodak
    python scripts/download_image.py --dataset clic2024 usc_textures --limit 40
    python scripts/download_image.py --all

``--limit`` = number of images per dataset.  For lossless-compression work prefer the
PNG/TIFF sets (kodak, div2k, clic2024, usc_textures); dtd / oxford_pet / eurosat are JPEG.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    DATA_ROOT, extract_media, http_download, run_download_cli, save_progress,
)

IMG = DATA_ROOT / "image"
IMG_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

DATASETS = {
    # --- lossless PNG/TIFF (use these for compression benchmarks) ---
    "kodak": dict(kind="hf", hf_id="msdkhairi/kodak", split="train", image_key="image",
                  license="free research (benchmark)", gain="low",
                  note="24 lossless 768x512 PNGs; via HF mirror because r0k.us IP-blocks some clusters"),
    "div2k": dict(kind="url_zip", url="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
                  license="academic research only (non-commercial)", gain="low"),
    "clic2024": dict(kind="url_zip",
                     url="https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip",
                     license="CLIC (research use)", gain="low",
                     note="CLIC professional validation — lossless PNG (the standard uncompressed benchmark)"),
    "usc_textures": dict(kind="url_zip", url="https://sipi.usc.edu/database/textures.zip",
                         license="USC-SIPI (research use)", gain="high",
                         note="USC-SIPI texture volume: lossless TIFF (Brodatz); classic compression benchmark, "
                              "homogeneous -> high online gain. Mostly grayscale (loaded as RGB)."),
    # --- JPEG / homogeneous (lossy source: not for lossless-ratio headline numbers) ---
    "dtd":   dict(kind="url_zip", url="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
                  license="research (Oxford VGG)", gain="high"),
    "eurosat": dict(kind="hf", hf_id="blanchon/EuroSAT_RGB", split="train", image_key="image",
                    license="MIT / Sentinel-2 open", gain="high"),
    "cifar10": dict(kind="hf", hf_id="uoft-cs/cifar10", split="train", image_key="img",
                    license="research (Krizhevsky)", gain="high",
                    note="32x32; pick a single class for max homogeneity (e.g. cats)"),
    "oxford_pet": dict(kind="url_zip",
                       url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
                       license="CC BY-SA 4.0", gain="high",
                       note="cats & dogs (~37 breeds); homogeneous pet imagery"),
}


def dl_url_zip(key, spec, out, limit):
    """Download an archive (zip/tar) fully, then extract the first ``limit`` images."""
    out.mkdir(parents=True, exist_ok=True)
    src = out / ("_source" + "".join(Path(spec["url"]).suffixes[-2:]))
    print(f"  [{key}] {spec['url']} (downloads archive, extracts {limit} images)")
    http_download(spec["url"], src, desc=f"{key} src")
    n = extract_media(src, out, limit, IMG_EXTS, "img")
    save_progress(out, files=n, source=spec["url"], gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({n} images)")


def dl_hf(key, spec, out, limit):
    """Stream an HF image dataset and save the first ``limit`` images as PNG."""
    from datasets import load_dataset
    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob("img_*.png")))
    if have >= limit:
        print(f"  [{key}] already {have} images (>= limit)"); return
    print(f"  [{key}] stream {spec['hf_id']} -> {limit} images")
    ds = load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"], streaming=True)
    idx = 0
    for row in ds:
        if idx < have:
            idx += 1; continue
        img = row[spec.get("image_key", "image")]
        if isinstance(img, dict) and img.get("bytes"):
            from PIL import Image
            img = Image.open(io.BytesIO(img["bytes"]))
        img.convert("RGB").save(out / f"img_{idx:06d}.png")
        idx += 1
        sys.stdout.write(f"\r    {idx}/{limit}"); sys.stdout.flush()
        if idx >= limit:
            break
    save_progress(out, files=idx, source=spec["hf_id"], gain=spec.get("gain"))
    print(f"\n  [{key}] -> {out}  ({idx} images)")


HANDLERS = {"url_zip": dl_url_zip, "hf": dl_hf}


def main() -> int:
    return run_download_cli("image", IMG, IMG, DATASETS, HANDLERS,
                            default_limit=24, limit_kind="count")


if __name__ == "__main__":
    raise SystemExit(main())
