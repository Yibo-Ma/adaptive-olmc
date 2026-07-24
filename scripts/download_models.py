#!/usr/bin/env python3
"""Download model weights into ``checkpoints/`` (catalog-driven).

Run from the repo root.

Examples
--------
    python scripts/download_models.py --model qwen3-1.7b
    python scripts/download_models.py --model qwen2.5-0.5b qwen2.5-7b qwen3-1.7b qwen3-4b qwen3-8b bgpt
    python scripts/download_models.py --model qwen3-1.7b --fast   # aria2c multi-connection (fastest)
    python scripts/download_models.py --model llama-3.2-1b --source modelscope   # gated on hf; token-free here
    python scripts/download_models.py --all
    python scripts/download_models.py --list

Endpoint handling
-----------------
By default it tries **hf-mirror.com first and falls back to huggingface.co** (so it
works whether you are behind the Great Firewall *or* on a network where the mirror
only redirects). Force one with ``--no-mirror`` (hf.co only) or ``--hf-endpoint URL``.
Downloads are resumable, so an interrupted pull continues.

For **gated** HF repos (e.g. Llama) in a mirror-only environment, hf-mirror cannot
serve the files without routing your HF token through it; pass ``--source modelscope``
instead to fetch the model's ``ms_repo`` from ModelScope token-free (needs
``pip install modelscope``).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import CKPT_ROOT, DEFAULT_HF_MIRROR  # noqa: E402

HF_CO = "https://huggingface.co"

# Model registry (this script owns it, like download_text/image/audio own theirs).
MODELS = {
    "qwen2.5-0.5b": dict(source="hf_repo", repo="Qwen/Qwen2.5-0.5B",    dir="Qwen2.5-0.5B"),
    "qwen2.5-7b":   dict(source="hf_repo", repo="Qwen/Qwen2.5-7B",      dir="Qwen2.5-7B"),
    "qwen3-0.6b":   dict(source="hf_repo", repo="Qwen/Qwen3-0.6B-Base", dir="Qwen3-0.6B-Base"),
    "qwen3-1.7b":   dict(source="hf_repo", repo="Qwen/Qwen3-1.7B-Base", dir="Qwen3-1.7B-Base"),
    "qwen3-4b":     dict(source="hf_repo", repo="Qwen/Qwen3-4B-Base",   dir="Qwen3-4B-Base"),
    "qwen3-8b":     dict(source="hf_repo", repo="Qwen/Qwen3-8B-Base",   dir="Qwen3-8B-Base"),
    "bgpt": dict(
        source="hf_files", repo="sander-wood/bgpt", dir="bgpt",
        files=["weights-image.pth", "weights-audio.pth", "weights-text.pth"],
        note="if your team uses custom bGPT checkpoints, drop them in checkpoints/bgpt/"),
    # Model-family ablation (text).  SmolLM2 is ungated Apache-2.0, so hf-mirror serves it
    # directly in a mirror-only (China) environment; the 135M size is a fast pipeline smoke
    # test.  Llama is GATED — see the note: in a mirror-only environment prefer ModelScope
    # (China-native, token-free) over routing HF_TOKEN through a third-party mirror.
    "llama-3.2-1b": dict(source="hf_repo", repo="meta-llama/Llama-3.2-1B",   dir="Llama-3.2-1B",
                         ms_repo="LLM-Research/Llama-3.2-1B",
                         note="gated. hf.co: accept the license + set HF_TOKEN. mirror-only "
                              "(hf.co blocked): use --source modelscope (China-native, no token)"),
    "smollm2-1.7b": dict(source="hf_repo", repo="HuggingFaceTB/SmolLM2-1.7B", dir="SmolLM2-1.7B"),
    "smollm2-135m": dict(source="hf_repo", repo="HuggingFaceTB/SmolLM2-135M", dir="SmolLM2-135M"),
}


def _endpoints(args) -> list[str]:
    if args.hf_endpoint:
        return [args.hf_endpoint]
    if args.no_mirror:
        return [HF_CO]
    return [DEFAULT_HF_MIRROR, HF_CO]        # mirror first, then fall back to hf.co


def download_repo(spec: dict, endpoint: str) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=spec["repo"], local_dir=str(CKPT_ROOT / spec["dir"]),
        endpoint=endpoint,
        # skip the duplicate consolidated weight when sharded files exist
        ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
    )


def download_files(spec: dict, endpoint: str) -> None:
    from huggingface_hub import hf_hub_download
    dest = CKPT_ROOT / spec["dir"]
    dest.mkdir(parents=True, exist_ok=True)
    for fname in spec["files"]:
        hf_hub_download(repo_id=spec["repo"], filename=fname,
                        local_dir=str(dest), endpoint=endpoint)


def download_modelscope(spec: dict) -> None:
    """Download a model from ModelScope (Alibaba) instead of Hugging Face.

    The escape hatch for gated HF repos (e.g. Llama) in a mirror-only environment:
    ModelScope is reachable inside China without the GFW and serves these weights
    token-free, so no HF credential ever transits the third-party hf-mirror.  Needs
    ``pip install modelscope`` and a ``ms_repo`` id in the catalog entry.  Same
    ignore set as the HF path so the two produce an equivalent ``checkpoints/<dir>``.
    """
    from modelscope import snapshot_download
    snapshot_download(
        spec["ms_repo"], local_dir=str(CKPT_ROOT / spec["dir"]),
        ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
    )


def _aria2_get(url: str, dest) -> None:
    """Download one file with aria2c: multi-connection, resumable (-c)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aria2c", "-x16", "-s16", "-k1M", "-c", "--auto-file-renaming=false",
           "--console-log-level=warn", "-d", str(dest.parent), "-o", dest.name]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:                                    # gated repos (e.g. Llama) need auth on hf.co
        cmd += ["--header", f"Authorization: Bearer {token}"]
    subprocess.run([*cmd, url], check=True)


def _repo_files(spec: dict, endpoint: str) -> list[str]:
    """The files to fetch for a model: explicit list for hf_files, else the repo's
    file listing minus duplicate consolidated weights."""
    if spec["source"] == "hf_files":
        return list(spec["files"])
    from huggingface_hub import HfApi
    ignore = ("*.pth", "*.msgpack", "*.h5", "original/*")
    return [f for f in HfApi(endpoint=endpoint).list_repo_files(spec["repo"])
            if not any(fnmatch.fnmatch(f, pat) for pat in ignore)]


def download_aria2(spec: dict, endpoint: str) -> None:
    """Fast path: resolve the repo's file list, then pull each file with aria2c.
    Much faster than the Python client on large weights; needs ``aria2c`` on PATH."""
    dest = CKPT_ROOT / spec["dir"]
    for f in _repo_files(spec, endpoint):
        _aria2_get(f"{endpoint}/{spec['repo']}/resolve/main/{f}", dest / f)


def fetch_model(name: str, spec: dict, endpoints: list[str],
                fast: bool = False, source: str = "hf") -> bool:
    if source == "modelscope":
        ms_repo = spec.get("ms_repo")
        if not ms_repo:
            print(f"  [{name}] no ModelScope mirror in the catalog (no ms_repo); "
                  f"omit --source modelscope for this model, or add its ms_repo id")
            return False
        try:
            print(f"  [{name}] {ms_repo} via ModelScope -> checkpoints/{spec['dir']}")
            download_modelscope(spec)
            print(f"  [{name}] done")
            return True
        except ImportError:
            print(f"  [{name}] --source modelscope needs the package: pip install modelscope")
            return False
        except Exception as e:
            print(f"  [{name}] FAILED via ModelScope: {type(e).__name__}: {str(e)[:120]}")
            print(f"        some ModelScope repos require accepting terms once at "
                  f"modelscope.cn/models/{ms_repo}")
            return False

    use_aria = fast and shutil.which("aria2c") is not None
    if fast and not use_aria:
        print(f"  [{name}] --fast needs aria2c on PATH (conda install -c conda-forge aria2); "
              f"using the normal downloader")
    if use_aria:
        fn = download_aria2
    else:
        fn = download_repo if spec["source"] == "hf_repo" else download_files
    last = None
    for ep in endpoints:
        try:
            print(f"  [{name}] {spec['repo']} via {ep}"
                  f"{' (aria2c)' if use_aria else ''} -> checkpoints/{spec['dir']}")
            fn(spec, ep)
            print(f"  [{name}] done")
            return True
        except Exception as e:
            last = e
            tail = endpoints[-1]
            if ep != tail:
                print(f"  [{name}] {ep} failed ({type(e).__name__}); falling back ...")
    print(f"  [{name}] FAILED on all endpoints: {type(last).__name__}: {str(last)[:120]}")
    print(f"        China tip: pip install modelscope && "
          f"modelscope download --model {spec['repo']} --local_dir checkpoints/{spec['dir']}")
    if spec.get("note"):
        print(f"        {spec['note']}")
    return False


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Download model weights into checkpoints/")
    p.add_argument("--model", nargs="*", help="model key(s) from the catalog")
    p.add_argument("--all", action="store_true", help="download every model")
    p.add_argument("--list", action="store_true")
    p.add_argument("--no-mirror", action="store_true", help="use huggingface.co only")
    p.add_argument("--hf-endpoint", default=None, help="force a single endpoint URL")
    p.add_argument("--fast", action="store_true",
                   help="use aria2c multi-connection downloads (much faster; needs aria2c on PATH)")
    p.add_argument("--source", choices=["hf", "modelscope"], default="hf",
                   help="download host. 'modelscope' (needs `pip install modelscope`) fetches a "
                        "model's ms_repo token-free — the route for gated HF repos (Llama) in a "
                        "mirror-only environment where hf.co is blocked")
    args = p.parse_args()

    if args.list:
        for k, s in MODELS.items():
            print(f"  {k:<14} {s['repo']:<26} -> checkpoints/{s['dir']}")
        return 0

    endpoints = _endpoints(args)
    if args.source == "modelscope":
        print("source: ModelScope (token-free; hf endpoints unused)")
    else:
        print(f"endpoints (in order): {endpoints}")

    keys = list(MODELS) if args.all else (args.model or [])
    if not keys:
        p.error("specify --model KEY..., --all, or --list")

    ok = True
    for k in keys:
        s = MODELS.get(k)
        if s is None:
            print(f"  [{k}] unknown model key — see --list"); ok = False; continue
        ok = fetch_model(k, s, endpoints, args.fast, args.source) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
