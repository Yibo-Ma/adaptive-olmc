"""Shared helpers for the download / normalize scripts.

Centralises three things every script needs:

* **hf-mirror by default** — many environments cannot reach ``huggingface.co``.
  We point ``HF_ENDPOINT`` at https://hf-mirror.com unless the caller overrides
  it (``--no-mirror`` / ``HF_ENDPOINT`` already set / ``--hf-endpoint``).
* **resumable, length-controlled HTTP** — :func:`http_download` uses HTTP Range
  so an interrupted download resumes from the current file size, and asking for
  *more* bytes later simply continues appending (never re-downloads).
* **progress manifests** — small JSON sidecars that record how much of a dataset
  has already been materialised, so a second run with a larger ``--limit``
  extends instead of restarting.

All paths are resolved relative to the repo root, so every script is meant to be
run from the repo root (``python scripts/<name>.py``).
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
CKPT_ROOT = REPO_ROOT / "checkpoints"

DEFAULT_HF_MIRROR = "https://hf-mirror.com"


# --------------------------------------------------------------------------
# Hugging Face endpoint (mirror)
# --------------------------------------------------------------------------

def setup_hf_endpoint(use_mirror: bool = True, endpoint: Optional[str] = None) -> str:
    """Configure the HF endpoint for this process (and child libraries).

    Precedence: explicit ``endpoint`` > existing ``$HF_ENDPOINT`` > mirror/default.
    Returns the endpoint actually in effect.
    """
    if endpoint:
        chosen = endpoint
    elif os.environ.get("HF_ENDPOINT"):
        chosen = os.environ["HF_ENDPOINT"]
    elif use_mirror:
        chosen = DEFAULT_HF_MIRROR
    else:
        chosen = "https://huggingface.co"
    os.environ["HF_ENDPOINT"] = chosen
    return chosen


def _enable_fast_download() -> None:
    """Turn on hf_transfer (Rust multi-threaded HTTP) for every HF download unless the
    user set ``HF_HUB_ENABLE_HF_TRANSFER`` explicitly.  No-op if the package isn't
    installed — setting the flag without it makes huggingface_hub raise at download time.
    """
    if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
        return                                   # respect an explicit opt-in / opt-out
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


_enable_fast_download()                          # runs on import, before any HF library is loaded

# Slow mirror: give the metadata HEAD and file download more than huggingface_hub's 10s default.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


# --------------------------------------------------------------------------
# Sizes / pretty printing
# --------------------------------------------------------------------------

def human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024
    return f"{f:.1f}TB"


def parse_size(s) -> Optional[int]:
    """Parse '10MB' / '500k' / '1_000_000' / int -> bytes. None passes through."""
    if s is None:
        return None
    if isinstance(s, int):
        return s
    s = str(s).strip().replace("_", "").upper()
    mult = 1
    for suf, m in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10),
                   ("T", 1 << 40), ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10), ("B", 1)):
        if s.endswith(suf):
            mult = m
            s = s[: -len(suf)]
            break
    return int(float(s) * mult)


# --------------------------------------------------------------------------
# Resumable HTTP download with optional byte cap
# --------------------------------------------------------------------------

def _session_with_retries(total: int = 4, backoff: float = 1.0):
    """A requests Session that retries connect / read / 5xx errors with backoff —
    academic and GitHub hosts are flaky, and a single hiccup shouldn't fail a dataset."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(total=total, connect=total, read=total, backoff_factor=backoff,
                  status_forcelist=(408, 429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET"]), raise_on_status=False)
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def http_download(
    url: str,
    dest: Path,
    limit_bytes: Optional[int] = None,
    chunk: int = 1 << 20,
    desc: str = "",
    retries: int = 4,
    timeout: int = 120,
) -> int:
    """Download ``url`` -> ``dest`` resumably; stop after ``limit_bytes`` total.

    Connect / read / 5xx errors are retried by the session; a mid-stream drop is
    retried here by resuming from the partial file (HTTP Range), so flaky links don't
    lose progress or fail the dataset.  Returns the final size of ``dest`` in bytes.
    """
    import requests
    import time

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = _session_with_retries(total=retries)

    for attempt in range(retries + 1):
        try:
            return _http_stream_once(session, url, dest, limit_bytes, chunk, desc, timeout)
        except (requests.RequestException, OSError) as e:
            have = dest.stat().st_size if dest.exists() else 0
            if limit_bytes is not None and have >= limit_bytes:
                return have
            if attempt >= retries:
                raise
            wait = min(2 ** attempt, 30)
            sys.stdout.write(f"\n  {desc or dest.name}: {type(e).__name__} — resume "
                             f"@ {human_bytes(have)}, retry {attempt + 1}/{retries} in {wait}s\n")
            sys.stdout.flush()
            time.sleep(wait)


def _http_stream_once(session, url, dest, limit_bytes, chunk, desc, timeout) -> int:
    cur = dest.stat().st_size if dest.exists() else 0
    if limit_bytes is not None and cur >= limit_bytes:
        return cur  # already have enough

    headers = {"Range": f"bytes={cur}-"} if cur else {}
    with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
        # 416 => our file is already complete; 200 w/ cur>0 => server ignored Range.
        if r.status_code == 416:
            return cur
        if cur and r.status_code == 200:
            dest.unlink(missing_ok=True)
            cur = 0
        r.raise_for_status()

        total = None
        clen = r.headers.get("Content-Length")
        if clen is not None:
            total = cur + int(clen)
        if limit_bytes is not None:
            total = min(total, limit_bytes) if total else limit_bytes

        written = cur
        with open(dest, "ab" if cur else "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if not block:
                    continue
                if limit_bytes is not None and written + len(block) > limit_bytes:
                    block = block[: limit_bytes - written]
                f.write(block)
                written += len(block)
                _progress(desc or dest.name, written, total)
                if limit_bytes is not None and written >= limit_bytes:
                    break
    sys.stdout.write("\n")
    return written


def _progress(name: str, done: int, total: Optional[int]) -> None:
    if total:
        pct = 100.0 * done / total
        sys.stdout.write(f"\r  {name}: {human_bytes(done)}/{human_bytes(total)} ({pct:4.1f}%)")
    else:
        sys.stdout.write(f"\r  {name}: {human_bytes(done)}")
    sys.stdout.flush()


def stream_hf_lines(url: str, timeout: int = 120):
    """Stream a line-based data file (``.jsonl`` / ``.jsonl.gz`` / ``.jsonl.xz``) over
    HTTP and yield its decoded lines *without* downloading the whole file — so a byte
    cap on the caller also caps the download.  Decompression is chosen by extension.

    Used to fetch HF data files straight from the mirror when a dataset's loading
    *script* hardcodes ``https://huggingface.co/...`` URLs (pile-of-law, edgar-corpus,
    ...), which ``HF_ENDPOINT`` cannot rewrite.  Redirects are followed, so it works
    whether the mirror serves the file directly or 308-redirects to the origin.
    """
    u = url.lower()
    if u.endswith(".xz"):
        import lzma
        dec = lzma.LZMADecompressor()
        feed, flush = dec.decompress, (lambda: b"")
    elif u.endswith(".gz"):
        import zlib
        dec = zlib.decompressobj(zlib.MAX_WBITS | 16)      # gzip
        feed, flush = dec.decompress, dec.flush
    else:
        feed, flush = (lambda b: b), (lambda: b"")         # plain text

    session = _session_with_retries()
    buf = b""
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            buf += feed(chunk)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("utf-8", errors="ignore")
    buf += flush()
    for line in buf.splitlines():
        if line:
            yield line.decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------
# Manifests (progress sidecars)
# --------------------------------------------------------------------------

def load_progress(d: Path) -> Dict:
    p = Path(d) / "_progress.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_progress(d: Path, **kv) -> None:
    p = Path(d) / "_progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = load_progress(d)
    cur.update(kv)
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")


def write_download_status(modality_root: Path, ok, failed, skipped=None) -> None:
    """Record per-modality download outcome so setup.py can report failures."""
    p = Path(modality_root) / "_download_status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"ok": list(ok), "failed": list(failed),
                             "skipped": list(skipped or [])}, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def read_download_status(modality_root: Path) -> Dict:
    p = Path(modality_root) / "_download_status.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# --------------------------------------------------------------------------
# Shared CLI + run loop for the per-modality downloaders
# --------------------------------------------------------------------------

def run_download_cli(modality: str, dataset_root: Path, status_root: Path,
                     datasets: Dict, handlers: Dict, default_limit,
                     limit_kind: str = "count") -> int:
    """Argument parsing + the download loop shared by download_{text,image,audio}.py.

    Each handler is called as ``handler(key, spec, out_dir, limit)`` and writes into
    ``out_dir``; any exception is caught and the dataset marked FAILED.  ``limit_kind``
    is ``"count"`` (an int, e.g. images/clips) or ``"bytes"`` (``parse_size``, e.g. 20MB).
    Writes ``_download_status.json`` under ``status_root`` and prints a one-line summary.
    """
    import argparse

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    unit = "bytes, e.g. 20MB" if limit_kind == "bytes" else "items"
    p = argparse.ArgumentParser(description=f"Download {modality.upper()} datasets -> {dataset_root}")
    p.add_argument("--dataset", nargs="*", help="dataset key(s) to download")
    p.add_argument("--all", action="store_true", help="download every dataset")
    p.add_argument("--limit", default=None, help=f"cap per dataset ({unit})")
    p.add_argument("--list", action="store_true", help="list datasets and exit")
    p.add_argument("--force", action="store_true", help="write even into a dir this tool did not create")
    p.add_argument("--no-mirror", action="store_true", help="use huggingface.co instead of hf-mirror")
    p.add_argument("--hf-endpoint", default=None, help="force a specific HF endpoint URL")
    args = p.parse_args()

    if args.list:
        for k, s in datasets.items():
            print(f"  {k:<20} gain={s.get('gain', '?'):<5} {s['kind']:<15} {s.get('license', '')}")
        return 0

    print(f"HF endpoint: {setup_hf_endpoint(use_mirror=not args.no_mirror, endpoint=args.hf_endpoint)}")
    if args.limit is None:
        limit = default_limit
    else:
        limit = parse_size(args.limit) if limit_kind == "bytes" else int(args.limit)

    keys = list(datasets) if args.all else (args.dataset or [])
    if not keys:
        p.error("specify --dataset KEY..., --all, or --list")

    ok, failed, skipped = [], [], []
    for k in keys:
        spec = datasets.get(k)
        if spec is None:
            print(f"  [{k}] unknown — see --list"); failed.append(k); continue
        out = Path(dataset_root) / k
        if not external_guard(out, args.force):
            skipped.append(k); continue
        try:
            handlers[spec["kind"]](k, spec, out, limit)
            ok.append(k)
        except Exception as e:
            print(f"  [{k}] FAILED: {type(e).__name__}: {str(e)[:140]}")
            failed.append(k)

    write_download_status(status_root, ok, failed, skipped)
    print(f"\n{modality.upper()}: {len(ok)} ok"
          + (f", {len(failed)} FAILED: {', '.join(failed)}" if failed else "")
          + (f", {len(skipped)} skipped: {', '.join(skipped)}" if skipped else ""))
    return 1 if failed else 0


# --------------------------------------------------------------------------
# Safety + archive helpers (shared by the three downloaders)
# --------------------------------------------------------------------------

def external_guard(out: Path, force: bool) -> bool:
    """Refuse to write into a dir holding data this tool did not create.

    Tool-made dirs carry a ``_progress.json`` marker; externally-provided data
    does not, so appending/overwriting would corrupt it.  ``--force`` overrides.
    """
    out = Path(out)
    if force or not out.exists():
        return True
    has_files = any(p.is_file() and p.name != "_progress.json" for p in out.rglob("*"))
    if has_files and not (out / "_progress.json").exists():
        print(f"  [{out.name}] REFUSING: {out} holds data not created by this tool "
              f"(no _progress.json). Use --force, or download into a clean dir.")
        return False
    return True


def _is_macos_junk(name: str) -> bool:
    """True for macOS zip cruft: the ``__MACOSX/`` folder and AppleDouble ``._``
    sidecars.  These carry image extensions but are not images (their content is a
    resource-fork header ``\\x00\\x05\\x16\\x07 ... Mac OS X ... ATTR``), so if the
    extension filter grabs them they land as tiny unopenable ``img_*.png`` files."""
    parts = name.replace("\\", "/").split("/")
    return "__MACOSX" in parts or parts[-1].startswith("._")


def read_zip_member(src: Path, member: Optional[str], limit_bytes: Optional[int] = None) -> bytes:
    """Read up to ``limit_bytes`` from one member of a zip file.

    With ``member=None`` the first *real* member is used, skipping macOS
    ``__MACOSX`` / ``._`` cruft that could otherwise sort ahead of the real file."""
    with zipfile.ZipFile(src) as zf:
        if member:
            name = member
        else:
            reals = [m for m in zf.namelist()
                     if not m.endswith("/") and not _is_macos_junk(m)]
            name = reals[0] if reals else zf.namelist()[0]
        with zf.open(name) as f:
            return f.read(limit_bytes) if limit_bytes else f.read()


def is_streamable_tar(url: str) -> bool:
    """True for tar(.gz/.bz2) — can be streamed and stopped early.

    zip cannot: its central directory is at the end, so extracting any member
    requires the whole file.
    """
    u = url.lower()
    return u.endswith((".tar", ".tgz", ".tar.gz", ".gz", ".tar.bz2", ".bz2"))


def stream_extract_tar(url: str, out_dir: Path, limit: int,
                       exts: Tuple[str, ...], prefix: str, desc: str = "") -> int:
    """Stream a tar(.gz/.bz2) over HTTP, extract up to ``limit`` files matching
    ``exts``, then stop — so only the bytes up to the limit-th file are downloaded.

    Files are renamed ``prefix_NNNNN.ext``.  Resumable: already-extracted files are
    skipped (a re-run re-streams from the start but only down to the new limit).
    Returns the number of files now present.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(out_dir.glob(f"{prefix}_*")))
    if have >= limit:
        return have

    u = url.lower()
    mode = "r|bz2" if u.endswith(".bz2") else ("r|gz" if u.endswith((".gz", ".tgz")) else "r|")
    n, seen = have, 0
    with _session_with_retries().get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True                       # undo any HTTP content-encoding
        with tarfile.open(fileobj=r.raw, mode=mode) as tf:
            for m in tf:
                if n >= limit:
                    break                                 # closes the connection -> stops downloading
                if not (m.isfile() and m.name.lower().endswith(exts)) or _is_macos_junk(m.name):
                    continue
                if seen < have:                           # resume: skip already-extracted
                    seen += 1
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                (out_dir / f"{prefix}_{n:05d}{Path(m.name).suffix.lower()}").write_bytes(f.read())
                n += 1
                sys.stdout.write(f"\r  {desc or prefix}: {n}/{limit} files")
                sys.stdout.flush()
    sys.stdout.write("\n")
    return n


def extract_media(archive: Path, out_dir: Path, limit: int,
                  exts: Tuple[str, ...], prefix: str) -> int:
    """Extract up to ``limit`` files matching ``exts`` from a zip/tar(.gz/.bz2)
    archive into ``out_dir``, renamed ``prefix_NNNNN.ext``.  Returns the count.

    Resumable: files already present (by index) are skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(out_dir.glob(f"{prefix}_*")))
    n = have
    al = str(archive).lower()
    if al.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            names = [m for m in zf.namelist()
                     if m.lower().endswith(exts) and not m.endswith("/")
                     and not _is_macos_junk(m)]
            for m in names[have:limit]:
                (out_dir / f"{prefix}_{n:05d}{Path(m).suffix.lower()}").write_bytes(zf.read(m))
                n += 1
    else:
        mode = "r:bz2" if al.endswith(".bz2") else ("r:gz" if al.endswith((".gz", ".tgz")) else "r:*")
        seen = 0
        with tarfile.open(archive, mode) as tf:
            for m in tf:
                if n >= limit:
                    break
                if m.isfile() and m.name.lower().endswith(exts) and not _is_macos_junk(m.name):
                    if seen < have:                  # skip already-extracted
                        seen += 1
                        continue
                    f = tf.extractfile(m)
                    if f is not None:
                        (out_dir / f"{prefix}_{n:05d}{Path(m.name).suffix.lower()}").write_bytes(f.read())
                        n += 1
    return n
