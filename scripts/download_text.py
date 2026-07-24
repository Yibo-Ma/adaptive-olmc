#!/usr/bin/env python3
"""Download TEXT datasets into ``data/text/raw/<key>/`` (part-*.txt + manifest.jsonl).

Only license-clean, verified-downloadable datasets.  Direct-URL / GitHub sources need
no Hugging Face; HF sources work through hf-mirror.  Datasets whose loading *script*
hardcodes huggingface.co URLs (pile-of-law, edgar) are fetched straight from the mirror
as ``.jsonl(.xz)`` via the ``hf_lines`` handler; tabular datasets (earnings parquet,
clinical-note CSVs) are fetched as ``.parquet`` / ``.csv`` straight from the mirror via the
``hf_table`` handler — both bypass ``load_dataset``'s repo-tree listing, which hits
huggingface.co and fails in a mirror-only env.  Run from the repo root.

    python scripts/download_text.py --list
    python scripts/download_text.py --dataset enwik9 --limit 20MB
    python scripts/download_text.py --dataset pile_of_law_eurlex medal --limit 50MB
    python scripts/download_text.py --all

``--limit`` = bytes per dataset.  Resumable / append-extend; refuses to overwrite
externally-provided dirs (no _progress.json) unless ``--force``.
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    DATA_ROOT, http_download, human_bytes, load_progress, read_zip_member,
    run_download_cli, save_progress, stream_hf_lines,
)

RAW = DATA_ROOT / "text" / "raw"
SHARD_BYTES = 50 << 20
DEFAULT_LIMIT = 20 << 20

LOGHUB_SYSTEMS = ["HDFS", "BGL", "Thunderbird", "Spirit", "Windows", "Linux",
                  "Android", "Apache", "OpenStack", "HPC", "Hadoop", "Zookeeper",
                  "Mac", "OpenSSH", "Proxifier", "HealthApp"]

DATASETS = {
    # --- direct URL / GitHub (no HF needed) ---
    "enwik8":  dict(kind="url_zip", url="http://mattmahoney.net/dc/enwik8.zip",
                    member="enwik8", license="CC-BY-SA (Wikipedia)", gain="low"),
    "enwik9":  dict(kind="url_zip", url="http://mattmahoney.net/dc/enwik9.zip",
                    member="enwik9", license="CC-BY-SA (Wikipedia)", gain="low"),
    "text8":   dict(kind="url_zip", url="http://mattmahoney.net/dc/text8.zip",
                    member="text8", license="CC-BY-SA (Wikipedia)", gain="low"),
    "silesia": dict(kind="github_members",
                    base="https://raw.githubusercontent.com/MiloszKrajewski/SilesiaCorpus/master",
                    members=["dickens", "webster", "reymont", "samba", "xml"],
                    license="public benchmark", gain="med"),
    "loghub":  dict(kind="github_logs",
                    base="https://raw.githubusercontent.com/logpai/loghub/master",
                    systems=LOGHUB_SYSTEMS, license="research-free", gain="high",
                    note="2k-line samples; full sets on Zenodo (logpai/loghub)"),
    "enron":   dict(kind="url_tar_text",
                    url="https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
                    license="public", gain="med"),
    # --- HF straight-from-mirror .jsonl(.xz) (loading script hardcodes hf.co) ---
    "pile_of_law_eurlex": dict(kind="hf_lines", repo="pile-of-law/pile-of-law",
                               files=["data/train.eurlex.jsonl.xz"], fields=["text"],
                               license="CC-BY-NC-SA (non-commercial)", gain="high"),
    "atticus_contracts":  dict(kind="hf_lines", repo="pile-of-law/pile-of-law",
                               files=[f"data/train.atticus_contracts.{i}.jsonl.xz" for i in range(5)],
                               fields=["text"], license="CC-BY 4.0", gain="high"),
    "edgar_corpus":       dict(kind="hf_lines", repo="eloukas/edgar-corpus",
                               files=["2020/train.jsonl"], fields=None,
                               license="public (SEC)", gain="high"),
    "beancounter":        dict(kind="hf_lines", repo="bradfordlevy/BeanCounter",
                               files=["deduped/bc-001-of-220.jsonl.gz"], fields=["text"],
                               license="ODC-By (SEC EDGAR)", gain="high",
                               note="NeurIPS'24 D&B; SEC EDGAR business text 1996-2023 (deduped 111B config). "
                                    ".jsonl.gz via hf_lines (/resolve/ gzip stream, byte-capped -> stops mid-shard; "
                                    "one shard is ~300MB but --limit halts early). Column=text. More citable "
                                    "finance corpus than edgar_corpus -> can be the finance main."),
    # --- HF via load_dataset (work through hf-mirror) ---
    "codesearchnet":      dict(kind="hf", hf_id="code_search_net", config="python",
                               split="train", fields=["whole_func_string"],
                               license="MIT (code: per-file OSS)", gain="high"),
    "medal":              dict(kind="hf", hf_id="McGill-NLP/medal", config=None, split="train",
                               fields=["text"], license="MIT + NLM terms", gain="med"),
    "hupd":               dict(kind="hf", hf_id="HUPD/hupd", config="sample", split="train",
                               fields=["title", "abstract", "claims", "background", "description"],
                               streaming=False, load_kwargs=dict(trust_remote_code=True, uniform_split=True),
                               license="CC-BY 4.0 (USPTO)", gain="high"),
    # --- HF table (parquet/csv) straight-from-mirror (/resolve/ direct file; no load_dataset) ---
    "earnings_calls":     dict(kind="hf_table", repo="kurry/sp500_earnings_transcripts",
                               files=["parquet_files/part-0.parquet"], fields=None,
                               license="MIT (dataset)", gain="high",
                               note="S&P500 earnings-call transcripts (high delta: ritualized format). "
                                    "Single ~1.8GB parquet — downloads fully then byte-caps; fields=None "
                                    "auto-picks the transcript column."),
    "pmc_patients":       dict(kind="hf_table", repo="zhengyun21/PMC-Patients",
                               files=["PMC-Patients.csv"], fields=["patient"],
                               license="CC-BY-NC-SA 4.0 (non-commercial)", gain="high",
                               note="REAL patient case reports from PMC (NeurIPS'23 D&B). ~167k summaries, "
                                    "homogeneous clinical narrative -> far bigger delta than medal's "
                                    "heterogeneous abstracts. Column=patient. Ungated CSV via /resolve/. "
                                    "RECOMMENDED medical replacement for medal."),
    "asclepius_notes":    dict(kind="hf_table", repo="starmpcc/Asclepius-Synthetic-Clinical-Notes",
                               files=["synthetic.csv"], fields=["note"],
                               license="CC-BY-NC-SA 4.0 (non-commercial)", gain="high",
                               note="157k SYNTHETIC discharge summaries (GPT-3.5 from PMC-Patients). Most "
                                    "templated -> likely the biggest delta, but synthetic (representativeness "
                                    "caveat). Column=note. Alternative to pmc_patients."),
    "codeparrot_python":  dict(kind="hf_lines", repo="codeparrot/codeparrot-clean",
                               files=["file-000000000001.json.gz"], fields=["content"],
                               license="permissive OSS (per-file)", gain="high",
                               note="Python source (CodeParrot-clean, dedup). Ungated; JSONL.gz at repo root, "
                                    "column=content, fetched via hf_lines (/resolve/ gzip line-stream) — the "
                                    "only mechanism that works mirror-only. 54 shards file-0000000000NN.json.gz "
                                    "(~240MB each); add more to `files` for more data. Replaces the dead "
                                    "bigcode/the-stack (gated 403) and codeparrot/github-code (load_dataset "
                                    "hits huggingface.co/api/tree)."),
    # --- NIH direct FTP (no HF): PMC Open Access commercial-use full text ---
    "pmc_oa":             dict(kind="url_tar_text",
                               url="https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/"
                                   "oa_comm_txt.PMC000xxxxxx.baseline.2024-12-18.tar.gz",
                               license="CC0/CC-BY (PMC OA commercial-use)", gain="high",
                               note="DEAD-END mirror-only (kept for record): NIH FTP host is reachable but "
                                    "directory autoindex is OFF (can't discover the dated tarball name), and "
                                    "the HF pmc/open_access datasets are script-only (fetch from NIH at load -> "
                                    "load_dataset dies on the mirror). To use: obtain the exact current tarball "
                                    "name on a networked machine and paste it into url. Otherwise keep `medal` "
                                    "for medical, or invest in MIMIC-IV-Note (credentialed, homogeneous -> big delta)."),
}


# ---------------------------------------------------------------------------
# Shard writer (shared by the handlers)
# ---------------------------------------------------------------------------

class Sharder:
    """Append text blobs into part-*.txt shards (<= SHARD_BYTES) with a manifest."""

    def __init__(self, out: Path, key: str):
        self.out, self.key = out, key
        prog = load_progress(out)
        self.rows = prog.get("rows", 0)
        self.bytes = prog.get("bytes", 0)
        self.shard = prog.get("shard", 0)
        out.mkdir(parents=True, exist_ok=True)
        self.man = open(out / "manifest.jsonl", "a", encoding="utf-8")
        self.part = open(out / f"part-{self.shard:05d}.txt", "a", encoding="utf-8")
        self.sz = (out / f"part-{self.shard:05d}.txt").stat().st_size

    def add(self, text: str, **meta) -> None:
        if self.sz >= SHARD_BYTES:
            self.part.close(); self.shard += 1; self.sz = 0
            self.part = open(self.out / f"part-{self.shard:05d}.txt", "a", encoding="utf-8")
        blob = text + "\n"
        nb = len(blob.encode("utf-8"))
        self.part.write(blob)
        rec = {"dataset": self.key, "row": self.rows, "shard": self.shard, "bytes": nb}
        rec.update(meta)
        self.man.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.rows += 1; self.bytes += nb; self.sz += nb

    def close(self, **prog) -> None:
        self.part.close(); self.man.close()
        save_progress(self.out, rows=self.rows, bytes=self.bytes, shard=self.shard, **prog)


def _row_text(row: dict, fields) -> str:
    if fields:
        parts = [str(row[f]) for f in fields if row.get(f) not in (None, "")]
    else:
        parts = [v for v in row.values() if isinstance(v, str) and v]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Handlers — each is handler(key, spec, out, limit)
# ---------------------------------------------------------------------------

def dl_hf(key, spec, out, limit):
    """Stream (or download) an HF dataset via load_dataset, sharding rows up to ``limit`` bytes."""
    from datasets import load_dataset
    prog = load_progress(out)
    if prog.get("bytes", 0) >= limit:
        print(f"  [{key}] already {human_bytes(prog['bytes'])} (>= limit)"); return
    streaming = spec.get("streaming", True)
    print(f"  [{key}] {'stream' if streaming else 'download'} {spec['hf_id']} "
          f"({spec.get('config') or 'default'}) -> {human_bytes(limit)}")
    # script-based datasets (medal, hupd, ...) need trust_remote_code on datasets>=2.16;
    # harmless for the parquet-native ones.  spec may override.
    load_kwargs = {"trust_remote_code": True, **spec.get("load_kwargs", {})}
    ds = load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"],
                      streaming=streaming, **load_kwargs)
    sh = Sharder(out, key)
    start = sh.rows
    rows = ds if streaming else (ds[i] for i in range(len(ds)))
    seen = 0
    for row in rows:
        if seen < start:
            seen += 1; continue
        seen += 1
        t = _row_text(row, spec.get("fields"))
        if t:
            sh.add(t, source=spec["hf_id"], config=spec.get("config"))
        if sh.bytes >= limit:
            break
    sh.close(source=spec["hf_id"], gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({human_bytes(sh.bytes)}, {sh.rows} docs)")
    if sh.bytes == 0:
        raise RuntimeError(
            "got 0 docs — this dataset's loader likely fetches from huggingface.co "
            "directly (bypassing the mirror), so it can't work in a mirror-only env")


def dl_hf_lines(key, spec, out, limit):
    """Fetch line-based HF data files (.jsonl / .jsonl.gz / .jsonl.xz) straight from the
    mirror, streaming + decompressing so ``--limit`` caps the download.  Bypasses the
    dataset's loading script, which hardcodes huggingface.co URLs (HF_ENDPOINT can't
    rewrite those, so load_dataset fails / returns empty in a mirror-only env)."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    out.mkdir(parents=True, exist_ok=True)
    if load_progress(out).get("bytes", 0) >= limit:
        print(f"  [{key}] already {human_bytes(load_progress(out)['bytes'])} (>= limit)"); return

    # fresh parse each run (streaming caps the download, so re-running to extend is cheap)
    for p in out.glob("part-*.txt"):
        p.unlink()
    (out / "manifest.jsonl").unlink(missing_ok=True)
    save_progress(out, rows=0, bytes=0, shard=0)

    sh = Sharder(out, key)
    for rel in spec["files"]:
        if sh.bytes >= limit:
            break
        url = f"{endpoint}/datasets/{spec['repo']}/resolve/main/{rel}"
        print(f"  [{key}] {url}")
        for line in stream_hf_lines(url):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = _row_text(rec, spec.get("fields"))
            if t:
                sh.add(t, source=spec["repo"], config=key)
            if sh.bytes >= limit:
                break
    sh.close(source=spec["repo"], gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({human_bytes(sh.bytes)}, {sh.rows} docs)")
    if sh.bytes == 0:
        raise RuntimeError("got 0 docs — the mirror data-file URL was unreachable")


def dl_hf_table(key, spec, out, limit):
    """Fetch tabular data file(s) (.parquet or .csv) straight from the mirror's
    ``/resolve/`` path and shard a text column up to ``limit`` bytes.  Direct file URL (no
    ``/api`` repo-tree listing), so it works on mirror-only clusters where
    ``load_dataset(streaming=True)`` fails (see the dl_hf caveat).  Rows are streamed in
    batches (parquet row-batches / csv chunks) so a large file is neither loaded whole into
    RAM nor scanned past the byte cap.  ``fields`` selects the text column(s); if None, the
    largest string column of the first batch is auto-picked."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    rev = spec.get("revision", "main")
    out.mkdir(parents=True, exist_ok=True)
    if load_progress(out).get("bytes", 0) >= limit:
        print(f"  [{key}] already {human_bytes(load_progress(out)['bytes'])} (>= limit)"); return

    # fresh parse each run (a byte cap makes re-running to extend cheap)
    for p in out.glob("part-*.txt"):
        p.unlink()
    (out / "manifest.jsonl").unlink(missing_ok=True)
    save_progress(out, rows=0, bytes=0, shard=0)

    sh = Sharder(out, key)
    fields = spec.get("fields")
    for rel in spec["files"]:
        if sh.bytes >= limit:
            break
        url = f"{endpoint}/datasets/{spec['repo']}/resolve/{rev}/{rel}"
        tmp = out / ("_table" + Path(rel).suffix)
        print(f"  [{key}] {url}")
        http_download(url, tmp, desc=Path(rel).name)
        if rel.lower().endswith(".csv"):
            import pandas as pd
            batches = (chunk.to_dict("records")
                       for chunk in pd.read_csv(tmp, chunksize=2000, dtype=str, keep_default_na=False))
        else:                                            # parquet, streamed by row-batch
            import pyarrow.parquet as pq
            batches = (b.to_pylist() for b in pq.ParquetFile(tmp).iter_batches(batch_size=2000))
        done = False
        for rows in batches:
            if not fields and rows:                      # auto-pick the largest text column
                cols = {k for r in rows for k, v in r.items() if isinstance(v, str)}
                fields = ([max(cols, key=lambda c: sum(len(str(r.get(c, ""))) for r in rows))]
                          if cols else [])
            for row in rows:
                t = _row_text(row, fields)
                if t:
                    sh.add(t, source=spec["repo"], config=key)
                if sh.bytes >= limit:
                    done = True; break
            if done:
                break
        tmp.unlink(missing_ok=True)
    sh.close(source=spec["repo"], gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({human_bytes(sh.bytes)}, {sh.rows} docs)")
    if sh.bytes == 0:
        raise RuntimeError("got 0 docs — file URL unreachable (gated? need HF_TOKEN) "
                           "or the picked column had no text")


def dl_url_zip(key, spec, out, limit):
    """Download a .zip and keep the first ``limit`` bytes of one member as part-00000.txt."""
    out.mkdir(parents=True, exist_ok=True)
    src = out / ("_source" + Path(spec["url"]).suffix)
    print(f"  [{key}] {spec['url']}")
    http_download(spec["url"], src, desc=f"{key} src")
    raw = read_zip_member(src, spec.get("member"), limit)
    (out / "part-00000.txt").write_bytes(raw)
    save_progress(out, bytes=len(raw), source=spec["url"], gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({human_bytes(len(raw))})")


def dl_github_members(key, spec, out, limit):
    """Download each named .zip member from a GitHub raw base into its own shard."""
    total = 0
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as man:
        for i, m in enumerate(spec["members"]):
            z = out / f"_{m}.zip"
            print(f"  [{key}] {m}")
            http_download(f"{spec['base']}/{m}.zip", z, desc=m)
            data = read_zip_member(z, None)
            (out / f"part-{i:05d}.txt").write_bytes(data)
            man.write(json.dumps({"dataset": key, "member": m, "shard": i, "bytes": len(data)}) + "\n")
            z.unlink(); total += len(data)
    save_progress(out, bytes=total, members=spec["members"], gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({human_bytes(total)}, {len(spec['members'])} members)")


def dl_github_logs(key, spec, out, limit):
    """Fetch each system's 2k-line log sample from a GitHub raw base (skips 404s)."""
    sh = Sharder(out, key)
    done = set(load_progress(out).get("systems", []))
    for sysname in spec["systems"]:
        if sysname in done or sh.bytes >= limit:
            continue
        url = f"{spec['base']}/{sysname}/{sysname}_2k.log"
        tmp = out / "_tmp.log"
        try:
            http_download(url, tmp, desc=sysname)
            sh.add(tmp.read_text(encoding="utf-8", errors="ignore"), system=sysname)
            done.add(sysname); tmp.unlink(missing_ok=True)
        except Exception as e:
            print(f"  [{key}] {sysname} skipped ({type(e).__name__})")
    sh.close(systems=sorted(done), gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({human_bytes(sh.bytes)}, {len(done)} systems)")


def dl_url_tar_text(key, spec, out, limit):
    """Download a large tar of text files, then shard members up to ``limit`` bytes."""
    out.mkdir(parents=True, exist_ok=True)
    src = out / ("_source" + "".join(Path(spec["url"]).suffixes[-2:]))
    print(f"  [{key}] {spec['url']} (large; downloads then extracts to {human_bytes(limit)})")
    http_download(spec["url"], src, desc=f"{key} src")
    sh = Sharder(out, key)
    mode = "r:gz" if str(src).endswith((".gz", ".tgz")) else ("r:bz2" if str(src).endswith(".bz2") else "r:*")
    with tarfile.open(src, mode) as tf:
        for m in tf:
            if sh.bytes >= limit:
                break
            if m.isfile():
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    txt = f.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if txt.strip():
                    sh.add(txt, path=m.name)
    sh.close(source=spec["url"], gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({human_bytes(sh.bytes)}, {sh.rows} files)")


HANDLERS = {"hf": dl_hf, "hf_lines": dl_hf_lines, "hf_table": dl_hf_table,
            "url_zip": dl_url_zip, "github_members": dl_github_members,
            "github_logs": dl_github_logs, "url_tar_text": dl_url_tar_text}


def main() -> int:
    return run_download_cli("text", RAW, RAW.parent, DATASETS, HANDLERS,
                            default_limit=DEFAULT_LIMIT, limit_kind="bytes")


if __name__ == "__main__":
    raise SystemExit(main())
