#!/usr/bin/env python3
"""Sweep harness — expand a grid, run one combo per SLURM array task, aggregate.

Config-driven and dependency-free.  One run == one JSON config == one array task
== one process.  The grid, the manifest, each run's ``config.json``, and
``eval_online.py`` all speak the SAME flat schema (eval_online's argument names),
so there is no flag-translation layer to drift.

    python experiments/sweep.py gen --grid experiments/grids/round1/search_text.json
    sbatch --array=0-N%K experiments/run_array.sbatch <group_dir>     # -> sweep.py run
    python experiments/sweep.py agg --group <group_dir>
    python experiments/sweep.py ls                                    # every sweep at a glance

Per-group layout::

    results/sweeps/<modality>/<ts>_<name>/
      grid.json  manifest.jsonl  meta.json  summary.csv  best_config.json
      runs/<id>/ config.json  status.json  result.json  run.log
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL = "evaluation/eval_online.py"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _load(path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_safe(path, default):
    """Read JSON, returning ``default`` on a missing/partial file (a run killed
    mid-write) so aggregation is safe while the array is still in flight."""
    try:
        return _load(path)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# gen — expand a grid into a manifest (one run config per line)
# ---------------------------------------------------------------------------

def _validate_spec(spec: Dict) -> None:
    """Fail fast (before submitting a whole array) on common grid-spec mistakes."""
    problems = []
    if not spec.get("modality"):
        problems.append("missing 'modality'")
    fixed, grid = spec.get("fixed", {}), spec.get("grid", {})
    if not grid:
        problems.append("'grid' is empty (nothing to sweep)")
    for k, v in grid.items():
        if not isinstance(v, list) or not v:
            problems.append(f"grid[{k!r}] must be a non-empty list")
    overlap = sorted(set(fixed) & set(grid))
    if overlap:
        problems.append(f"keys in both 'fixed' and 'grid' (grid would win): {overlap}")
    if problems:
        raise SystemExit("Invalid grid spec:\n  - " + "\n  - ".join(problems))


def build_configs(spec: Dict) -> List[Dict]:
    """Cartesian product of ``grid`` over ``fixed`` -> flat eval_online configs."""
    modality = spec["modality"]
    fixed = dict(spec.get("fixed", {}))
    grid = spec.get("grid", {})
    keys = sorted(grid)                                   # deterministic order
    runs: List[Dict] = []
    for i, values in enumerate(itertools.product(*(grid[k] for k in keys))):
        cfg = {**fixed, **dict(zip(keys, values)), "modality": modality}
        if "lora_r" in cfg and "lora_alpha" not in cfg:
            cfg["lora_alpha"] = 2 * int(cfg["lora_r"])    # effective scaling = alpha/r
        runs.append({"run_id": f"{i:04d}", "config": cfg})
    return runs


def cmd_gen(args) -> int:
    spec = _load(args.grid)
    _validate_spec(spec)
    modality = spec["modality"]
    name = args.name or spec.get("group_name") or Path(args.grid).stem
    runs = build_configs(spec)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = REPO_ROOT / args.results_root / modality / f"{ts}_{name}"
    (group / "runs").mkdir(parents=True, exist_ok=True)
    rel, n = os.path.relpath(group, REPO_ROOT), len(runs)
    submit = (f"sbatch --array=0-{n - 1}%{args.max_parallel} "
              f"experiments/run_array.sbatch {rel}")

    with open(group / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _dump(group / "grid.json", spec)
    _dump(group / "meta.json", {
        "created_at": _now(), "git_commit": _git_commit(), "modality": modality,
        "group_name": name, "grid_file": str(args.grid), "n_runs": n, "submit": submit,
    })

    print(f"  modality: {modality}   runs: {n}   group: {rel}\n")
    print("Submit the array (each combo = one independent job):\n")
    print(f"  {submit}\n")
    print(f"Then aggregate:\n\n  python experiments/sweep.py agg --group {rel}\n")
    return 0


# ---------------------------------------------------------------------------
# run — execute one manifest row (the body of each array task)
# ---------------------------------------------------------------------------

def _status(run_dir: Path, run_id: str, state: str, **extra) -> None:
    _dump(run_dir / "status.json", {
        "run_id": run_id, "status": state, "updated_at": _now(),
        "git_commit": _git_commit(), "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID",
                                       os.environ.get("SLURM_JOB_ID", "local")),
        "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        **extra,
    })


def cmd_run(args) -> int:
    group = Path(args.group)
    rows = _read_jsonl(args.manifest or group / "manifest.jsonl")
    if not (0 <= args.index < len(rows)):
        print(f"[error] index {args.index} out of range 0..{len(rows) - 1}", file=sys.stderr)
        return 2

    row = rows[args.index]
    run_id, config = row["run_id"], row["config"]
    run_dir = group / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_json = run_dir / "result.json"

    if args.skip_done and result_json.exists():
        print(f"[skip] {run_id} already done")
        return 0

    _dump(run_dir / "config.json", config)               # the reproducible spec
    _status(run_dir, run_id, "started")

    argv = [sys.executable, "-u", EVAL, "--config", str(run_dir / "config.json"),
            "--json", str(result_json)]
    print(f"[run {run_id}] " + " ".join(argv))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}      # live, complete run.log on kill
    with open(run_dir / "run.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()

    ok = rc == 0 and result_json.exists()              # rc 0 but no metrics == failed
    _status(run_dir, run_id, "finished" if ok else "failed", returncode=rc)
    print(f"[run {run_id}] exit {rc}")
    return rc


# ---------------------------------------------------------------------------
# agg — collect per-run results into summary.csv + best_config.json
# ---------------------------------------------------------------------------

def _mode(results: List[Dict], mode: str) -> Optional[Dict]:
    return next((r for r in results if r.get("mode") == mode), None)


def _collect(group: Path):
    rows, keys = [], set()
    for run_dir in sorted((group / "runs").glob("*")):
        cfg_p, st_p, res_p = (run_dir / "config.json", run_dir / "status.json",
                              run_dir / "result.json")
        config = _load_safe(cfg_p, None)
        if config is None:
            continue
        keys.update(config)
        row = {"run_id": run_dir.name,
               "status": _load_safe(st_p, {}).get("status", "?"),
               **config}
        if res_p.exists():
            results = _load_safe(res_p, {}).get("results", [])
            s, o = _mode(results, "static"), _mode(results, "online")
            if o:
                row.update(online_rate=round(o["bpsp"], 5), online_ratio=round(o["ratio"], 4),
                           online_bpb=round(o["bpb"], 5), online_comp_s=round(o["comp_s"], 1))
            if s:
                row["static_rate"] = round(s["bpsp"], 5)
            if s and o and s["bpsp"]:
                row["delta_pct"] = round((s["bpsp"] - o["bpsp"]) / s["bpsp"] * 100, 2)
        rows.append(row)
    return rows, sorted(keys)


def cmd_agg(args) -> int:
    group = Path(args.group)
    rows, param_keys = _collect(group)
    if not rows:
        print("No runs found under", group / "runs")
        return 1

    metrics = ["static_rate", "online_rate", "delta_pct",
               "online_ratio", "online_bpb", "online_comp_s"]
    fields = ["run_id", "status"] + param_keys + metrics
    with open(group / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    done = sorted((r for r in rows if r.get("online_rate") is not None),
                  key=lambda r: r["online_rate"])
    print(f"  {len(rows)} runs ({len(done)} done, {len(rows) - len(done)} pending/failed)")
    print(f"  summary -> {os.path.relpath(group / 'summary.csv')}")
    if not done:
        return 0

    best = done[0]
    # A ready-to-run config for the Stage-3 lossless confirmation of the winner.
    winner = {k: best[k] for k in param_keys if k in best}
    winner.update(mode="both", no_decompress=False)
    _dump(group / "best_config.json", winner)

    varied = [k for k in param_keys
              if len({str(r.get(k)) for r in rows}) > 1 and k != "modality"]
    cols = ["run_id", "online_rate", "static_rate", "delta_pct"] + varied
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in done[:args.top])) for c in cols}
    print(f"\n  best online_rate = {best['online_rate']}  (run {best['run_id']})")
    print(f"  confirm losslessly: python {EVAL} --config "
          f"{os.path.relpath(group / 'best_config.json')}\n")
    print("  " + "  ".join(c.ljust(width[c]) for c in cols))
    for r in done[:args.top]:
        print("  " + "  ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    return 0


# ---------------------------------------------------------------------------
# ls — one line per sweep group (managing many experiments)
# ---------------------------------------------------------------------------

def cmd_ls(args) -> int:
    root = REPO_ROOT / args.root
    groups = sorted(p.parent for p in root.glob("*/*/meta.json"))
    if not groups:
        print("No sweeps under", os.path.relpath(root))
        return 0
    print(f"  {'group':<52}{'runs':>6}{'done':>6}{'best_rate':>12}")
    print("  " + "-" * 74)
    for g in groups:
        meta = _load(g / "meta.json")
        best, done = "", 0
        csv_p = g / "summary.csv"
        if csv_p.exists():
            with open(csv_p, encoding="utf-8") as f:
                rates = [float(r["online_rate"]) for r in csv.DictReader(f)
                         if r.get("online_rate")]
            done = len(rates)
            best = f"{min(rates):.5f}" if rates else ""
        print(f"  {os.path.relpath(g, root):<52}{meta.get('n_runs', '?'):>6}{done:>6}{best:>12}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="expand a grid into a manifest")
    g.add_argument("--grid", required=True)
    g.add_argument("--name", default=None, help="group label (default: grid filename)")
    g.add_argument("--results-root", default="results/sweeps")
    g.add_argument("--max-parallel", type=int, default=16)
    g.set_defaults(func=cmd_gen)

    r = sub.add_parser("run", help="run one manifest row (= one array task)")
    r.add_argument("--group", required=True)
    r.add_argument("--index", type=int, required=True, help="= SLURM_ARRAY_TASK_ID")
    r.add_argument("--manifest", default=None)
    r.add_argument("--skip-done", action="store_true")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("agg", help="collect results into summary.csv + best_config.json")
    a.add_argument("--group", required=True)
    a.add_argument("--top", type=int, default=10)
    a.set_defaults(func=cmd_agg)

    ls = sub.add_parser("ls", help="list every sweep group and its best rate")
    ls.add_argument("--root", default="results/sweeps")
    ls.set_defaults(func=cmd_ls)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
