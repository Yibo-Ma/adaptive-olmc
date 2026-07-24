# Sweep harness (SLURM job array)

> **Role:** reference for the `sweep.py` tool. For the end-to-end experiment recipe
> (which grids to run in what order) see [RUNBOOK.md](RUNBOOK.md); for project setup and
> single runs see the [top-level README](../README.md).

**One parameter combination = one JSON config = one array task = one process.**
Each combo gets its own GPU and runs in parallel up to a `%K` cap — the efficient
way to sweep on a cluster. (The online run is sequential *within* a stream, so
cross-combo parallelism is the axis to exploit.)

Config-driven and dependency-free: the grid, the manifest, each run's
`config.json`, and `eval_online.py` all speak the **same flat schema**
(`eval_online.py`'s argument names), so there is no flag-translation layer.

## One tool, four verbs

```bash
# 1. expand a grid -> manifest (prints the exact sbatch line + run count)
python experiments/sweep.py gen --grid experiments/grids/round1/search_text.json

# 2. submit the array (range + %concurrency printed by step 1)
sbatch --array=0-242%16 experiments/run_array.sbatch results/sweeps/text/<group>

# 3. collect -> summary.csv (sorted, best first) + best_config.json
python experiments/sweep.py agg --group results/sweeps/text/<group>

# 4. list every sweep you have run, with its best rate
python experiments/sweep.py ls

# 5. plot per-chunk rate curves from any finished runs (no GPU; see evaluation/rate_curve.py)
python evaluation/rate_curve.py results/sweeps/text/<group>/runs/*/result.json --rolling 10
```

`run_array.sbatch` just calls `sweep.py run --index $SLURM_ARRAY_TASK_ID`.

## Output layout (tidy — one dir per group, one subdir per run)

```
results/sweeps/<modality>/<ts>_<name>/
  grid.json          # the spec that produced this group (archived)
  manifest.jsonl     # one line per run: {run_id, config}
  meta.json          # n_runs, git commit, created_at
  summary.csv        # aggregate output — the table you read
  best_config.json   # winner's config, ready for the lossless confirmation run
  runs/<id>/
    config.json      # the flat eval_online config (reproducible spec)
    status.json      # started/finished/failed + provenance (git, host, slurm ids)
    result.json      # {args, config, results} — metrics
    run.log          # full stdout/stderr
```

## Notes

- **Grids are plain JSON** using `eval_online.py`'s own arg names. `fixed` applies
  to every run; `grid` lists are swept as a full Cartesian product. `lora_alpha`
  defaults to `2*lora_r`. Recent-only adaptation is `train_on_all: false`.
- **Reproduce / confirm a single run** with the same schema, no harness:
  `python evaluation/eval_online.py --config results/sweeps/.../runs/0007/config.json`
  (explicit CLI flags still override the file).
- **Sweeps run compress-only** (`no_decompress: true`) — `bpc/bpsp/bps` only needs
  the compress side. `agg` writes `best_config.json` with `mode: both` and
  `no_decompress: false`, so confirming the winner losslessly is one command:
  `python evaluation/eval_online.py --config <group>/best_config.json`.
- **Resumable**: resubmitting the same array re-runs only rows missing
  `result.json` (`--skip-done`). `agg` is safe to run while the array is in flight.
- **`online_rate`** is the content-relative rate (bpc text / bpsp image / bps
  audio); lower is better. `delta_pct` is the static→online improvement.
- **Cluster setup**: edit `run_array.sbatch` once (`PROJECT_DIR`, `CONDA_ENV`,
  `#SBATCH` resources). Grid `model`/`data` paths point at this repo's layout —
  change them to your cluster's paths.
```
