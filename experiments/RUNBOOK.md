# Sweep runbook

> **Role:** the end-to-end experiment recipe — which grids to run, in what order, and how
> to read each result. For the `sweep.py` command reference see [README.md](README.md); for
> the baseline/metric definitions the headline table uses see [BASELINES.md](BASELINES.md).

Five stages, all driven by `sweep.py` (tool docs: `README.md`). Objective per run:
**`online_rate`** (bpc text / bpsp image / bps audio), lower is better; `delta_pct`
= static→online gain. Grids live in `grids/`, named by stage.

```
Stage 0   losslessness + determinism check  (not a sweep — must pass first)
Stage 1   hyperparameter search             search_{text,image,audio}.json
Stage 1.5 model-scaling (text)              models_text.json
Stage 2   amortization curve                clone a headline grid, swap axis -> size
Stage 3   headline main table               headline_{text,image,audio}.json
Stage 4   ablations                         clone a headline grid, swap axis -> one knob
```

Method in one line: **tune cheap on a dev dataset → freeze one config → model-scaling
→ size-scaling to the plateau → headline over the committed datasets (baseline / static /
online) → ablations.** Search is compress-only; losslessness is confirmed once (Stage 3).

## 0. One-time

Edit `run_array.sbatch` (`PROJECT_DIR`, `CONDA_SH`, `CONDA_ENV`, `#SBATCH` resources) for
your cluster. Data present now covers Stage 0/1/1.5; Stage 3 also needs `kodak`,
`usc_textures`, `librispeech` — download those in parallel while Stage 1 runs.

## Stage 0 — losslessness + determinism (must pass before any sweep)

Verifies bit-exact round-trip on this GPU stack. Each is **one line** (a `\` continuation
must have no trailing space).

```bash
python evaluation/eval_online.py --modality text --mode both --model checkpoints/Qwen3-4B-Base --data data/text/normalized/qwen3/pile_of_law_eurlex --max-bytes 30000 --chunk-size 512 --train-interval 4 --epochs-per-train 1
python evaluation/eval_online.py --modality image --mode both --data data/image/clic2024 --image-count 2 --image-crop 100 --train-interval 4 --epochs-per-train 1
python evaluation/eval_online.py --modality audio --mode both --data data/audio/ljspeech --audio-clips 3 --chunk-ms 1000 --train-interval 4 --epochs-per-train 1
```

Each must print `round-trip: OK` for **both** STATIC and ONLINE. A `FAILED` means the
deterministic stack differs (GPU/driver/lib) — fix before sweeping.

## Stage 1 — hyperparameter search (iterative rounds)

Stage 1 runs in ROUNDS: a broad round 1, then extend/refine rounds until the winner
stops moving. Grids live in `grids/round<N>/search_{text,image,audio}.json`.

Run one round (per modality; example = round 1, text):

```bash
python experiments/sweep.py gen --grid experiments/grids/round1/search_text.json --max-parallel 32
# -> prints the exact sbatch line + run count
sbatch --array=0-N%32 experiments/run_array.sbatch results/sweeps/text/<group>
python experiments/sweep.py ls                                        # progress, any time
python experiments/sweep.py agg --group results/sweeps/text/<group>   # summary.csv (sorted)
```

Repeat for image / audio; for the next round swap `round1/` → `round2/` (etc.).

**Round decision rule** (read `summary.csv`, look at the winning row):
- **A swept knob sits at an EDGE of its range** (best `lr` = the max, best
  `train_interval` = the min, …) → the optimum is outside the grid → **EXTEND that
  direction** next round.
- **A knob is INTERIOR** (winner not at either end) → it has converged → **narrow / fix** it.
- Keep the `target_modules` axis (attn vs attn+MLP) until one clearly wins, then fix it.
- **Converged = no knob is on an edge.** Then freeze the winner → Stage 1.5. Reaching
  this usually takes round 2–4 — that's expected, not a problem.

Each round's grid is kept under `grids/round<N>/`, and every run's exact spec is also
archived in its group's `grid.json`, so the whole tuning trajectory is reproducible.

## Stage 1.5 — model-scaling (text; picks the headline model)

Paste the Stage-1 text winner hyperparams into `models_text.json`'s `fixed`, then:

```bash
python experiments/sweep.py gen --grid experiments/grids/models_text.json
sbatch --array=0-2%3 experiments/run_array.sbatch results/sweeps/text/<group>
python experiments/sweep.py agg --group results/sweeps/text/<group>
```

Headline model = the best `online_rate` that is compute-feasible (bigger is usually
lower; the bpc-vs-model-size curve is itself a paper result).

## Freeze the config

Paste the winner hyperparams + chosen model into each `headline_{text,image,audio}.json`
`fixed` block. **These grids are now the single source of truth** cloned by Stage 2/4.

## Stage 2 — amortization curve (per modality)

Clone a headline grid, move ONE dataset into `fixed`, replace `grid` with the size ladder:

```bash
cp experiments/grids/headline_text.json experiments/grids/scaling_text.json
# edit scaling_text.json:  fixed.data = one report dataset;  delete fixed.max_bytes;
#   "grid": {"max_bytes": [1000000, 4000000, 16000000, 64000000, 256000000]}
python experiments/sweep.py gen --grid experiments/grids/scaling_text.json
sbatch --array=0-4%5 experiments/run_array.sbatch results/sweeps/text/<group>
python experiments/sweep.py agg --group results/sweeps/text/<group>
```

Plot `online_rate` & `static_rate` vs the size axis → the plateau = the headline size,
and where online overtakes static (crossover). Image: sweep `image_count`
[8,16,32,64,128] with `image_crop: 0`. Audio: sweep `audio_clips` [16,32,64,128,256].

## Stage 3 — headline main table

Set each `headline_*.json` `fixed.max_bytes` / count to the Stage-2 plateau, then:

```bash
python experiments/sweep.py gen --grid experiments/grids/headline_text.json
sbatch --array=0-3%4 experiments/run_array.sbatch results/sweeps/text/<group>
python experiments/sweep.py agg --group results/sweeps/text/<group>   # the paper table
```

Repeat for image (`--array=0-2`) and audio (`--array=0-3`). `summary.csv` gives, per
dataset, `static_rate` + `online_rate` + `delta_pct`. Two things live **outside** the sweep:

- **Traditional baselines** (text: bz2/brotli/zlib; audio: FLAC; image: PNG/JPEG-XL/WebP)
  at the same data sizes — standard tools / `evaluation/*baselines*` — as extra columns.
- **One lossless confirmation** (a real round-trip, not compress-only): agg already wrote
  `best_config.json` with `mode: both`, `no_decompress: false`, so:
  `python evaluation/eval_online.py --config results/sweeps/text/<group>/best_config.json`

## Stage 4 — ablations (one axis each)

Clone a headline grid, fix `data` to one dataset, sweep one knob:

```bash
cp experiments/grids/headline_text.json experiments/grids/ablation_text.json
# edit:  fixed.data = one dataset;  delete fixed.target_modules;
#   "grid": {"target_modules": ["q_proj,v_proj", "q_proj,k_proj,v_proj,o_proj",
#            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"]}
python experiments/sweep.py gen --grid experiments/grids/ablation_text.json
sbatch --array=0-2%3 experiments/run_array.sbatch results/sweeps/text/<group>
python experiments/sweep.py agg --group results/sweeps/text/<group>
```

Same pattern for `lora_r` (capacity), `train_interval` / `epochs_per_train`
(compute–ratio Pareto), `train_on_all: [false, true]` (recent vs all-seen), or a
hyperparameter-sensitivity grid (small ranges around the winner → shows the method is not
fragile, which defends reporting one fixed config everywhere).

## Ops cheatsheet

| Need | Command |
|---|---|
| List all sweeps + best rate | `python experiments/sweep.py ls` |
| Aggregate (safe mid-flight) | `python experiments/sweep.py agg --group <g>` |
| Resume failed/killed rows | re-run the same `sbatch --array=…` line (`--skip-done` skips finished) |
| Re-run only row 37 | `sbatch --array=37 experiments/run_array.sbatch <group>` |
| Reproduce one run locally | `python evaluation/eval_online.py --config <group>/runs/<id>/config.json` |
| Inspect a run | `<group>/runs/<id>/{config.json,status.json,result.json,run.log}` |
