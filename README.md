# OnlineLMCompress

Lossless compression with large models, extended with an **online** twist: the model
keeps learning from the data it has already seen while it compresses.

Two modes share one pipeline:

- **Static** — a fixed pretrained model compresses chunks in batches (the team's base
  method, expressed through a modality backend).
- **Online** — every `train_interval` chunks, a LoRA adapter is fine-tuned on the
  just-coded chunks. The decoder replays the *exact* same init + training schedule on the
  chunks it decodes, so it reaches a bit-identical model state at every interval boundary
  and **no adapter is ever transmitted**. Losslessness relies on deterministic, fp32,
  bit-exact forward/backward passes (see `utils/determinism.py`) plus the padding trick in
  the decoder.

The better the model understands the data, the better it compresses — and online
adaptation lets it understand the current stream better as it goes.

## Documentation

Start here, then follow the map. Each doc has one job:

| Doc | Read it when you want to… |
|-----|---------------------------|
| **README.md** (this file) | install the environment, fetch models + data, and run a single compression / round-trip |
| [experiments/README.md](experiments/README.md) | learn the sweep tool — expand a grid, launch a SLURM array, aggregate `summary.csv` |
| [experiments/RUNBOOK.md](experiments/RUNBOOK.md) | run the full experiment program end to end: the staged search → freeze → scale → headline → ablations recipe |
| [experiments/BASELINES.md](experiments/BASELINES.md) | know which baselines to compare against and exactly how every rate/ratio number is defined |
| [bgpt/README.md](bgpt/README.md) | reference the vendored bGPT byte model (upstream; used for the image/audio backends) |

New here? Do the three steps in **[Setup](#setup-fresh-clone--ready-to-run)**, then run the first command in **[Usage](#usage)**.

## How it works

```
raw data ──to_chunks──▶ chunks ──interval──▶ encode_interval ──▶ arithmetic code
                                                  │
                                          (online) after each full
                                           interval: LoRA fine-tune
                                                  │
decoder replays init + training deterministically ▶ decode_interval (padding trick)
```

One modality-agnostic scheduler drives all three modalities through a single
`OnlineBackend` seam; only the model / tokenization / loss differ per modality:

| Modality | Backbone | Compressor | Unit fed to the coder |
|----------|----------|------------|------------------------|
| text  | causal LLM (e.g. Qwen2.5-0.5B) | `LLMCompressor` (token-level) | text → token chunks |
| image | bGPT (byte-level)             | `BGPTCompressor` (byte-level) | image → 32×32 BMP patches |
| audio | bGPT (byte-level)             | `BGPTCompressor` (byte-level) | audio → fixed-duration 8 kHz/8-bit WAV chunks |

The entropy backend (`arithmetic_coder/`, constriction range coder) and the base
compressors (`compression/{base,llm,bgpt}_compressor.py`) are shared, unchanged, with the
team's base repo.

## Layout

```
arithmetic_coder/        range coder (constriction) — shared with base
compression/
  base_compressor.py     shared encode kernel + padded-decode loop
  llm_compressor.py      token-level (text)
  bgpt_compressor.py     byte-level (image/audio)
  online/                the online layer
    static_compressor.py / online_compressor.py   the two schedulers
    base.py config.py trainer.py
    backends/            text_backend, image_backend, audio_backend (+ shared parents)
utils/                   determinism, archive, tokenization & I/O helpers
bgpt/                    vendored bGPT model (utils.py, config.py)
evaluation/
  eval_online.py         online/static round-trip eval (this project); --json records
                         per-chunk coded sizes so curves can be drawn later without a GPU
  rate_curve.py          per-chunk rate curves + cumulative saved bytes from --json files
                         (pure consumer: no model, no GPU; matplotlib optional)
  traditional_baselines.py   gzip/lzma2/brotli/cmix, PNG/WebP/JPEG-XL, FLAC/OptimFROG anchors
  eval_bgpt.py eval_llm.py image_codec_baselines.py   base benchmarks (kept for reference)
experiments/
  sweep.py               grid -> manifest -> SLURM array -> summary.csv (see experiments/README.md)
  grids/                 every sweep spec, one JSON per grid; the "note" field in each file
                         records the goal and findings driving it (round1..8, headline_*, rate_curve1)
  RUNBOOK.md             cluster workflow, BASELINES.md: baseline provenance
scripts/
  setup.py               one-command: models + data + normalize (orchestrates the below)
  download_models.py     weights  -> checkpoints/   (hf-mirror aware, resumable)
  download_text.py       text  datasets -> data/text/raw/  (byte-capped, resumable)
  download_image.py      image datasets -> data/image/     (count-capped, resumable)
  download_audio.py      audio datasets -> data/audio/     (count-capped, tar streamed)
  download_manual.py     special sources (chestxray14 / musdb18 / icbhi)
  normalize_text.py      tokenizer round-trip normalize: data/text/raw -> data/text/normalized/<tag>/
data/                    text/{raw,normalized}/  image/  audio/   (git-ignored; filled by scripts)
checkpoints/             Qwen2.5-0.5B, Qwen3-*, bgpt/weights-*.pth (git-ignored; filled by scripts)
```

## Setup (fresh clone → ready to run)

`data/` and `checkpoints/` ship empty. Everything runs from the repo root and defaults to
**hf-mirror.com** (pass `--no-mirror` or `--hf-endpoint URL` if your network prefers
`huggingface.co`).

**Environment** (conda-forge, one command; `nodefaults` avoids the Anaconda ToS prompt):

```bash
conda env create -f environment.yml     # python 3.11 + ffmpeg + pinned pip deps (see requirements.txt)
conda activate olmc
# GPU: torch from PyPI is CUDA-enabled on Linux; to pin a CUDA build:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Then, one command for models + data + normalize** (all-3-modality runnable):

```bash
python scripts/setup.py            # add --dry-run to preview, --no-mirror to skip the mirror
```

Or step by step:

```bash
# 1. model weights -> checkpoints/
python scripts/download_models.py --model qwen2.5-0.5b bgpt      # or --all / --list

# 2. datasets -> data/   (--limit = bytes for text, item count for image/audio)
python scripts/download_text.py  --dataset pile_of_law_eurlex --limit 20MB
python scripts/download_image.py --dataset kodak                 # 24 images
python scripts/download_audio.py --dataset librispeech --limit 50
python scripts/download_text.py  --list                          # per-modality catalog (each has --list)

# 3. normalize text so it survives the tokenizer round-trip (AFTER models — needs the tokenizer)
python scripts/normalize_text.py --dataset pile_of_law_eurlex    # or all datasets
```

Re-running a download with a larger `--limit` **extends** the existing data (HF streams
skip rows already written; URL files resume via HTTP Range) — it never re-downloads. The
downloader refuses to write into a dir holding externally-provided data (no
`_progress.json`) unless `--force`.

## Usage

Run from the repo root. `--mode both` runs static then online and verifies a lossless
round-trip with a fresh model reload.

```bash
# text — default data is normalized/<tokenizer-tag>/pile_of_law_eurlex (tag from --model)
python evaluation/eval_online.py --modality text  --mode both --max-bytes 80000
python evaluation/eval_online.py --modality text  --mode both --data data/text/normalized/qwen2.5/medal

# image — concatenate several images into one online stream (usc_textures = lossless TIFF texture)
python evaluation/eval_online.py --modality image --mode both --data data/image/usc_textures --image-count 8

# audio — concatenate several clips; each clip is chunked then the lists are concatenated
python evaluation/eval_online.py --modality audio --mode both --data data/audio/librispeech --audio-clips 8 --chunk-ms 1000

# text on another model family (ablation) — download, normalize for its tokenizer, then eval.
# smollm2-* is ungated, so hf-mirror serves it directly:
python scripts/download_models.py --model smollm2-1.7b
python scripts/normalize_text.py --models checkpoints/SmolLM2-1.7B
python evaluation/eval_online.py --modality text --mode both --model checkpoints/SmolLM2-1.7B \
    --data data/text/normalized/smollm2-1.7b/pile_of_law_eurlex

# llama-3.2-1b is GATED on Hugging Face. On hf.co: accept the license + set HF_TOKEN.
# In a mirror-only (hf.co-blocked) environment, fetch it token-free from ModelScope:
python scripts/download_models.py --model llama-3.2-1b --source modelscope   # pip install modelscope
python scripts/normalize_text.py --models checkpoints/Llama-3.2-1B
python evaluation/eval_online.py --modality text --mode both --model checkpoints/Llama-3.2-1B \
    --data data/text/normalized/llama-3.2-1b/pile_of_law_eurlex
```

Each run prints a comparison table with `bpb` (bits per coded byte) and a content-relative
rate — `bpsp` per sub-pixel (image), `bps` per 8 kHz sample (audio), `bpc` per byte (text),
the number to report — plus the static→online delta and `round-trip: OK`.

Key flags: `--train-interval`, `--epochs-per-train`, `--lora-r/--lora-alpha`, `--lr`,
`--chunk-size` (text tokens/chunk), `--image-count`, `--audio-clips`, `--chunk-ms`,
`--no-decompress` (compress-only), `--shuffle-seed` (permute the chunk *coding order*;
the decoder inverts it, so the round-trip stays lossless with zero side information —
the control experiment separating stream-locality gains from domain-level gains).

> **Determinism.** Online losslessness is bit-fragile: the decoder must reproduce the
> encoder's logits *and* its LoRA updates exactly. Compress and decompress must use the
> same settings on the same software/hardware stack; the archive stores a settings +
> environment fingerprint and refuses to decode on a mismatch.

## Analysis: per-chunk rate curves

`eval_online.py --json run.json` records each chunk's coded size alongside the
aggregate numbers. `rate_curve.py` turns those records into the prequential picture
(per-chunk rate static-vs-online with update boundaries marked, plus cumulative saved
bytes with the break-even chunk) — it never loads a model, so you can run many
experiments first and plot whichever you like afterwards:

```bash
python evaluation/eval_online.py --modality text --mode both --json results/run.json ...
python evaluation/rate_curve.py results/run.json                          # PNG next to the JSON
python evaluation/rate_curve.py results/sweeps/text/*/runs/*/result.json \
    --rolling 10 --out-dir results/figures                                # a whole sweep at once
```

## Sweeps

Hyperparameter grids run through `experiments/sweep.py` (grid JSON → manifest → SLURM
array or local loop → `summary.csv`); every run writes the same `--json` record, so
curves come free. See [experiments/README.md](experiments/README.md) for the four-verb
workflow and `experiments/grids/` for every spec used so far — each grid's `note` field
documents the question it answers and the findings that motivated it. The current text
experiments run Qwen3-0.6B-Base on `data/text/raw/{pile_of_law_eurlex,enwik9,beancounter}`
at chunk 2048 (see `grids/rate_curve1/` for the exact shared config).
