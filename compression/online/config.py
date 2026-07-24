"""Configuration for online LoRA-adaptive compression.

This config is modality-agnostic.  It is *not* stored verbatim in the archive —
only a hash of it is (see utils.online_archive) — so compression and
decompression must be invoked with identical settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class OnlineLearningConfig:
    """All tunable knobs for online LoRA adaptation."""

    # --- LoRA architecture ---
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0          # keep 0 for determinism

    # --- Training hyper-parameters ---
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_seq_len: int = 512
    epochs_per_train: int = 3

    # --- Online schedule ---
    train_interval: int = 4            # train (and batch) every N chunks
    train_on_recent_only: bool = True  # True: recent interval; False: all seen so far

    # --- Determinism ---
    base_seed: int = 42
