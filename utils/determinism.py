"""Determinism helpers for bit-exact online compression.

Losslessness in the online setting is bit-fragile: the decoder must reproduce
the encoder's logits *and* its LoRA weight updates exactly.  That requires the
whole stack to run deterministically (fp32, deterministic kernels, no TF32) and
every randomised step (LoRA init, per-phase training) to be seeded identically
on both ends.  These helpers centralise that contract.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def ensure_deterministic(seed: int = 42) -> None:
    """Pin every global knob that can perturb bit-exact computation.

    Call once at process start, before any model is created.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(seed)


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    """Block until all CUDA work is done (no-op on CPU) -- used for timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
