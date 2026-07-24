"""OnlineTrainer: one deterministic LoRA training phase.

The encoder and decoder call ``train_phase`` with identical data, seeds and
phase index, so the weight updates are bit-identical at every interval boundary
and no adapter is ever transmitted.  The optimizer is created once and persisted
across phases (its momentum state must evolve identically on both ends).
"""
from __future__ import annotations

from typing import Callable, List

import torch
from torch.nn.utils import clip_grad_norm_

from compression.online.config import OnlineLearningConfig
from utils.determinism import set_seed, sync


def build_optimizer(model: torch.nn.Module, cfg: OnlineLearningConfig):
    """AdamW over LoRA-only (requires_grad) parameters, in deterministic order."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


class OnlineTrainer:

    def __init__(self, cfg: OnlineLearningConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def train_phase(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        windows: List[torch.Tensor],
        phase_idx: int,
        loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
        verbose: bool = True,
    ) -> None:
        # Per-phase seed: identical on encoder and decoder for this phase.
        set_seed(self.cfg.base_seed + 10000 * (phase_idx + 1))
        sync(self.device)
        model.train()

        if not windows:
            model.eval()
            return

        trainable = [p for p in model.parameters() if p.requires_grad]
        for epoch in range(self.cfg.epochs_per_train):
            epoch_loss, n = 0.0, 0
            for w in windows:
                loss = loss_fn(model, w)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(trainable, self.cfg.grad_clip)
                optimizer.step()
                epoch_loss += float(loss.item())
                n += 1
            if verbose:
                print(f"      [train] phase={phase_idx} "
                      f"epoch={epoch + 1}/{self.cfg.epochs_per_train} "
                      f"loss={epoch_loss / max(n, 1):.4f}")

        model.eval()
        sync(self.device)
