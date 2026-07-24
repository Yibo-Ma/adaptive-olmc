"""OnlineCompressor: LoRA-adaptive chunked compression.

Compress chunk-by-chunk; after every full interval of ``train_interval`` chunks,
fine-tune a LoRA adapter on those (just-coded) chunks.  Weights are constant
within an interval, so its chunks share one forward batch.  The decoder replays
the identical init + training schedule on the *decoded* chunks, reaching a
bit-identical model state at every interval boundary, so no adapter is ever
transmitted.

The partial trailing interval is coded but not trained (mirrored on both ends).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

from compression.online.backends.base import ChunkUnit, OnlineBackend
from compression.online.base import _ChunkedCompressor
from compression.online.config import OnlineLearningConfig
from compression.online.trainer import OnlineTrainer, build_optimizer


class OnlineCompressor(_ChunkedCompressor):

    ROLE = "online"

    def __init__(
        self, backend: OnlineBackend, device: torch.device, cfg: OnlineLearningConfig,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        super().__init__(backend, device, shuffle_seed=shuffle_seed)
        self.cfg = cfg
        self.optimizer = None
        self.trainer = None

    def _settings(self) -> Dict:
        return asdict(self.cfg)

    def setup(self) -> None:
        self.backend.load_backbone()
        self.backend.attach_lora(self.cfg)          # before make_compressor: wrap PEFT model
        self.compressor = self.backend.make_compressor()
        self.optimizer = build_optimizer(self.backend.model, self.cfg)
        self.trainer = OnlineTrainer(self.cfg, self.device)

    # ------------------------------------------------------------------

    def _train(self, train_chunks: List[ChunkUnit], phase: int) -> None:
        windows = self.backend.build_training_windows(train_chunks, self.cfg)
        self.trainer.train_phase(
            self.backend.model, self.optimizer, windows, phase, self.backend.compute_loss,
        )

    def compress(self, raw: Any, framing: bytes = b"") -> bytes:
        chunks = self._prepare_chunks(raw)
        total_ob = self.backend.raw_size_bytes(raw)

        all_cds = []
        seen: List[ChunkUnit] = []
        phase = 0
        for group, is_tail in self._iter_intervals(chunks, self.cfg.train_interval):
            all_cds.extend(self.backend.encode_interval(self.compressor, group))
            seen.extend(group)
            if not is_tail:
                train_chunks = group if self.cfg.train_on_recent_only else seen
                self._train(train_chunks, phase)
                phase += 1

        return self._assemble_archive(self.ROLE, all_cds, total_ob, framing)

    def decompress(self, archive_bytes: bytes) -> Any:
        total_ob, framing, cds = self._open_archive(self.ROLE, archive_bytes)

        decoded: List[ChunkUnit] = []
        seen: List[ChunkUnit] = []
        phase = 0
        for group, is_tail in self._iter_intervals(cds, self.cfg.train_interval):
            chunk_units = self.backend.decode_interval(self.compressor, group)
            decoded.extend(chunk_units)
            seen.extend(chunk_units)
            if not is_tail:
                train_chunks = chunk_units if self.cfg.train_on_recent_only else seen
                self._train(train_chunks, phase)
                phase += 1

        return self._finalize(self._restore_order(decoded), total_ob, framing)
