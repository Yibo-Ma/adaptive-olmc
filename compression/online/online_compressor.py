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
        shuffle_seed: Optional[int] = None, measure_gk: bool = False,
    ) -> None:
        super().__init__(backend, device, shuffle_seed=shuffle_seed)
        self.cfg = cfg
        self.optimizer = None
        self.trainer = None
        # g_k instrument (see _record_gk).  None disables it and keeps the coding
        # path byte-identical; a list collects one prequential record per training
        # boundary.  Encoder-side only — decoding never needs it.
        self.gk_records: Optional[List[Dict]] = [] if measure_gk else None

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
        # Materialised so a training boundary can look ahead to the next interval
        # (the g_k probe).  Same grouping the decoder replays.
        intervals = list(self._iter_intervals(chunks, self.cfg.train_interval))

        all_cds = []
        seen: List[ChunkUnit] = []
        phase = 0
        for k, (group, is_tail) in enumerate(intervals):
            all_cds.extend(self.backend.encode_interval(self.compressor, group))
            seen.extend(group)
            if is_tail:
                continue
            train_chunks = group if self.cfg.train_on_recent_only else seen
            nxt = intervals[k + 1][0] if k + 1 < len(intervals) else None
            if self.gk_records is None or nxt is None:
                self._train(train_chunks, phase)
            else:
                self._record_gk(phase, group, nxt, train_chunks)
            phase += 1

        return self._assemble_archive(self.ROLE, all_cds, total_ob, framing)

    # ------------------------------------------------------------------
    # g_k instrument (encoder-side measurement; off unless measure_gk=True)
    # ------------------------------------------------------------------

    def _measure_interval(self, chunks: List[ChunkUnit]):
        """(bits, tokens, orig_bytes) for one interval at the current model state."""
        bits = sum(self.backend.measure_interval_bits(self.compressor, chunks))
        tokens = sum(len(c.token_ids) for c in chunks)
        orig_bytes = self.backend.raw_size_bytes(self.backend.from_chunks(chunks))
        return bits, tokens, orig_bytes

    def _record_gk(
        self, phase: int, curr: List[ChunkUnit], nxt: List[ChunkUnit],
        train_chunks: List[ChunkUnit],
    ) -> None:
        """Measure the prequential g_k and Δ_in around one training step.

        g_k = L(next | S_k) − L(next | S_{k+1}) isolates the marginal effect of
        *this* update on the *next* interval; Δ_in = L(curr | S_k) − L(curr | S_{k+1})
        its effect on the interval it trained on.  We store only raw code lengths
        (bits) plus each interval's token and original-byte counts, so the consumer
        (evaluation/gk_curve.py) can form the differences and normalise them any way
        it likes (Δbpb, relative %, bits/token) — the model run stays the sole
        producer of ground-truth numbers.

        The two pre-update reads see state S_k; ``_train`` advances it to S_{k+1}
        and reseeds the global RNG at entry (utils.determinism.set_seed), so these
        inference-only forwards cannot perturb the training trajectory — losslessness
        is preserved and the archive is byte-identical to an un-instrumented run.
        """
        curr_pre, curr_tok, curr_by = self._measure_interval(curr)
        next_pre, next_tok, next_by = self._measure_interval(nxt)
        self._train(train_chunks, phase)
        curr_post, _, _ = self._measure_interval(curr)
        next_post, _, _ = self._measure_interval(nxt)
        self.gk_records.append({
            "phase": phase,
            "curr": {"tokens": curr_tok, "bytes": curr_by,
                     "bits_pre": curr_pre, "bits_post": curr_post},
            "next": {"tokens": next_tok, "bytes": next_by,
                     "bits_pre": next_pre, "bits_post": next_post},
        })

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
