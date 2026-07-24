"""StaticCompressor: fixed-model chunked compression (no online learning).

This is the team's batched LLMCompressor path expressed through the modality
backend, so the same class serves text / audio / image.  Chunks are grouped
into fixed-size batches; encode and decode use identical grouping so the
(B, L_max) layout matches and logits are bit-exact.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from compression.online.backends.base import ChunkUnit, OnlineBackend
from compression.online.base import _ChunkedCompressor


class StaticCompressor(_ChunkedCompressor):

    ROLE = "static"

    def __init__(
        self, backend: OnlineBackend, device: torch.device, batch_chunks: int = 4,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        super().__init__(backend, device, shuffle_seed=shuffle_seed)
        self.batch_chunks = batch_chunks

    def _settings(self) -> Dict:
        return {"batch_chunks": self.batch_chunks}

    def setup(self) -> None:
        self.backend.load_backbone()
        self.compressor = self.backend.make_compressor()

    # ------------------------------------------------------------------

    def compress(self, raw: Any, framing: bytes = b"") -> bytes:
        chunks = self._prepare_chunks(raw)
        total_ob = self.backend.raw_size_bytes(raw)

        all_cds = []
        for group, _ in self._iter_intervals(chunks, self.batch_chunks):
            all_cds.extend(self.backend.encode_interval(self.compressor, group))

        return self._assemble_archive(self.ROLE, all_cds, total_ob, framing)

    def decompress(self, archive_bytes: bytes) -> Any:
        total_ob, framing, cds = self._open_archive(self.ROLE, archive_bytes)

        decoded: List[ChunkUnit] = []
        for group, _ in self._iter_intervals(cds, self.batch_chunks):
            decoded.extend(self.backend.decode_interval(self.compressor, group))

        return self._finalize(self._restore_order(decoded), total_ob, framing)
