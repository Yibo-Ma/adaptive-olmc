"""_ChunkedCompressor: pipeline shared by StaticCompressor and OnlineCompressor.

Owns the modality-agnostic plumbing (interval iteration, archive assembly /
parsing, fingerprint wiring) so the two concrete compressors only express
*what happens per interval*.  This is deliberately a thin base, not a model
base class — the model lives in the backend.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from compression.types import CompressedData
from compression.online.backends.base import ChunkUnit, OnlineBackend
from utils import online_archive as ar


class _ChunkedCompressor(ABC):

    def __init__(self, backend: OnlineBackend, device: torch.device,
                 shuffle_seed: Optional[int] = None) -> None:
        self.backend = backend
        self.device = device
        self.shuffle_seed = shuffle_seed   # None = natural stream order
        self.compressor = None        # set in setup()

    # ------------------------------------------------------------------
    # Interval iteration
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_intervals(
        items: List, size: int
    ) -> Iterator[Tuple[List, bool]]:
        """Yield (group, is_partial_tail).

        A group shorter than ``size`` is the final partial leftover; the online
        scheduler trains after every *full* group (including a final full one)
        but never after the partial tail, matching the reference schedule.
        """
        for start in range(0, len(items), size):
            group = items[start:start + size]
            yield group, (len(group) < size)

    # ------------------------------------------------------------------
    # Coding order (shuffle control)
    # ------------------------------------------------------------------
    # The coding order is a pure function of (shuffle_seed, n_chunks): both
    # endpoints derive the identical permutation from the archive's chunk
    # count, so zero side information is transmitted.  ``None`` keeps the
    # natural stream order.  A dedicated random.Random instance is used
    # because the *global* RNGs are reserved for the deterministic LoRA
    # init/training replay (utils.determinism.set_seed) — drawing from them
    # here would perturb that replay and break losslessness.

    def _chunk_order(self, n: int) -> Optional[List[int]]:
        """order[i] = stream index of the chunk coded at position i."""
        if self.shuffle_seed is None:
            return None
        order = list(range(n))
        random.Random(self.shuffle_seed).shuffle(order)
        return order

    def _prepare_chunks(self, raw: Any) -> List[ChunkUnit]:
        """Chunk ``raw`` and arrange the chunks in coding order."""
        chunks = [c for c in self.backend.to_chunks(raw) if c.token_ids]
        order = self._chunk_order(len(chunks))
        return chunks if order is None else [chunks[j] for j in order]

    def _restore_order(self, decoded: List[ChunkUnit]) -> List[ChunkUnit]:
        """Invert the coding order so reassembly (framing) sees stream order."""
        order = self._chunk_order(len(decoded))
        if order is None:
            return decoded
        restored: List[Optional[ChunkUnit]] = [None] * len(decoded)
        for pos, j in enumerate(order):
            restored[j] = decoded[pos]
        return restored          # type: ignore[return-value]

    def _coding_settings(self) -> Dict:
        """Settings folded into the archive hash: subclass knobs + coding order.

        ``shuffle_seed`` is added only when set, so natural-order archives keep
        their existing hash (backward compatible), while a shuffled archive
        refuses to decode without the exact same seed.
        """
        settings = dict(self._settings())
        if self.shuffle_seed is not None:
            settings["shuffle_seed"] = self.shuffle_seed
        return settings

    # ------------------------------------------------------------------
    # Archive helpers
    # ------------------------------------------------------------------

    def _assemble_archive(
        self, role: str,
        cds: List[CompressedData], total_original_bytes: int,
        framing: bytes = b"",
    ) -> bytes:
        meta = ar.build_meta(
            role, self.backend.modality, self._coding_settings(),
            self.backend.model_fingerprint(), self.device,
        )
        return ar.build_archive(
            [cd.original_length for cd in cds],
            [cd.compressed_bytes for cd in cds],
            total_original_bytes, meta, framing,
        )

    def _open_archive(
        self, role: str, archive_bytes: bytes,
    ) -> Tuple[int, bytes, List[CompressedData]]:
        meta, total_ob, framing, original_lengths, payloads = ar.parse_archive(archive_bytes)
        ar.validate_meta(
            meta, role, self.backend.modality, self._coding_settings(),
            self.backend.model_fingerprint(), self.device,
        )
        cds = [
            CompressedData(compressed_bytes=p, original_length=ol)
            for p, ol in zip(payloads, original_lengths)
        ]
        return total_ob, framing, cds

    def _finalize(
        self, decoded_chunks: List[ChunkUnit], total_ob: int, framing: bytes,
    ) -> Any:
        # Integrity check on the canonical payload (blobs / string) — cheap and
        # unchanged; then rebuild the presentation medium via the backend, which
        # consumes ``framing`` (identity for text, retile/regroup for image/audio).
        canonical = self.backend.from_chunks(decoded_chunks)
        if self.backend.raw_size_bytes(canonical) != total_ob:
            raise ValueError(
                "Decoded data size does not match archive metadata "
                f"({self.backend.raw_size_bytes(canonical)} != {total_ob}).")
        return self.backend.reconstruct(canonical, framing)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def compress(self, raw: Any, framing: bytes = b"") -> bytes:
        ...

    @abstractmethod
    def decompress(self, archive_bytes: bytes) -> Any:
        ...
