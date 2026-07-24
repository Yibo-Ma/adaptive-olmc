"""OnlineBackend: the single modality-specific seam.

Everything that varies between text / audio / image lives behind this ABC:
how raw data maps to/from token chunks, which backbone + compressor to build,
how a LoRA adapter is attached, and how a training loss is computed.  The
scheduler (StaticCompressor / OnlineCompressor) and the trainer are written
once against this interface and never branch on modality.

A backend owns the model after ``load_backbone`` / ``attach_lora`` are called;
the compressor returned by ``make_compressor`` wraps that *same* model object,
so in-place LoRA updates between intervals are seen by subsequent forward passes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from compression.base_compressor import BaseCompressor
from compression.types import CompressedData
from compression.online.config import OnlineLearningConfig


@dataclass
class ChunkUnit:
    """One chunk's canonical payload: the token-id sequence to be compressed.

    ``meta`` carries any modality-specific bookkeeping needed to reassemble the
    original data (unused for plain text; e.g. image/audio framing later).
    """
    token_ids: List[int]
    meta: Dict[str, Any] = field(default_factory=dict)


class OnlineBackend(ABC):
    """Modality adapter.  Holds the model; bridges to a BaseCompressor."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load_backbone(self) -> None:
        """Load the (fp32) backbone into ``self.model`` and any aux state."""

    @abstractmethod
    def attach_lora(self, cfg: OnlineLearningConfig) -> None:
        """Wrap ``self.model`` with a deterministically-initialised LoRA adapter."""

    @abstractmethod
    def make_compressor(self) -> BaseCompressor:
        """Return a BaseCompressor wrapping the current ``self.model``."""

    # ------------------------------------------------------------------
    # Data <-> token chunks  (pure bijection, no compression logic)
    # ------------------------------------------------------------------

    @abstractmethod
    def to_chunks(self, raw: Any) -> List[ChunkUnit]:
        ...

    @abstractmethod
    def from_chunks(self, chunks: List[ChunkUnit]) -> Any:
        """Canonical decoded payload (string for text, blob list for byte media).

        This is the object the integrity check sizes; ``reconstruct`` turns it
        into the presentation medium.
        """

    @abstractmethod
    def raw_size_bytes(self, raw: Any) -> int:
        """Size of the original data in bytes (for ratio + integrity check)."""

    def reconstruct(self, canonical: Any, framing: bytes) -> Any:
        """Rebuild the presentation medium from the canonical payload + framing.

        Default: identity — text needs no framing, the string *is* the medium.
        Byte media (image/audio) override this to retile / regroup decoded blobs
        into full images / per-clip audio using the compact ``framing`` blob.
        """
        return canonical

    # ------------------------------------------------------------------
    # Interval encode / decode  (bridges to the compressor's signature)
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_interval(
        self, compressor: BaseCompressor, chunks: List[ChunkUnit]
    ) -> List[CompressedData]:
        ...

    @abstractmethod
    def decode_interval(
        self, compressor: BaseCompressor, cds: List[CompressedData]
    ) -> List[ChunkUnit]:
        ...

    # ------------------------------------------------------------------
    # Training  (default = next-token CE over concatenated chunk windows)
    # ------------------------------------------------------------------

    def build_training_windows(
        self, chunks: List[ChunkUnit], cfg: OnlineLearningConfig
    ) -> List[torch.Tensor]:
        """Concatenate chunk tokens and slice into <= max_seq_len windows.

        Default works for any autoregressive token model (text/audio/image-GPT).
        """
        if not chunks:
            return []
        flat: List[int] = []
        for c in chunks:
            flat.extend(c.token_ids)
        all_tokens = torch.tensor([flat], dtype=torch.long, device=self.device)
        windows: List[torch.Tensor] = []
        for start in range(0, all_tokens.shape[1], cfg.max_seq_len):
            w = all_tokens[:, start:start + cfg.max_seq_len]
            if w.shape[1] >= 2:
                windows.append(w)
        return windows

    def compute_loss(self, model: torch.nn.Module, window: torch.Tensor) -> torch.Tensor:
        """Next-token cross-entropy for a [1, L] token window."""
        logits = model(window[:, :-1]).logits
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), window[:, 1:].reshape(-1)
        )

    # ------------------------------------------------------------------
    # Identity / fingerprint
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def modality(self) -> str:
        ...

    @abstractmethod
    def model_fingerprint(self) -> Dict[str, str]:
        """Modality-specific bits folded into the archive env hash."""

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]
