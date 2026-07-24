"""TextBackend: text compression on a causal LLM (paper §3.4).

Only the text<->token mapping is defined here; all model/coding/training
machinery is inherited from _LLMTokenBackend.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from compression.online.backends.base import ChunkUnit
from compression.online.backends.llm_token_backend import _LLMTokenBackend


class TextBackend(_LLMTokenBackend):

    def __init__(
        self, model_path: str, device: torch.device, chunk_size_tokens: int = 512,
    ) -> None:
        super().__init__(model_path, device)
        self.chunk_size_tokens = chunk_size_tokens

    def load_backbone(self) -> None:
        super().load_backbone()
        self.pad_id = self.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Data <-> token chunks
    # ------------------------------------------------------------------

    def to_chunks(self, raw: str) -> List[ChunkUnit]:
        ids = self.tokenizer(raw, add_special_tokens=False)["input_ids"]
        n = self.chunk_size_tokens
        return [ChunkUnit(token_ids=ids[s:s + n]) for s in range(0, len(ids), n)]

    def from_chunks(self, chunks: List[ChunkUnit]) -> str:
        all_ids: List[int] = []
        for c in chunks:
            all_ids.extend(c.token_ids)
        return self.tokenizer.decode(
            all_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def raw_size_bytes(self, raw: str) -> int:
        return len(raw.encode("utf-8"))

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def modality(self) -> str:
        return "text"

    def model_fingerprint(self) -> Dict[str, str]:
        return {"model": self.model_path, "dtype": "float32", "modality": "text"}
