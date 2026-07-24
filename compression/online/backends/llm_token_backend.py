"""_LLMTokenBackend: shared machinery for any autoregressive causal-LM backend.

Currently only TextBackend builds on this (paper §3.4); image and audio use the
byte-level bGPT backend instead (see _BGPTByteBackend).  A causal-LM subclass
differs only in how raw data maps to/from token ids — everything else (fp32 load,
deterministic LoRA, the LLMCompressor wrapper, interval encode/decode, and the
next-token-CE training defaults) lives here.

Subclasses must implement: to_chunks / from_chunks / raw_size_bytes / modality /
model_fingerprint, and set ``self.pad_id`` (a valid token id used only for
right-padding future positions, which the causal mask ignores).
"""
from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from compression.base_compressor import BaseCompressor
from compression.llm_compressor import LLMCompressor
from compression.types import CompressedData
from compression.online.backends.base import ChunkUnit, OnlineBackend
from compression.online.config import OnlineLearningConfig
from utils.determinism import set_seed
from utils.text_utils import load_lm_tokenizer, pad_token_ids


class _LLMTokenBackend(OnlineBackend):

    def __init__(self, model_path: str, device: torch.device) -> None:
        super().__init__(device)
        self.model_path = model_path
        self.tokenizer = None
        self.pad_id = 0          # subclasses may refine after load_backbone()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_backbone(self) -> None:
        self.tokenizer = load_lm_tokenizer(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM
            .from_pretrained(self.model_path, torch_dtype=torch.float32)
            .to(self.device)
            .eval()
        )

    def attach_lora(self, cfg: OnlineLearningConfig) -> None:
        set_seed(cfg.base_seed)          # deterministic LoRA init (both ends identical)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.eval()

    def make_compressor(self) -> BaseCompressor:
        return LLMCompressor(self.model, self.tokenizer, device=self.device)

    # ------------------------------------------------------------------
    # Interval encode / decode (shared; pad_id differs per subclass)
    # ------------------------------------------------------------------

    def encode_interval(
        self, compressor: BaseCompressor, chunks: List[ChunkUnit]
    ) -> List[CompressedData]:
        input_ids, attn = pad_token_ids(
            [c.token_ids for c in chunks], self.pad_id, device=self.device
        )
        return compressor.compress_batch(input_ids, attn)

    def decode_interval(
        self, compressor: BaseCompressor, cds: List[CompressedData]
    ) -> List[ChunkUnit]:
        decoded = compressor.decompress_batch(cds, show_progress=True)
        return [ChunkUnit(token_ids=t[0].cpu().tolist()) for t in decoded]
