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

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from compression.base_compressor import BaseCompressor
from compression.llm_compressor import LLMCompressor
from compression.types import CompressedData, PromptContext
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

    def _prompt_ctx(self, ctx_ids: Optional[List[int]]) -> Optional[PromptContext]:
        """Cross-chunk context as a PromptContext, or None for the BOS-prefixed default.

        One ``[1, ctx_len]`` row is broadcast across the interval, so every chunk in
        the batch is conditioned on exactly the same already-coded tail.  The decoder
        rebuilds the identical tail from what it has decoded, so nothing is
        transmitted; ``prefix_length`` makes the coder skip the context positions and
        bill only the chunk's own tokens (right-padding stays safe — each sequence is
        sliced to ``prefix_length + its own length`` before coding).
        """
        if not ctx_ids:
            return None
        ids = torch.tensor(ctx_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return PromptContext(mode="tokens", token_ids=ids)

    def encode_interval(
        self, compressor: BaseCompressor, chunks: List[ChunkUnit],
        ctx_ids: Optional[List[int]] = None,
    ) -> List[CompressedData]:
        input_ids, attn = pad_token_ids(
            [c.token_ids for c in chunks], self.pad_id, device=self.device
        )
        return compressor.compress_batch(
            input_ids, attn, prompt_ctx=self._prompt_ctx(ctx_ids))

    def decode_interval(
        self, compressor: BaseCompressor, cds: List[CompressedData],
        ctx_ids: Optional[List[int]] = None,
    ) -> List[ChunkUnit]:
        decoded = compressor.decompress_batch(
            cds, prompt_ctx=self._prompt_ctx(ctx_ids), show_progress=True)
        return [ChunkUnit(token_ids=t[0].cpu().tolist()) for t in decoded]

    def measure_interval_bits(
        self, compressor: BaseCompressor, chunks: List[ChunkUnit],
        ctx_ids: Optional[List[int]] = None,
    ) -> List[float]:
        # measure_batch_bits mirrors the BOS-prefixed coding path only; billing a
        # context-conditioned archive while measuring BOS-conditioned bits would make
        # every probe inconsistent with the coder, so refuse instead of misreporting.
        if ctx_ids:
            raise NotImplementedError(
                "measure_interval_bits does not mirror cross-chunk context yet; "
                "run the g_k / branch / adjacency probes without --ctx-tokens.")
        # Mirror encode_interval exactly (same BOS-prefixed padding), so the bits
        # reported are the ones the coder would bill under the current state.
        input_ids, attn = pad_token_ids(
            [c.token_ids for c in chunks], self.pad_id, device=self.device
        )
        return compressor.measure_batch_bits(input_ids, attn)
