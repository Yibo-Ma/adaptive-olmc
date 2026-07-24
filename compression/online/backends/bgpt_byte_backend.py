"""_BGPTByteBackend: shared machinery for bGPT byte-level backends (image/audio).

bGPT is a hierarchical byte model (a patch-level GPT2 + a byte-level GPT2), so it
differs from the LLM family on every axis the ABC abstracts:

* model build + checkpoint load (two GPT2Config sub-models, dropout 0 for
  determinism, seeded init);
* LoRA on GPT2 ``c_attn`` with ``task_type=None`` (the forward signature is
  ``(patches, masks)``, not a standard causal-LM);
* interval encode/decode via the team's ``BGPTCompressor`` (segments of
  ``(bytes, ext)``), not padded token tensors;
* training loss read straight from the model output — ``bGPTLMHeadModel.forward``
  passes ``labels`` internally, so ``out.loss`` is the byte-level CE.

A "chunk" here is one byte segment (an image BMP patch, or an audio WAV chunk);
its bytes are carried as ``ChunkUnit.token_ids`` (byte values 0-255).  Subclasses
set ``self.ext`` ("bmp"/"wav") and provide identity/data plumbing.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from transformers import GPT2Config
from peft import get_peft_model, LoraConfig

from compression.base_compressor import BaseCompressor
from compression.bgpt_compressor import BGPTCompressor
from compression.types import CompressedData
from compression.online.backends.base import ChunkUnit, OnlineBackend
from compression.online.config import OnlineLearningConfig
from utils.determinism import set_seed
from utils.bgpt_codec_utils import (
    bytes_to_padded_tokens, extension_tokens, pad_input_for_bgpt,
)

from bgpt.utils import bGPTLMHeadModel
from bgpt.config import BYTE_NUM_LAYERS, HIDDEN_SIZE, PATCH_NUM_LAYERS, PATCH_SIZE

PATCH_LENGTH = 512        # matches evaluation/eval_bgpt.py
_INIT_SEED = 42           # seed model construction so non-checkpoint params match both ends


class _BGPTByteBackend(OnlineBackend):

    ext: str = "bin"      # set by subclass

    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        super().__init__(device)
        self.checkpoint_path = checkpoint_path
        self.patch_size = PATCH_SIZE

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_backbone(self) -> None:
        set_seed(_INIT_SEED)          # any non-checkpoint param inits identically on both ends
        patch_cfg = GPT2Config(
            num_hidden_layers=PATCH_NUM_LAYERS, max_length=PATCH_LENGTH,
            max_position_embeddings=PATCH_LENGTH, hidden_size=HIDDEN_SIZE,
            n_head=HIDDEN_SIZE // 64, vocab_size=1,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        byte_cfg = GPT2Config(
            num_hidden_layers=BYTE_NUM_LAYERS, max_length=PATCH_SIZE + 1,
            max_position_embeddings=PATCH_SIZE + 1, hidden_size=HIDDEN_SIZE,
            n_head=HIDDEN_SIZE // 64, vocab_size=257,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        model = bGPTLMHeadModel(patch_cfg, byte_cfg)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["model"], strict=False)
        self.model = model.to(self.device).eval()

    def attach_lora(self, cfg: OnlineLearningConfig) -> None:
        set_seed(cfg.base_seed)       # deterministic LoRA init (both ends identical)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            bias="none",
            task_type=None,           # custom forward(patches, masks) → plain PeftModel
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.eval()

    def make_compressor(self) -> BaseCompressor:
        return BGPTCompressor(self.model, patch_size=self.patch_size, device=self.device)

    # ------------------------------------------------------------------
    # Data <-> chunks  (raw = list of byte-blobs; one blob per chunk)
    # ------------------------------------------------------------------
    # The caller pre-splits the medium into bGPT-native segments (32x32 BMP
    # patches for image, fixed-duration WAV chunks for audio) and passes the
    # blob list as ``raw``.  Round-trip is byte-exact at the blob level.

    def to_chunks(self, raw: List[bytes]) -> List[ChunkUnit]:
        return [ChunkUnit(token_ids=list(b)) for b in raw if b]

    def from_chunks(self, chunks: List[ChunkUnit]) -> List[bytes]:
        return [bytes(c.token_ids) for c in chunks]

    def raw_size_bytes(self, raw: List[bytes]) -> int:
        return sum(len(b) for b in raw)

    # ------------------------------------------------------------------
    # Interval encode / decode  (BGPTCompressor speaks (bytes, ext))
    # ------------------------------------------------------------------

    def encode_interval(
        self, compressor: BaseCompressor, chunks: List[ChunkUnit]
    ) -> List[CompressedData]:
        segments = [(bytes(c.token_ids), self.ext) for c in chunks]
        return compressor.compress_batch(segments)

    def decode_interval(
        self, compressor: BaseCompressor, cds: List[CompressedData]
    ) -> List[ChunkUnit]:
        # ext is fixed per modality, so it need not be stored in the archive;
        # restore it on the rebuilt CompressedData before decoding.
        for cd in cds:
            cd.metadata["ext"] = self.ext
        blobs = compressor.decompress_batch(cds, show_progress=True)
        return [ChunkUnit(token_ids=list(b)) for b in blobs]

    # ------------------------------------------------------------------
    # Training  (one bGPT batch per interval; loss from the model output)
    # ------------------------------------------------------------------

    def build_training_windows(
        self, chunks: List[ChunkUnit], cfg: OnlineLearningConfig
    ) -> List[dict]:
        if not chunks:
            return []
        payloads = [
            bytes_to_padded_tokens(bytes(c.token_ids), self.patch_size) for c in chunks
        ]
        ext_ids = [extension_tokens(self.ext, self.patch_size)] * len(chunks)
        batch = pad_input_for_bgpt(payloads, ext_ids, self.device, self.patch_size)
        return [batch]

    def compute_loss(self, model: torch.nn.Module, window: dict) -> torch.Tensor:
        # bGPTLMHeadModel.forward mutates masks in place (masks[:, 0] = 0); clone
        # so the same training window can be reused across epochs.
        out = model(patches=window["patches"], masks=window["masks"].clone())
        return out.loss

    # ------------------------------------------------------------------
    # Identity (modality / fingerprint set by subclass)
    # ------------------------------------------------------------------

    def model_fingerprint(self) -> Dict[str, str]:
        return {"model": self.checkpoint_path, "dtype": "float32",
                "modality": self.modality}
