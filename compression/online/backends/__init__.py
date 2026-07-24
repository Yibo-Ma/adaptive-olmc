"""Modality backends for the online compression layer.

  TextBackend   — causal LLM (paper §3.4)
  ImageBackend  — bGPT byte-level, 32x32 BMP patches (team-aligned)
  AudioBackend  — bGPT byte-level, WAV chunks (team-aligned)
"""
from compression.online.backends.base import ChunkUnit, OnlineBackend
from compression.online.backends.text_backend import TextBackend
from compression.online.backends.image_backend import ImageBackend
from compression.online.backends.audio_backend import AudioBackend

__all__ = [
    "ChunkUnit", "OnlineBackend", "TextBackend", "ImageBackend", "AudioBackend",
]
