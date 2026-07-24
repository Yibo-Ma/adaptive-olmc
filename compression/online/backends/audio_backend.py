"""AudioBackend: lossless bGPT audio compression (aligned with the team).

The medium is pre-split into fixed-duration 8 kHz / mono / 8-bit WAV chunks
(see utils.audio_utils.chunk_pydub_audio); each chunk is one byte-blob.  All
machinery is inherited from _BGPTByteBackend — only the extension and identity
differ.
"""
from __future__ import annotations

from typing import List

from compression.online.backends.bgpt_byte_backend import _BGPTByteBackend


class AudioBackend(_BGPTByteBackend):

    ext = "wav"

    @property
    def modality(self) -> str:
        return "audio"

    def reconstruct(self, canonical: List[bytes], framing: bytes):
        """Regroup the decoded WAV chunk-blobs into one sample-exact PCM byte
        string per clip.  With no framing the canonical blob list is returned
        unchanged.
        """
        if not framing:
            return canonical
        from utils.audio_utils import reassemble_pcm_from_blobs
        return reassemble_pcm_from_blobs(canonical, framing)
