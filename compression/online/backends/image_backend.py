"""ImageBackend: lossless bGPT image compression (aligned with the team).

The medium is pre-split into 32x32 BMP patches (see utils.img_utils.patchify_image);
each patch is one byte-blob chunk.  All machinery is inherited from
_BGPTByteBackend — only the extension and identity differ.
"""
from __future__ import annotations

from typing import List

from compression.online.backends.bgpt_byte_backend import _BGPTByteBackend


class ImageBackend(_BGPTByteBackend):

    ext = "bmp"

    @property
    def modality(self) -> str:
        return "image"

    def reconstruct(self, canonical: List[bytes], framing: bytes):
        """Retile the decoded BMP patch-blobs back into whole PIL images.

        With no framing (older archives / patch-level use) the canonical blob
        list is returned unchanged.
        """
        if not framing:
            return canonical
        from utils.img_utils import reassemble_images_from_blobs
        return reassemble_images_from_blobs(canonical, framing)
