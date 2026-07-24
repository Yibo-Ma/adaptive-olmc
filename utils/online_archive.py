"""Chunked archive format + settings/environment fingerprint.

Modality- and mode-agnostic container for the online/static compressors.

Layout
------
    MAGIC | version
    varint(len(meta_json)) | meta_json
    varint(len(framing)) | framing
    varint(total_original_bytes) | varint(num_chunks)
    per chunk: varint(original_length) varint(payload_len)
    concatenated chunk payloads

``framing`` is a small, data-dependent descriptor (image sizes, audio clip
lengths, …) the backend uses to rebuild the original medium from decoded chunks;
it is empty for text.  It is kept out of the metadata hash on purpose — it is
data, not a setting.

``meta_json`` carries the *role* (static/online), the *modality*, and compact
hashes of (a) the compression settings and (b) the software+hardware
environment.  Lossless decoding is only guaranteed when both hashes match, so
``validate_meta`` rejects a mismatched archive with a clear error instead of
silently producing garbage.
"""
from __future__ import annotations

import hashlib
import json
from importlib.metadata import version as _pkg_version_lookup
from typing import Dict, List, Tuple

import numpy as np
import torch

ARCHIVE_MAGIC = b"OLMC"      # Online LM Compress
ARCHIVE_VERSION = 2          # v2 adds the framing section after the metadata block


# ---------------------------------------------------------------------------
# varint
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint only supports non-negative integers.")
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        out.append(b | 0x80 if value else b)
        if not value:
            return bytes(out)


def decode_varint(buffer: bytes, offset: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        if offset >= len(buffer):
            raise EOFError("Unexpected EOF while reading varint.")
        b = buffer[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def short_hash(obj) -> str:
    """16-hex-char digest of a JSON-canonicalised object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _pkg_version(name: str) -> str:
    try:
        return _pkg_version_lookup(name)
    except Exception:
        return "?"


def env_fingerprint(model_fp: Dict[str, str], device: torch.device) -> Dict[str, str]:
    """Everything that can perturb bit-exact logits -> must match for lossless decode.

    ``model_fp`` carries the modality-specific bits (model path, dtype, modality);
    this function adds the shared software/hardware stack.
    """
    fp = dict(model_fp)
    fp.update({
        "device": device.type,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": _pkg_version("transformers"),
        "peft": _pkg_version("peft"),
        "constriction": _pkg_version("constriction"),
    })
    if device.type == "cuda":
        try:
            fp["gpu"] = torch.cuda.get_device_name(device)
            fp["cuda"] = str(torch.version.cuda)
        except Exception:
            pass
    return fp


def build_meta(
    role: str, modality: str, settings: Dict,
    model_fp: Dict[str, str], device: torch.device,
) -> Dict[str, str]:
    """Compact metadata: role + modality + hashes of settings and environment."""
    return {
        "role": role,
        "modality": modality,
        "s": short_hash(settings),
        "e": short_hash(env_fingerprint(model_fp, device)),
    }


def validate_meta(
    meta: Dict, role: str, modality: str, settings: Dict,
    model_fp: Dict[str, str], device: torch.device,
) -> None:
    """Raise with a clear message if the archive cannot be safely decoded here."""
    if not meta or "role" not in meta:
        raise ValueError(
            "Archive has no metadata block; cannot verify it is safe to "
            "decompress with these settings.")
    if meta.get("role") != role:
        raise ValueError(
            f"Archive role mismatch: written as '{meta.get('role')}' but being "
            f"decompressed as '{role}'.")
    if meta.get("modality") != modality:
        raise ValueError(
            f"Archive modality mismatch: written as '{meta.get('modality')}' but "
            f"being decompressed as '{modality}'.")
    if meta.get("s") != short_hash(settings):
        raise ValueError(
            "Compression settings mismatch: the config passed to decompression "
            "differs from the one used at compression. Lossless decoding requires "
            "identical settings.")
    if meta.get("e") != short_hash(env_fingerprint(model_fp, device)):
        raise ValueError(
            "Environment fingerprint mismatch: model, dtype, device, GPU, or a "
            "library version (torch/transformers/peft/constriction/numpy) differs "
            "from compression time. Bit-exact lossless decoding is not guaranteed; "
            "decompress on the same setup that produced the archive.")


# ---------------------------------------------------------------------------
# Archive (de)serialisation
# ---------------------------------------------------------------------------

def build_archive(
    original_lengths: List[int],
    payloads: List[bytes],
    total_original_bytes: int,
    meta: Dict,
    framing: bytes = b"",
) -> bytes:
    if len(original_lengths) != len(payloads):
        raise ValueError("original_lengths and payloads must have equal length.")

    header = bytearray()
    header.extend(ARCHIVE_MAGIC)
    header.append(ARCHIVE_VERSION)

    meta_bytes = json.dumps(meta or {}, sort_keys=True).encode("utf-8")
    header.extend(encode_varint(len(meta_bytes)))
    header.extend(meta_bytes)

    header.extend(encode_varint(len(framing)))
    header.extend(framing)

    header.extend(encode_varint(total_original_bytes))
    header.extend(encode_varint(len(original_lengths)))

    body = bytearray()
    for olen, payload in zip(original_lengths, payloads):
        header.extend(encode_varint(olen))
        header.extend(encode_varint(len(payload)))
        body.extend(payload)

    return bytes(header) + bytes(body)


def parse_archive(
    archive_bytes: bytes,
) -> Tuple[Dict, int, bytes, List[int], List[bytes]]:
    if not archive_bytes.startswith(ARCHIVE_MAGIC):
        raise ValueError("Invalid archive magic.")

    offset = len(ARCHIVE_MAGIC)
    version = archive_bytes[offset]
    offset += 1
    if version != ARCHIVE_VERSION:
        raise ValueError(f"Unsupported archive version: {version}")

    meta_len, offset = decode_varint(archive_bytes, offset)
    meta = (json.loads(archive_bytes[offset:offset + meta_len].decode("utf-8"))
            if meta_len else {})
    offset += meta_len

    framing_len, offset = decode_varint(archive_bytes, offset)
    framing = bytes(archive_bytes[offset:offset + framing_len])
    offset += framing_len

    total_original_bytes, offset = decode_varint(archive_bytes, offset)
    num_chunks, offset = decode_varint(archive_bytes, offset)

    original_lengths: List[int] = []
    payload_lens: List[int] = []
    for _ in range(num_chunks):
        olen, offset = decode_varint(archive_bytes, offset)
        plen, offset = decode_varint(archive_bytes, offset)
        original_lengths.append(olen)
        payload_lens.append(plen)

    payloads: List[bytes] = []
    for plen in payload_lens:
        payloads.append(archive_bytes[offset:offset + plen])
        offset += plen
    if offset != len(archive_bytes):
        raise ValueError("Archive payload length / metadata mismatch.")

    return meta, total_original_bytes, framing, original_lengths, payloads


def write_archive(filename: str, archive_bytes: bytes) -> None:
    with open(filename, "wb") as f:
        f.write(archive_bytes)


def read_archive(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()
