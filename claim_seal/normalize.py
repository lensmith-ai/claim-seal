"""norm/v1: the transformation that turns captured source bytes into the text a span addresses.

Why this is its own file: a hash reproduces BYTES, it does not reproduce a TRANSFORMATION. A
later verifier that holds the raw digest and the normalized digest still cannot check a span
unless it can re-run the exact transformation between them. So the normalizer's own source is
part of the evidence and gets its own sha256 (normalizer_sha256), recorded beside the two
artifact digests. Keeping it in one dependency-free file is what makes that digest meaningful:
a digest over a module that imports half a tree proves nothing.

Do not edit a published normalizer in place. Its digest is cited by sealed records. Add a
`norm/v2` beside `norm/v1` and leave v1 alone.
"""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

# Offsets are half-open [start, end) over the CODE POINTS of the normalized text, never over raw
# bytes (which move under any charset change) and never over rendered text (not reproducible).
COORDINATE_SYSTEM = "normalized-codepoint"
NORMALIZER_ID = "norm/v1"


def norm_v1(raw: bytes, config: dict | None = None) -> str:
    """Decode UTF-8 strictly, normalize line endings, apply Unicode NFC.

    Deliberately boring and total: it must produce the same output on any machine, and it must
    not drop content (no whitespace collapsing, no tag stripping), because a span's offsets
    address THIS string and a lossy step would make a quoted passage unlocatable.
    """
    encoding = (config or {}).get("encoding", "utf-8")
    text = raw.decode(encoding, errors="strict")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


NORMALIZERS = {NORMALIZER_ID: norm_v1}


def normalizer_sha256() -> str:
    """sha256 of THIS FILE: the transformation's identity, recorded wherever it is used.

    It hashes this library's own normalizer source, so it differs from the digest the original
    composer records for norm/v1 (a different file), although the transformation is the same.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def normalize(normalizer_id: str, raw: bytes, config: dict | None = None) -> str:
    if normalizer_id not in NORMALIZERS:
        raise KeyError(f"unknown normalizer {normalizer_id!r}; known: {sorted(NORMALIZERS)}")
    return NORMALIZERS[normalizer_id](raw, config)
