"""
text.py — MiniLM caption + transcript encoder
=============================================

Step 3.7 of the Content Analyzer build.

Encodes the video's caption (from the upload payload) and the
transcript text produced by `services/audio.py` into a single
384-dimensional text vector via
`sentence-transformers/all-MiniLM-L6-v2`.

Pipeline per video
------------------
    caption (str), transcript (str)
        | concatenate with separator (caption first)
        v
    combined text
        | tokenizer truncation to model max_seq_length
        v
    MiniLM forward pass
        v
    text_embedding (384,)

Empty input
-----------
A video can legitimately have neither caption nor transcript (no
caption supplied + silent video that step 3.6 short-circuited). The
model would happily encode an empty string into a noisy, non-
informative vector; instead we return a zero vector. Step 3.8 (modal
fusion) detects the zero vector and downweights the text modality
accordingly. This mirrors the "silent audio = empty transcript, not
an error" convention from `services/audio.py`.

Truncation
----------
MiniLM's max_seq_length is 256 tokens (~1000 chars of typical English).
We rely on the tokenizer's built-in truncation, so long transcripts are
silently truncated at the end. Caption comes first in the concatenation
so it survives even when the transcript is huge.

Normalization
-------------
We do NOT L2-normalize the returned vector here. Step 3.8 (modal
fusion) owns the final per-modality normalization so the relative
scale of visual vs text is decided in one place — same convention as
`services/visual.py`.

Lifecycle
---------
The model file is ~90 MB and the first load takes a couple of seconds.
We pay that cost once at worker startup via an explicit `load()`, then
call `encode()` per video.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env


log = logging.getLogger(__name__)


# Documented output dimensionality for all-MiniLM-L6-v2. Checked at
# load time in case a different model is configured via TEXT_MODEL.
MINILM_L6_DIM = 384


# --- Errors ------------------------------------------------------------------

class TextEncodeError(RuntimeError):
    """Raised when the MiniLM backend fails on an otherwise-valid string."""


# --- Tunables (env-backed) ---------------------------------------------------

def _text_model_name() -> str:
    return os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# --- Text composition --------------------------------------------------------

def _combine(caption: str | None, transcript: str | None) -> str:
    """Concatenate caption + transcript with a single separator.

    Caption goes first so it survives tokenizer truncation when the
    transcript is very long. Empty / whitespace-only segments are
    dropped so we never produce stray separators like ". hello".
    """
    parts: list[str] = []
    if caption and caption.strip():
        parts.append(caption.strip())
    if transcript and transcript.strip():
        parts.append(transcript.strip())
    return ". ".join(parts)


# --- Encoder -----------------------------------------------------------------

class TextEncoder:
    """Thin wrapper around `sentence-transformers` for caption+transcript.

    Lazy: `sentence_transformers` / `torch` are imported inside `load()`
    so other modules in this package can be unit-tested without paying
    the import cost (or even having the packages installed for non-ML
    paths).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ):
        self.model_name = model_name or _text_model_name()
        self.device = device
        self._model = None
        self._dim: Optional[int] = None

    # --- Lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Load MiniLM weights. Idempotent."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        log.info("[INFO] loading text encoder %s on %s", self.model_name, self.device)
        model = SentenceTransformer(self.model_name, device=self.device)

        self._model = model
        # `get_embedding_dimension` is the new name (sentence-transformers >= 3.x);
        # fall back to the legacy `get_sentence_embedding_dimension` for older releases.
        dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        self._dim = int(dim_fn())

        if self._dim != MINILM_L6_DIM:
            log.warning(
                "[WARN] text encoder dim=%d (expected %d for MiniLM-L6-v2). "
                "Downstream fusion (step 3.8) must accommodate the new dim.",
                self._dim, MINILM_L6_DIM,
            )
        log.info("[OK] text encoder loaded (dim=%d)", self._dim)

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("encoder not loaded; call .load() first")
        return self._dim

    # --- Encoding ------------------------------------------------------------

    def encode(self, caption: str | None, transcript: str | None) -> np.ndarray:
        """Encode caption + transcript into a single text vector.

        Returns a float32 numpy array of shape (embedding_dim,). When
        both inputs are empty/whitespace, returns the zero vector —
        the documented "no text signal" sentinel that fusion (step 3.8)
        detects. Never raises on empty input.
        """
        if self._model is None:
            self.load()

        combined = _combine(caption, transcript)
        if not combined:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            vec = self._model.encode(
                combined,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as e:
            raise TextEncodeError(f"sentence-transformers encode failed: {e}") from e

        return np.asarray(vec, dtype=np.float32)


# --- Module-level singleton --------------------------------------------------

_encoder: Optional[TextEncoder] = None


def get_encoder() -> TextEncoder:
    """Return a process-wide, lazily-loaded encoder.

    Production startup should instead instantiate the encoder explicitly
    in `main.py` (step 3.10) and inject it into the consumer handler,
    so load-time errors fire at startup, not on the first event delivery.
    """
    global _encoder
    if _encoder is None:
        enc = TextEncoder()
        enc.load()
        _encoder = enc
    return _encoder
