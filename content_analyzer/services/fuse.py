"""
fuse.py — modal fusion into the final content_embedding
=======================================================

Step 3.8 of the Content Analyzer build.

Combines the 512-d visual vector from `services/visual.py` and the
384-d text vector from `services/text.py` into a single 384-d
`content_embedding` that matches the Atlas Vector Search index defined
in `database/scripts/vector_index_def.json`
(`numDimensions: 384, similarity: cosine`).

Strategy
--------
The index is 384-d cosine, so concatenation (512 + 384 = 896) is not
viable. With no training data available yet, learning a projection
would be premature. We use a deterministic, untrained projection:

    1. Project visual 512 -> 384 with a fixed-seed Gaussian random
       matrix. Random Gaussian projections approximately preserve
       L2 distances (Johnson-Lindenstrauss); for an untrained
       baseline this is the principled cheap choice.
    2. L2-normalize each modality independently so neither dominates
       the sum by raw magnitude.
    3. Weighted sum: alpha * visual_unit + (1 - alpha) * text_unit.
       Default alpha=0.6 — a mild visual lean because CLIP frame
       embeddings are more robust than Whisper-tiny transcripts on
       short / noisy / non-English clips.
    4. L2-normalize the result. The Atlas index uses cosine
       similarity, which is magnitude-invariant, but downstream code
       (Recall, Ranking) often dot-products the vector directly and
       expects unit norm.

Determinism
-----------
The projection matrix is built once at first use from a fixed seed
(`_PROJECTION_SEED`). Changing that seed invalidates every embedding
ever written — treat it as part of the analyzer version key. If you
need to rotate the projection, bump `ANALYZER_VERSION` so existing
docs get re-embedded on next delivery.

Empty modalities
----------------
The text encoder (step 3.7) returns a zero vector when there is no
caption and no transcript. That is handled naturally: a zero text
vector L2-normalizes to itself (zero), so the weighted sum reduces to
the projected visual contribution, and the final L2-normalize fixes
the magnitude. No special-case branch needed.

If both modalities are zero (theoretically possible if the visual
encoder ever returns zero), we return the zero vector rather than
NaN — the caller decides whether to skip the write or surface a
failure.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env


log = logging.getLogger(__name__)


# Dimensions are documented constants, not configuration. The Atlas index
# def and the encoder outputs are the source of truth — these match them.
VISUAL_DIM = 512   # CLIP ViT-B/32 projection_dim (services/visual.py)
TEXT_DIM = 384     # MiniLM-L6-v2 dim (services/text.py)
INDEX_DIM = 384    # database/scripts/vector_index_def.json

# Fixed seed for the random projection. Bumping this invalidates every
# embedding already written, so do not change it without also bumping
# ANALYZER_VERSION (the idempotency key in step 3.9 / writer.py).
_PROJECTION_SEED = 42


class FusionError(RuntimeError):
    """Raised when the input vectors do not have the expected shapes."""


# --- Tunables (env-backed) ---------------------------------------------------

def _fusion_alpha() -> float:
    """Weight on the visual modality. 1.0 = visual only, 0.0 = text only."""
    try:
        a = float(os.getenv("ANALYZER_FUSION_ALPHA", "0.6"))
    except ValueError:
        a = 0.6
    # Clamp into [0, 1] so a stray config value cannot produce a vector
    # that is a linear extrapolation of the two modalities.
    return max(0.0, min(1.0, a))


# --- Projection matrix (lazy, deterministic) ---------------------------------

_PROJECTION_MATRIX: Optional[np.ndarray] = None


def _projection_matrix() -> np.ndarray:
    """Return the fixed (INDEX_DIM, VISUAL_DIM) Gaussian projection matrix.

    Built once per process from `_PROJECTION_SEED`. The exact scale of
    the entries does not affect the output because we L2-normalize the
    projected vector before the weighted sum.
    """
    global _PROJECTION_MATRIX
    if _PROJECTION_MATRIX is None:
        rng = np.random.default_rng(_PROJECTION_SEED)
        _PROJECTION_MATRIX = rng.standard_normal(
            (INDEX_DIM, VISUAL_DIM)
        ).astype(np.float32)
        log.info(
            "[INFO] built fusion projection matrix %s (seed=%d)",
            _PROJECTION_MATRIX.shape, _PROJECTION_SEED,
        )
    return _PROJECTION_MATRIX


# --- Math helpers ------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector. The zero vector is returned unchanged."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


# --- Public API --------------------------------------------------------------

def fuse(
    visual: np.ndarray,
    text: np.ndarray,
    alpha: float | None = None,
) -> np.ndarray:
    """Combine a 512-d visual vector and a 384-d text vector.

    Returns a float32 numpy array of shape `(INDEX_DIM,)` ready to be
    written to `videos.content_embedding`. Raises `FusionError` if
    either input has the wrong shape.

    `alpha` overrides `ANALYZER_FUSION_ALPHA` if supplied. Useful for
    tests; production callers should leave it `None`.
    """
    if visual.shape != (VISUAL_DIM,):
        raise FusionError(
            f"expected visual shape ({VISUAL_DIM},), got {visual.shape}"
        )
    if text.shape != (TEXT_DIM,):
        raise FusionError(
            f"expected text shape ({TEXT_DIM},), got {text.shape}"
        )

    a = _fusion_alpha() if alpha is None else max(0.0, min(1.0, float(alpha)))

    projected = _projection_matrix() @ visual.astype(np.float32, copy=False)
    v_unit = _l2_normalize(projected)
    t_unit = _l2_normalize(text.astype(np.float32, copy=False))

    fused = a * v_unit + (1.0 - a) * t_unit
    return _l2_normalize(fused)
