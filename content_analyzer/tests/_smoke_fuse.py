"""Ad-hoc smoke check for modal fusion (step 3.8).

No ML dependencies — only numpy. Exercises:

  1. `fuse` on random non-zero inputs returns a (INDEX_DIM,) float32
     vector with unit L2 norm.
  2. Zero text + non-zero visual: result is unit-norm and equals the
     L2-normalized projected visual (because alpha-weighted text is
     zero and the final normalize collapses the visual scale).
  3. Zero visual + zero text: result is the zero vector (not NaN).
  4. Mismatched shapes raise `FusionError`.
  5. The projection matrix is deterministic across two calls and
     across `_projection_matrix()` invocations.
  6. Different inputs produce different outputs.
  7. alpha=1.0 ignores text; alpha=0.0 ignores visual.
  8. `_fusion_alpha` clamps out-of-range env values into [0, 1].

Run: `python -m content_analyzer.tests._smoke_fuse`
"""

from __future__ import annotations

import logging
import os

import numpy as np

import content_analyzer._path  # noqa: F401

from content_analyzer.services.fuse import (
    INDEX_DIM,
    TEXT_DIM,
    VISUAL_DIM,
    FusionError,
    _fusion_alpha,
    _l2_normalize,
    _projection_matrix,
    fuse,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


def main() -> int:
    rng = np.random.default_rng(0)

    # === Case 1: normal inputs ===
    visual = rng.standard_normal(VISUAL_DIM).astype(np.float32)
    text = rng.standard_normal(TEXT_DIM).astype(np.float32)
    vec = fuse(visual, text)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32, f"expected float32, got {vec.dtype}"
    assert vec.shape == (INDEX_DIM,), (
        f"expected shape ({INDEX_DIM},), got {vec.shape}"
    )
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5), (
        f"expected unit norm, got {np.linalg.norm(vec)}"
    )
    log.info("[OK] normal inputs -> (%d,) float32 unit-norm vector", vec.shape[0])

    # === Case 2: zero text -> matches normalized projected visual ===
    zero_text = np.zeros(TEXT_DIM, dtype=np.float32)
    vec_v_only = fuse(visual, zero_text)
    expected = _l2_normalize(_projection_matrix() @ visual)
    assert np.allclose(vec_v_only, expected, atol=1e-5), (
        "zero-text fusion should equal normalized projected visual"
    )
    assert np.isclose(np.linalg.norm(vec_v_only), 1.0, atol=1e-5)
    log.info("[OK] zero text falls back to projected visual direction")

    # === Case 3: both zero -> zero vector, not NaN ===
    zero_v = np.zeros(VISUAL_DIM, dtype=np.float32)
    vec_zero = fuse(zero_v, zero_text)
    assert vec_zero.shape == (INDEX_DIM,)
    assert not np.any(vec_zero), "both-zero input should return zero vector"
    assert not np.any(np.isnan(vec_zero)), "fusion must not return NaN"
    log.info("[OK] both-zero input returns zero vector (no NaN)")

    # === Case 4: shape mismatch -> FusionError ===
    bad_visual = np.zeros(VISUAL_DIM + 1, dtype=np.float32)
    try:
        fuse(bad_visual, text)
    except FusionError as e:
        log.info("[OK] wrong visual shape raised FusionError: %s", e)
    else:
        log.error("[FAIL] expected FusionError for wrong visual shape")
        return 1

    bad_text = np.zeros(TEXT_DIM - 1, dtype=np.float32)
    try:
        fuse(visual, bad_text)
    except FusionError as e:
        log.info("[OK] wrong text shape raised FusionError: %s", e)
    else:
        log.error("[FAIL] expected FusionError for wrong text shape")
        return 1

    # === Case 5: projection matrix is deterministic ===
    m1 = _projection_matrix()
    m2 = _projection_matrix()
    assert m1 is m2, "projection matrix should be cached (same object)"
    assert m1.shape == (INDEX_DIM, VISUAL_DIM)
    # Force a fresh build in a separate process would require subprocess; here
    # we just check that the values are stable across runs by reseeding the
    # same rng and re-deriving.
    fresh = np.random.default_rng(42).standard_normal((INDEX_DIM, VISUAL_DIM)).astype(np.float32)
    assert np.array_equal(m1, fresh), (
        "projection matrix is not the expected seeded Gaussian"
    )
    log.info("[OK] projection matrix is deterministic and cached")

    # === Case 6: different inputs -> different outputs ===
    visual_b = rng.standard_normal(VISUAL_DIM).astype(np.float32)
    text_b = rng.standard_normal(TEXT_DIM).astype(np.float32)
    vec_b = fuse(visual_b, text_b)
    assert not np.allclose(vec, vec_b, atol=1e-3), (
        "different inputs should produce different fused vectors"
    )
    log.info("[OK] distinct inputs produce distinct fused vectors")

    # === Case 7: alpha=1.0 ignores text; alpha=0.0 ignores visual ===
    vec_alpha1 = fuse(visual, text, alpha=1.0)
    vec_v_only_again = fuse(visual, zero_text, alpha=1.0)
    assert np.allclose(vec_alpha1, vec_v_only_again, atol=1e-5), (
        "alpha=1.0 should be insensitive to the text vector"
    )

    vec_alpha0 = fuse(visual, text, alpha=0.0)
    vec_t_only = fuse(zero_v, text, alpha=0.0)
    assert np.allclose(vec_alpha0, vec_t_only, atol=1e-5), (
        "alpha=0.0 should be insensitive to the visual vector"
    )
    log.info("[OK] alpha endpoints correctly ignore the unweighted modality")

    # === Case 8: _fusion_alpha clamps env values into [0, 1] ===
    saved = os.environ.get("ANALYZER_FUSION_ALPHA")
    try:
        os.environ["ANALYZER_FUSION_ALPHA"] = "2.5"
        assert _fusion_alpha() == 1.0, "above-range alpha should clamp to 1.0"
        os.environ["ANALYZER_FUSION_ALPHA"] = "-1.0"
        assert _fusion_alpha() == 0.0, "below-range alpha should clamp to 0.0"
        os.environ["ANALYZER_FUSION_ALPHA"] = "not-a-number"
        assert _fusion_alpha() == 0.6, "garbage alpha should fall back to default"
    finally:
        if saved is None:
            os.environ.pop("ANALYZER_FUSION_ALPHA", None)
        else:
            os.environ["ANALYZER_FUSION_ALPHA"] = saved
    log.info("[OK] _fusion_alpha clamps and defaults correctly")

    log.info("[OK] all fusion cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
