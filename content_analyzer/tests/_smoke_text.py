"""Ad-hoc smoke check for MiniLM text encoding (step 3.7).

Exercises:

  1. `_combine` joins caption + transcript with the expected separator
     and drops empty / whitespace-only segments.
  2. (MiniLM deps available) `TextEncoder` loads and reports dim=384.
  3. `encode(caption, transcript)` returns a 1-D float32 vector of the
     correct length, with at least one non-zero entry, for a normal
     non-empty input.
  4. `encode("", "")` and `encode(None, "  ")` return the zero vector
     of the correct length (the documented "no text signal" sentinel).
  5. Two different non-empty inputs produce different non-zero vectors.
  6. Very long transcripts do not raise — tokenizer truncation kicks in.

ML-dependent cases skip cleanly with a [WARN] when
`sentence-transformers` or `torch` are missing (typical state before
`pip install -r content_analyzer/requirements.txt`).
"""

from __future__ import annotations

import logging

import numpy as np

import content_analyzer._path  # noqa: F401

from content_analyzer.services.text import (
    MINILM_L6_DIM,
    _combine,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


def _deps_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch                  # noqa: F401
    except Exception as e:
        log.warning("[WARN] sentence-transformers / torch unavailable: %s", e)
        return False
    return True


def main() -> int:
    # === Case 1: _combine ===
    assert _combine("hello", "world") == "hello. world"
    assert _combine("hello", "") == "hello"
    assert _combine("", "world") == "world"
    assert _combine(None, None) == ""
    assert _combine("  ", "  ") == ""
    assert _combine("  hello  ", "  world  ") == "hello. world"
    log.info("[OK] _combine handles caption/transcript permutations")

    if not _deps_available():
        log.warning(
            "[WARN] skipping MiniLM cases "
            "(pip install -r content_analyzer/requirements.txt)"
        )
        return 0

    from content_analyzer.services.text import TextEncoder

    enc = TextEncoder()
    enc.load()
    assert enc.embedding_dim == MINILM_L6_DIM, (
        f"expected dim {MINILM_L6_DIM}, got {enc.embedding_dim}"
    )
    log.info("[OK] text encoder loaded (dim=%d)", enc.embedding_dim)

    # === Case 3: normal input -> 1-D float32 vector ===
    vec = enc.encode("a person riding a skateboard", "today I went to the park")
    assert isinstance(vec, np.ndarray), f"expected ndarray, got {type(vec)}"
    assert vec.dtype == np.float32, f"expected float32, got {vec.dtype}"
    assert vec.shape == (MINILM_L6_DIM,), (
        f"expected shape ({MINILM_L6_DIM},), got {vec.shape}"
    )
    assert np.any(vec != 0), "expected non-zero vector for non-empty input"
    log.info("[OK] non-empty input produced (%d,) non-zero float32 vector", vec.shape[0])

    # === Case 4: empty input -> zero vector ===
    zero = enc.encode("", "")
    assert zero.shape == (MINILM_L6_DIM,)
    assert zero.dtype == np.float32
    assert not np.any(zero), "expected all-zero vector for empty input"
    log.info("[OK] empty input produced zero vector")

    zero2 = enc.encode(None, "   ")
    assert not np.any(zero2), "expected all-zero vector for None+whitespace input"
    log.info("[OK] None / whitespace input produced zero vector")

    # === Case 5: different inputs -> different non-zero vectors ===
    v_a = enc.encode("a cat", None)
    v_b = enc.encode("a dog", None)
    assert not np.allclose(v_a, v_b), "expected different vectors for different inputs"
    assert np.any(v_a) and np.any(v_b)
    log.info("[OK] distinct inputs produce distinct non-zero vectors")

    # === Case 6: very long transcript -> no raise ===
    long_text = "the quick brown fox jumps over the lazy dog. " * 500  # ~22 KB
    v_long = enc.encode("short caption", long_text)
    assert v_long.shape == (MINILM_L6_DIM,)
    assert np.any(v_long)
    log.info(
        "[OK] long transcript (%d chars) handled via tokenizer truncation",
        len(long_text),
    )

    log.info("[OK] all text cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
