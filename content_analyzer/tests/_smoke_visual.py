"""Ad-hoc smoke check for the CLIP visual encoder (step 3.5).

Generates a handful of synthetic JPEG frames with PIL (no ffmpeg
required) and exercises:

  1. `load()` succeeds and reports the expected projection_dim (512).
  2. `encode()` returns a numpy float32 array of shape (dim,).
  3. Determinism: encoding the same frames twice yields the same vector.
  4. Discrimination: encoding visually different frames produces a
     different vector (cosine < 1 - epsilon).
  5. Corrupt frames are dropped, not crashed on; an all-corrupt input
     raises `VisualEncodeError`.

This pulls CLIP weights (~150 MB) on first run. Skips cleanly with a
[WARN] if `torch` / `transformers` are not installed.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import content_analyzer._path  # noqa: F401


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


def _deps_available() -> bool:
    """All of numpy / Pillow / torch / transformers are required for the
    real path. Any miss means the analyzer requirements aren't installed
    yet — skip cleanly rather than crash."""
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        log.warning("[WARN] analyzer ML deps unavailable: %s", e)
        return False
    return True


def main() -> int:
    if not _deps_available():
        log.warning(
            "[WARN] skipping visual smoke test "
            "(pip install -r content_analyzer/requirements.txt)"
        )
        return 0

    # Imports are deferred until after the skip guard so an unconfigured
    # env can still execute this file without ImportError.
    import numpy as np
    from PIL import Image

    from content_analyzer.services.visual import (
        CLIPVisualEncoder,
        VisualEncodeError,
    )

    def _make_solid_frames(dst_dir: Path, color: tuple[int, int, int], n: int) -> list[Path]:
        """Write `n` solid-color 64x64 JPEGs."""
        out: list[Path] = []
        for i in range(n):
            p = dst_dir / f"{color[0]:03d}_{color[1]:03d}_{color[2]:03d}_{i:03d}.jpg"
            Image.new("RGB", (64, 64), color=color).save(p, "JPEG")
            out.append(p)
        return out

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    workdir = Path(tempfile.mkdtemp(prefix="smoke_visual_"))
    try:
        # === Case 1: load ===
        encoder = CLIPVisualEncoder()
        encoder.load()
        assert encoder.embedding_dim == 512, (
            f"expected 512-d, got {encoder.embedding_dim}"
        )
        log.info("[OK] CLIP loaded with embedding_dim=%d", encoder.embedding_dim)

        # === Case 2: shape + dtype ===
        red_frames = _make_solid_frames(workdir, (220, 30, 30), n=8)
        v_red = encoder.encode(red_frames)
        assert v_red.shape == (encoder.embedding_dim,), f"shape={v_red.shape}"
        assert v_red.dtype == np.float32, f"dtype={v_red.dtype}"
        log.info("[OK] encode() returned shape=%s dtype=%s", v_red.shape, v_red.dtype)

        # === Case 3: determinism ===
        v_red_again = encoder.encode(red_frames)
        assert np.allclose(v_red, v_red_again, atol=1e-5), (
            "same input should give the same vector"
        )
        log.info("[OK] determinism: identical frames -> identical vector")

        # === Case 4: discrimination ===
        blue_frames = _make_solid_frames(workdir, (30, 30, 220), n=8)
        v_blue = encoder.encode(blue_frames)
        cos = _cosine(v_red, v_blue)
        assert cos < 0.999, (
            f"red vs blue should differ; cosine={cos:.4f}"
        )
        log.info("[OK] discrimination: red vs blue cosine=%.4f (< 0.999)", cos)

        # === Case 5: corrupt-frame handling ===
        bad_dir = workdir / "bad"
        bad_dir.mkdir()
        bad_path = bad_dir / "corrupt.jpg"
        bad_path.write_bytes(b"NOT A JPEG")

        # Mix of one good + one bad -> still produces a vector, warns about the bad one.
        mixed = [red_frames[0], bad_path]
        v_mixed = encoder.encode(mixed)
        assert v_mixed.shape == (encoder.embedding_dim,)
        log.info("[OK] mixed good+corrupt input still produced a vector")

        # All-bad input -> VisualEncodeError.
        try:
            encoder.encode([bad_path])
        except VisualEncodeError as e:
            log.info("[OK] all-corrupt input raised VisualEncodeError: %s", e)
        else:
            log.error("[FAIL] expected VisualEncodeError for all-corrupt input")
            return 1

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    log.info("[OK] all CLIP encoder cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
