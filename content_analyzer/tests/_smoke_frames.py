"""Ad-hoc smoke check for the ffmpeg frame sampler (step 3.4).

Generates a 4-second synthetic test video with `ffmpeg -f lavfi testsrc`,
then exercises:

  1. Default-count sampling: produces exactly ANALYZER_FRAME_COUNT frames.
  2. Explicit count: `count=4` produces 4 frames.
  3. Frame ordering: returned paths are in lexicographic (== time) order
     and each file is a non-empty JPEG.
  4. Short-video grace: a 0.5s video still yields at least one frame.
  5. Context manager cleanup: temp dir is removed on exit.
  6. Missing video raises FrameExtractionError.

Skips cleanly (exit 0 with a [WARN]) when ffmpeg is not on PATH.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import content_analyzer._path  # noqa: F401

from content_analyzer.services.frames import (
    FrameExtractionError,
    _default_frame_count,
    _ffmpeg_bin,
    extract_frames,
    frames,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


def _ffmpeg_available() -> bool:
    return shutil.which(_ffmpeg_bin()) is not None


def _make_testsrc(dst: Path, duration_seconds: float = 4.0) -> bool:
    """Synthesize a small test video using ffmpeg's lavfi testsrc filter."""
    result = subprocess.run(
        [
            _ffmpeg_bin(),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration_seconds}:size=320x240:rate=10",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _is_jpeg(path: Path) -> bool:
    with path.open("rb") as f:
        header = f.read(3)
    return header[:2] == b"\xff\xd8" and len(header) >= 3


def main() -> int:
    if not _ffmpeg_available():
        log.warning("[WARN] ffmpeg not on PATH; skipping frames smoke test")
        return 0

    workdir = Path(tempfile.mkdtemp(prefix="smoke_frames_"))
    try:
        # --- Generate a 4s test video ---
        video = workdir / "test.mp4"
        if not _make_testsrc(video, duration_seconds=4.0):
            log.error("[FAIL] could not synthesize test video with ffmpeg")
            return 1
        log.info("[OK] generated synthetic test video at %s (%d bytes)",
                 video, video.stat().st_size)

        # === Case 1: default count ===
        default_n = _default_frame_count()
        out_dir = workdir / "default"
        produced = extract_frames(video, out_dir)
        assert len(produced) == default_n, (
            f"expected {default_n} frames, got {len(produced)}"
        )
        log.info("[OK] default count produced %d frames", len(produced))

        # === Case 2: explicit count ===
        out_dir2 = workdir / "explicit"
        produced2 = extract_frames(video, out_dir2, count=4)
        assert len(produced2) == 4, f"expected 4 frames, got {len(produced2)}"
        log.info("[OK] explicit count=4 produced %d frames", len(produced2))

        # === Case 3: ordering + valid JPEGs ===
        names = [p.name for p in produced2]
        assert names == sorted(names), f"frames not in lexicographic order: {names}"
        for p in produced2:
            assert p.stat().st_size > 0, f"empty frame: {p}"
            assert _is_jpeg(p), f"not a JPEG: {p}"
        log.info("[OK] frames ordered and valid JPEGs: %s", names)

        # === Case 4: short video (0.5s) still yields >=1 frame ===
        short_video = workdir / "short.mp4"
        if _make_testsrc(short_video, duration_seconds=0.5):
            out_short = workdir / "short_out"
            produced_short = extract_frames(short_video, out_short, count=8)
            assert len(produced_short) >= 1, (
                f"short video should yield >=1 frame, got {len(produced_short)}"
            )
            log.info("[OK] short 0.5s video yielded %d frame(s)", len(produced_short))
        else:
            log.warning("[WARN] could not synthesize short video; skipping case 4")

        # === Case 5: context manager cleans up ===
        with frames(video, count=3) as imgs:
            assert len(imgs) == 3, f"ctx mgr expected 3 frames, got {len(imgs)}"
            tmpdir = imgs[0].parent
            assert tmpdir.exists(), "tmpdir should exist inside the with-block"
        assert not tmpdir.exists(), (
            f"tmpdir should have been removed on exit, still here: {tmpdir}"
        )
        log.info("[OK] frames() context manager cleaned up %s", tmpdir)

        # === Case 6: missing video raises FrameExtractionError ===
        try:
            extract_frames(workdir / "does_not_exist.mp4", workdir / "nope")
        except FrameExtractionError as e:
            log.info("[OK] missing video raised FrameExtractionError: %s",
                     str(e).splitlines()[0])
        else:
            log.error("[FAIL] expected FrameExtractionError for missing video")
            return 1

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    log.info("[OK] all frame-sampler cases passed; cleaned up %s", workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
