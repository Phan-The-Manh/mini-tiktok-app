"""
frames.py — ffmpeg frame sampling
=================================

Step 3.4 of the Content Analyzer build.

Sample N evenly-spaced frames from a video file and write them to a
temp directory as JPEGs. The visual encoder (CLIP, step 3.5) then
mean-pools the per-frame embeddings into a single visual vector.

Sampling strategy
-----------------
Given a video of duration D and a target frame count N, we pick
timestamps at the midpoint of each of N equal segments:

    t_i = D * (i + 0.5) / N   for i in 0..N-1

Midpoints (rather than 0..D-1 endpoints) avoid the common pathology of
sampling the first black frame and the trailing fade-out — both are
common in user-generated content and would degrade the visual vector.

Robustness
----------
- If duration is unknown (ffprobe failed or the container has no
  metadata), we fall back to integer-second seeks (0, 1, 2, ...).
  ffmpeg will simply refuse to write past the end of the video; we
  drop those and continue.
- If the video is shorter than N seconds, midpoints still fall in
  range — fractional seconds work with `-ss`.
- Per-frame failures do not abort the run; the worst case is a
  shorter list returned. We only raise `FrameExtractionError` when
  *zero* frames were produced (the video is unusable for embedding).

Two surfaces, matching `downloader.py`:

- `extract_frames(video_path, dest_dir, count=...) -> list[Path]`
    Caller owns the directory.

- `frames(video_path, count=...) -> contextmanager[list[Path]]`
    Preferred surface for the consumer pipeline. Creates a fresh temp
    dir, yields the frame list, and removes the directory on exit
    whether the body succeeded or raised.

ffmpeg/ffprobe must be on PATH (or `FFMPEG_BIN` set in the env). Unlike
the Upload Service there is no passthrough fallback here — CLIP needs
pixels, and there is no other way to extract them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env


class FrameExtractionError(RuntimeError):
    """Raised when no frames could be extracted from a video."""


# --- ffmpeg / ffprobe binaries -----------------------------------------------

def _ffmpeg_bin() -> str:
    return os.getenv("FFMPEG_BIN", "ffmpeg")


def _ffprobe_bin() -> str:
    """ffprobe ships alongside ffmpeg; derive its path from FFMPEG_BIN."""
    ffmpeg = _ffmpeg_bin()
    if ffmpeg.endswith("ffmpeg"):
        return ffmpeg[: -len("ffmpeg")] + "ffprobe"
    if ffmpeg.endswith("ffmpeg.exe"):
        return ffmpeg[: -len("ffmpeg.exe")] + "ffprobe.exe"
    return "ffprobe"


def _default_frame_count() -> int:
    raw = os.getenv("ANALYZER_FRAME_COUNT", "8")
    try:
        n = int(raw)
    except ValueError:
        n = 8
    return max(1, n)


# --- internals ---------------------------------------------------------------

def _probe_duration_seconds(video_path: Path) -> float:
    """Return the video duration in seconds, or 0.0 if unknown."""
    try:
        result = subprocess.run(
            [
                _ffprobe_bin(),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 0.0
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _seek_timestamps(duration: float, count: int) -> list[float]:
    """Evenly-spaced midpoints inside [0, duration].

    Falls back to integer-second seeks when duration is unknown; ffmpeg
    will fail past EOF and those frames are dropped by the caller.
    """
    if count <= 0:
        return []
    if duration > 0:
        return [duration * (i + 0.5) / count for i in range(count)]
    return [float(i) for i in range(count)]


def _grab_one_frame(video_path: Path, timestamp: float, dest: Path) -> bool:
    """Extract a single JPEG at `timestamp`. Returns True on success."""
    try:
        result = subprocess.run(
            [
                _ffmpeg_bin(),
                "-y",
                "-ss", f"{timestamp:.3f}",   # fast seek (placed before -i)
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "3",                 # 1-31, lower = better quality
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


# --- public API --------------------------------------------------------------

def extract_frames(
    video_path: str | Path,
    dest_dir: str | Path,
    count: int | None = None,
) -> list[Path]:
    """Sample `count` evenly-spaced frames from `video_path` into `dest_dir`.

    Returns the produced frames in time order. The list is at most
    `count` long and may be shorter when the video is too short to
    yield every requested timestamp. Raises `FrameExtractionError` if
    zero frames were produced.
    """
    video_path = Path(video_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise FrameExtractionError(f"video file missing or empty: {video_path}")

    n = count if count is not None else _default_frame_count()
    duration = _probe_duration_seconds(video_path)
    timestamps = _seek_timestamps(duration, n)

    produced: list[Path] = []
    for i, ts in enumerate(timestamps):
        out = dest_dir / f"frame_{i:03d}.jpg"
        if _grab_one_frame(video_path, ts, out):
            produced.append(out)

    if not produced:
        raise FrameExtractionError(
            f"ffmpeg produced no frames from {video_path} "
            f"(duration={duration:.2f}s, requested={n})"
        )
    return produced


@contextmanager
def frames(
    video_path: str | Path,
    count: int | None = None,
) -> Iterator[list[Path]]:
    """Sample frames into a fresh temp dir; clean it up on exit.

    Usage:
        with frames(video_path) as imgs:
            embeddings = encode_clip(imgs)
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="content_analyzer_frames_"))
    try:
        yield extract_frames(video_path, tmpdir, count=count)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
