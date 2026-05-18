"""
transcoder.py — FFmpeg wrapper
==============================

Two responsibilities:
  1. probe_duration_seconds(path)  -- read video length via ffprobe
  2. transcode(path, out_dir)      -- normalize to mp4/h264 + extract thumbnail

Why normalize?
  Users upload all sorts of formats (mov, webm, hevc, weird audio codecs).
  Mobile/web players want a consistent baseline (mp4 + h264 + aac).
  We pick conservative settings (CRF 28, preset veryfast) — small files,
  good-enough quality, fast on CPU.

ffmpeg / ffprobe must be on PATH. If they are missing, transcoding is
disabled and the original file is passed through unchanged. Duration is
then estimated by reading any "Duration" line ffprobe emits, or 0.0
as a last resort. The Content Analyzer can still process the raw file
in that case — this just means slower playback and bigger storage.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscodeResult:
    video_path: Path                 # normalized mp4 (or original if passthrough)
    thumbnail_path: Path | None      # jpg thumbnail (None if disabled/failed)
    duration_seconds: float
    passthrough: bool                # True when we skipped transcoding


def _ffmpeg_bin() -> str:
    return os.getenv("FFMPEG_BIN", "ffmpeg")


def _ffprobe_bin() -> str:
    # ffprobe ships alongside ffmpeg; same dir, same naming.
    ffmpeg = _ffmpeg_bin()
    if ffmpeg.endswith("ffmpeg"):
        return ffmpeg[:-len("ffmpeg")] + "ffprobe"
    if ffmpeg.endswith("ffmpeg.exe"):
        return ffmpeg[:-len("ffmpeg.exe")] + "ffprobe.exe"
    return "ffprobe"


def ffmpeg_available() -> bool:
    """Returns True if both ffmpeg and ffprobe can be invoked."""
    return shutil.which(_ffmpeg_bin()) is not None and shutil.which(_ffprobe_bin()) is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # capture_output=True so ffmpeg's noisy stderr doesn't spam our logs.
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe_duration_seconds(path: str | Path) -> float:
    """Return the video duration in seconds, or 0.0 if unknown."""
    if not ffmpeg_available():
        return 0.0
    result = _run([
        _ffprobe_bin(),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ])
    if result.returncode != 0:
        return 0.0
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def _transcode_video(src: Path, dst: Path) -> bool:
    """Re-encode `src` to mp4/h264/aac at `dst`. Returns True on success."""
    result = _run([
        _ffmpeg_bin(),
        "-y",                       # overwrite output
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "veryfast",      # fast encode, fine on a laptop
        "-crf", "28",               # decent quality / small file
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",  # web-friendly: moov atom at the front
        "-vf", "scale='min(720,iw)':-2",  # cap width at 720, keep aspect, even height
        str(dst),
    ])
    return result.returncode == 0 and dst.exists()


def _extract_thumbnail(src: Path, dst: Path) -> bool:
    """Grab a single frame ~1s in. Returns True on success."""
    result = _run([
        _ffmpeg_bin(),
        "-y",
        "-ss", "00:00:01",          # seek before -i = fast seek
        "-i", str(src),
        "-vframes", "1",
        "-q:v", "3",                # 1-31, lower = better
        str(dst),
    ])
    return result.returncode == 0 and dst.exists()


def transcode(src: str | Path, out_dir: str | Path, basename: str) -> TranscodeResult:
    """
    Normalize `src` to mp4/h264 in `out_dir/{basename}.mp4` and produce a thumbnail.

    If ffmpeg is unavailable or ENABLE_TRANSCODE=false, falls back to passthrough:
    returns the original path and no thumbnail. The caller is expected to upload
    whatever `video_path` points at.
    """
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enabled = os.getenv("ENABLE_TRANSCODE", "true").lower() == "true"
    if not enabled or not ffmpeg_available():
        return TranscodeResult(
            video_path=src,
            thumbnail_path=None,
            duration_seconds=probe_duration_seconds(src),
            passthrough=True,
        )

    dst_video = out_dir / f"{basename}.mp4"
    dst_thumb = out_dir / f"{basename}.jpg"

    if not _transcode_video(src, dst_video):
        # If transcoding fails (e.g. corrupt input), fall back to the original.
        # We'd rather store something than reject the upload outright.
        return TranscodeResult(
            video_path=src,
            thumbnail_path=None,
            duration_seconds=probe_duration_seconds(src),
            passthrough=True,
        )

    thumb_path = dst_thumb if _extract_thumbnail(dst_video, dst_thumb) else None
    return TranscodeResult(
        video_path=dst_video,
        thumbnail_path=thumb_path,
        duration_seconds=probe_duration_seconds(dst_video),
        passthrough=False,
    )
