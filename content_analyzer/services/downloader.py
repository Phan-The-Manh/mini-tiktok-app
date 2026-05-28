"""
downloader.py — MinIO object -> local temp file
==============================================

Step 3.3 of the Content Analyzer build.

The handler receives a `VideoUploadedEvent` whose `url` field looks like
`http://localhost:9000/videos/v_abc123.mp4` (path-style MinIO/S3). Before
we can run ffmpeg / CLIP / Whisper over the video we need the bytes on
local disk. This module is the single place that bridges "URL on the
event" to "Path on the local filesystem".

Two surfaces:

- `download_video(url, dest_dir=None) -> Path`
    Direct call. Creates a fresh temp directory if `dest_dir` is omitted.
    Caller owns cleanup.

- `downloaded(url) -> contextmanager[Path]`
    Preferred surface for the consumer pipeline. Creates a temp dir,
    yields the downloaded path, and removes the directory on exit
    *whether the body succeeded or raised*. This satisfies the TODO 3.3
    requirement: "Clean up temp files on success and failure."

URL parsing is path-style (`<host>/<bucket>/<key>`), which matches both
the local MinIO setup and the documented migration path to Cloudflare R2
or Backblaze B2. Virtual-hosted style (`<bucket>.<host>/<key>`) would
need a different parser; we'd add it then, not preemptively.

Failures are wrapped in `DownloadError` so the consumer's retry/DLQ path
sees a stable exception type regardless of which MinIO/SDK error fired.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from minio.error import S3Error  # type: ignore[import-not-found]

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env
from client import get_minio  # type: ignore[import-not-found]


class DownloadError(RuntimeError):
    """Raised when a video cannot be fetched from object storage."""


def parse_minio_url(url: str) -> tuple[str, str]:
    """Extract `(bucket, object_key)` from a path-style MinIO URL.

    `http://host:9000/videos/v_abc.mp4`  -> `("videos", "v_abc.mp4")`
    `http://host:9000/videos/sub/v.mp4`  -> `("videos", "sub/v.mp4")`
    """
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    parts = path.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise DownloadError(
            f"cannot parse bucket+key from URL (expected path-style): {url!r}"
        )
    return parts[0], parts[1]


def download_video(url: str, dest_dir: Path | None = None) -> Path:
    """Download the object referenced by `url` to a local file.

    If `dest_dir` is omitted, a fresh temp directory is created. The
    returned `Path` lives inside that directory and keeps the object key's
    basename (so `.mp4` etc. is preserved for downstream ffmpeg calls).

    The caller is responsible for removing the directory; prefer the
    `downloaded()` context manager unless you have a reason not to.
    """
    bucket, key = parse_minio_url(url)

    target_dir = Path(dest_dir) if dest_dir is not None else Path(
        tempfile.mkdtemp(prefix="content_analyzer_")
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / Path(key).name

    client = get_minio()
    try:
        client.fget_object(bucket_name=bucket, object_name=key, file_path=str(dest))
    except S3Error as e:
        raise DownloadError(
            f"failed to download {bucket}/{key} from MinIO: {e}"
        ) from e
    except Exception as e:
        raise DownloadError(
            f"unexpected error downloading {bucket}/{key}: {e}"
        ) from e

    if not dest.exists() or dest.stat().st_size == 0:
        raise DownloadError(
            f"downloaded file is missing or empty: {dest}"
        )
    return dest


@contextmanager
def downloaded(url: str) -> Iterator[Path]:
    """Download `url` into a fresh temp dir; clean the dir up on exit.

    Usage:
        with downloaded(event.url) as video_path:
            frames = sample_frames(video_path)
            ...
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="content_analyzer_"))
    try:
        yield download_video(url, dest_dir=tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
