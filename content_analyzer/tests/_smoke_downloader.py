"""Ad-hoc smoke check for the MinIO downloader (step 3.3).

Seeds a small object directly into the configured MinIO bucket, then
exercises:

  1. URL parsing (bucket + key extraction).
  2. `download_video()` round-trip — bytes match.
  3. `downloaded()` context manager — file exists inside the block,
     temp dir is gone after.
  4. Missing object raises `DownloadError`.

Cleans up the seeded object on the way out. Like `_smoke_consumer.py`,
this is a contract check for the current step, not the final 3.12 smoke.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from io import BytesIO

import content_analyzer._path  # noqa: F401
from client import get_minio  # type: ignore[import-not-found]

from content_analyzer.services.downloader import (
    DownloadError,
    download_video,
    downloaded,
    parse_minio_url,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


def _public_url(bucket: str, key: str) -> str:
    prefix = os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000").rstrip("/")
    return f"{prefix}/{bucket}/{key}"


def main() -> int:
    client = get_minio()
    bucket = os.getenv("MINIO_BUCKET", "videos")
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    suffix = uuid.uuid4().hex[:8]
    key = f"smoke_{suffix}.mp4"
    payload = b"FAKE-MP4-BYTES-" + suffix.encode() * 64  # ~520 bytes, non-trivial
    url = _public_url(bucket, key)

    client.put_object(
        bucket_name=bucket,
        object_name=key,
        data=BytesIO(payload),
        length=len(payload),
        content_type="video/mp4",
    )

    try:
        # === Case 1: URL parsing ===
        b, k = parse_minio_url(url)
        assert (b, k) == (bucket, key), f"parse failed: got ({b!r}, {k!r})"
        log.info("[OK] URL parsed to bucket=%s key=%s", b, k)

        # === Case 2: download_video round-trip ===
        path = download_video(url)
        try:
            assert path.exists(), f"downloaded file missing: {path}"
            got = path.read_bytes()
            assert got == payload, (
                f"content mismatch: {len(got)} bytes vs {len(payload)} expected"
            )
            log.info("[OK] download_video wrote %d bytes to %s",
                     path.stat().st_size, path)
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

        # === Case 3: context manager cleans up ===
        with downloaded(url) as p:
            assert p.exists() and p.read_bytes() == payload, "ctx download mismatch"
            tmpdir = p.parent
            assert tmpdir.exists(), "tmpdir should exist inside the with-block"
        assert not tmpdir.exists(), (
            f"tmpdir should have been removed on exit, still here: {tmpdir}"
        )
        log.info("[OK] downloaded() context manager cleaned up %s", tmpdir)

        # === Case 4: missing object raises DownloadError ===
        bad_url = _public_url(bucket, f"does_not_exist_{suffix}.mp4")
        try:
            download_video(bad_url)
        except DownloadError as e:
            log.info("[OK] missing object raised DownloadError: %s",
                     str(e).splitlines()[0])
        else:
            log.error("[FAIL] expected DownloadError for missing object")
            return 1
    finally:
        try:
            client.remove_object(bucket, key)
        except Exception as e:
            log.warning("[WARN] cleanup remove_object failed: %s", e)

    log.info("[OK] all 4 cases passed; cleaned up MinIO test object")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
