"""
storage.py — MinIO object storage
=================================

Thin wrapper over the shared MinIO client from `database/client.py`.
Responsibilities:
  - Ensure the video + thumbnail buckets exist on startup.
  - Upload a local file to MinIO under a deterministic object key.
  - Build the public URL used by the frontend / Mongo document.

We deliberately do NOT presign URLs here. For local dev MinIO is configured
with its default `minioadmin` credentials and serves objects publicly via
`MINIO_PUBLIC_URL` (configured to make buckets public, or to be wrapped by
a CDN in production). Presigning is a one-liner change when needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import upload_service._path  # noqa: F401  side-effect: configures sys.path + env

from client import get_minio  # type: ignore[import-not-found]


def video_bucket() -> str:
    return os.getenv("MINIO_BUCKET", "videos")


def thumbnail_bucket() -> str:
    return os.getenv("MINIO_THUMBNAIL_BUCKET", "thumbnails")


def public_url_prefix() -> str:
    # No trailing slash — we add the bucket/key suffix when building URLs.
    return os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000").rstrip("/")


def ensure_buckets() -> None:
    """Create the video + thumbnail buckets if they don't exist. Idempotent."""
    client = get_minio()
    for name in (video_bucket(), thumbnail_bucket()):
        if not client.bucket_exists(name):
            client.make_bucket(name)


def upload_file(
    local_path: str | Path,
    object_key: str,
    bucket: str,
    content_type: str,
) -> str:
    """
    Upload `local_path` to `bucket/object_key` and return its public URL.

    We pass content_type explicitly because MinIO's auto-detection from
    the filename is unreliable for browsers playing the resulting object.
    """
    client = get_minio()
    client.fput_object(
        bucket_name=bucket,
        object_name=object_key,
        file_path=str(local_path),
        content_type=content_type,
    )
    return f"{public_url_prefix()}/{bucket}/{object_key}"
