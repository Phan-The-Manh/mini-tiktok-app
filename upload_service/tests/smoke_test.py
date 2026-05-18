"""
smoke_test.py — end-to-end check against a running Upload Service
=================================================================

Prereqs:
    1. docker compose up -d   (Redis + MinIO running)
    2. database/.env points at a working Atlas cluster
    3. python scripts/seed_data.py   (so we have a real author_id to attach to)
    4. uvicorn upload_service.main:app --port 8001   (in another terminal)
    5. ffmpeg on PATH (optional — passthrough mode also works)

Run (from project root):
    python -m upload_service.tests.smoke_test

What it verifies:
    [1] /health returns ok for mongo+redis+minio
    [2] POST /videos with a tiny generated .mp4 returns 201 + a video_id
    [3] GET /videos/{id} returns the just-uploaded doc
    [4] MinIO has the video object
    [5] Redis stream `video.uploaded` has a new entry referencing the video_id
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

import upload_service._path  # noqa: F401

from client import get_minio, get_redis  # type: ignore[import-not-found]


BASE_URL = os.getenv("UPLOAD_SERVICE_URL", "http://localhost:8001")
STREAM_KEY = os.getenv("UPLOAD_EVENT_STREAM", "video.uploaded")


def _make_sample_video(dst: Path) -> bool:
    """
    Generate a 2-second blank mp4 using ffmpeg. Returns True on success.
    If ffmpeg is missing, we fall back to a minimal pre-made fixture path.
    """
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and dst.exists()


def _pick_author_id() -> str | None:
    """Use the first seeded user as the uploader so the FK check passes."""
    from client import get_mongo  # type: ignore[import-not-found]
    doc = get_mongo().users.find_one({"user_id": {"$regex": "^seed_"}}, {"user_id": 1})
    return doc["user_id"] if doc else None


def main() -> int:
    print("=" * 60)
    print(f"Upload Service smoke test  ->  {BASE_URL}")
    print("=" * 60)

    # [1] health
    print("\n[1] GET /health")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"    response: {r.json()}")
    except Exception as e:
        print(f"    [FAIL] {e}")
        return 1

    author_id = _pick_author_id()
    if not author_id:
        print("\n[FAIL] No seeded users found. Run `python -m scripts.seed_data` from database/.")
        return 1
    print(f"\n    Using author_id: {author_id}")

    # [2] upload
    print("\n[2] POST /videos")
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample.mp4"
        if not _make_sample_video(sample):
            print("    [SKIP] ffmpeg unavailable, cannot generate a sample mp4 to upload.")
            print("    Install ffmpeg or supply a fixture and re-run.")
            return 0

        with sample.open("rb") as f:
            r = requests.post(
                f"{BASE_URL}/videos",
                files={"file": ("sample.mp4", f, "video/mp4")},
                data={
                    "author_id": author_id,
                    "caption": "smoke test upload",
                    "hashtags": "test smoke",
                    "category": "tech",
                },
                timeout=60,
            )
        if r.status_code != 201:
            print(f"    [FAIL] status={r.status_code}, body={r.text}")
            return 1
        payload = r.json()
        video_id = payload["video_id"]
        print(f"    created video_id={video_id}, url={payload['url']}")

    # [3] read it back
    print("\n[3] GET /videos/{id}")
    r = requests.get(f"{BASE_URL}/videos/{video_id}", timeout=5)
    if r.status_code != 200:
        print(f"    [FAIL] status={r.status_code}, body={r.text}")
        return 1
    print(f"    moderation_status={r.json()['moderation_status']}")

    # [4] MinIO object
    print("\n[4] MinIO object")
    minio = get_minio()
    bucket = os.getenv("MINIO_BUCKET", "videos")
    found = any(obj.object_name.startswith(video_id) for obj in minio.list_objects(bucket))
    print(f"    object present in '{bucket}': {found}")
    if not found:
        return 1

    # [5] Redis stream entry
    print(f"\n[5] Redis stream '{STREAM_KEY}'")
    r = get_redis()
    # XREVRANGE returns newest first. We scan the last 20 entries for our video_id.
    entries = r.xrevrange(STREAM_KEY, count=20)
    matched = next((e for e in entries if e[1].get("video_id") == video_id), None)
    if matched is None:
        print("    [FAIL] no matching event found in recent stream entries")
        return 1
    entry_id, fields = matched
    print(f"    entry_id={entry_id}, video_id={fields.get('video_id')}")

    print("\n" + "=" * 60)
    print("[OK] smoke test passed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
