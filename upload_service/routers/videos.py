"""
routers/videos.py — Upload + read endpoints
===========================================

POST /videos          multipart upload (file + author_id + caption + hashtags + category)
GET  /videos/{id}     fetch the stored Video document
GET  /health          liveness + dependency check

Flow for POST /videos:
    1. Receive multipart form. Stream the upload to a temp file (no whole-file
       buffering in memory — uploads can be hundreds of MB).
    2. Transcode to mp4/h264 + extract thumbnail (or passthrough if ffmpeg missing).
    3. Upload video + thumbnail to MinIO.
    4. Insert a Video document into MongoDB. moderation_status=PENDING,
       distribution_stage=TEST_POOL_1, content_embedding=[] (filled later).
    5. XADD VideoUploadedEvent to Redis Streams `video.uploaded`.
    6. Return UploadResponse so the client can navigate to / display the new video.

Failure handling: each step uses HTTPException with a clear error code, and
the temp directory is cleaned up in a finally block regardless of outcome.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import upload_service._path  # noqa: F401  configures sys.path before importing shared schemas

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from client import get_minio, get_mongo, get_redis  # type: ignore[import-not-found]
from schemas import DistributionStage, ModerationStatus, Video  # type: ignore[import-not-found]

from upload_service.schemas.api import UploadResponse, VideoOut
from upload_service.schemas.events import VideoUploadedEvent
from upload_service.services import events as event_bus
from upload_service.services import storage, transcoder

router = APIRouter()

# Allowed input extensions. We're permissive on input format since ffmpeg
# normalizes everything. The MIME check is a sanity guard, not security.
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def _new_video_id() -> str:
    # Short, URL-safe, sortable-ish. Matches the "v_..." style of seed data
    # but uses a uuid4 suffix so collisions are impossible.
    return f"v_{uuid.uuid4().hex[:12]}"


def _parse_hashtags(raw: str) -> list[str]:
    """Accept either '#a,#b' or 'a b c' — strip '#' and whitespace, dedupe in order."""
    if not raw:
        return []
    parts = raw.replace(",", " ").split()
    seen: dict[str, None] = {}
    for p in parts:
        cleaned = p.lstrip("#").strip().lower()
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen.keys())


@router.post(
    "/videos",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a video",
)
async def upload_video(
    file: UploadFile = File(..., description="The video file (mp4/mov/webm/etc.)"),
    author_id: str = Form(..., description="user_id of the creator"),
    caption: str = Form("", description="Free-form caption"),
    hashtags: str = Form("", description="Space- or comma-separated tags"),
    category: str | None = Form(None, description="Optional category, e.g. 'cooking'"),
):
    # --- Validate the upload ---
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # --- Confirm the author exists ---
    # We don't run real auth yet (Upload Service handles that later per CLAUDE.md),
    # but we at least verify the author_id references a real user so downstream
    # services don't trip on dangling references.
    db = get_mongo()
    if db.users.find_one({"user_id": author_id}, {"_id": 1}) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"author_id '{author_id}' does not exist",
        )

    video_id = _new_video_id()
    work_dir = Path(tempfile.mkdtemp(prefix=f"upload-{video_id}-"))

    try:
        # --- Stream the upload to disk in chunks (avoid loading into RAM) ---
        src_path = work_dir / f"src{suffix}"
        with src_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                out.write(chunk)

        # --- Transcode (or passthrough if ffmpeg is missing) ---
        result = transcoder.transcode(src_path, work_dir, basename=video_id)
        if result.duration_seconds <= 0:
            # Without a duration we can't compute watch_pct downstream — reject.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not determine video duration. Install ffmpeg or upload an mp4 with valid metadata.",
            )

        # --- Upload to MinIO ---
        video_object_key = f"{video_id}.mp4" if not result.passthrough else f"{video_id}{suffix}"
        video_url = storage.upload_file(
            local_path=result.video_path,
            object_key=video_object_key,
            bucket=storage.video_bucket(),
            content_type="video/mp4" if not result.passthrough else (file.content_type or "application/octet-stream"),
        )

        thumbnail_url: str | None = None
        if result.thumbnail_path is not None:
            thumbnail_url = storage.upload_file(
                local_path=result.thumbnail_path,
                object_key=f"{video_id}.jpg",
                bucket=storage.thumbnail_bucket(),
                content_type="image/jpeg",
            )

        # --- Build + persist the Video document ---
        now = datetime.utcnow()
        video = Video(
            video_id=video_id,
            author_id=author_id,
            uploaded_at=now,
            url=video_url,
            thumbnail_url=thumbnail_url,
            duration_seconds=round(result.duration_seconds, 2),
            caption=caption,
            hashtags=_parse_hashtags(hashtags),
            category=category,
            # content_embedding stays empty — Content Analyzer fills it in.
            moderation_status=ModerationStatus.PENDING,
            distribution_stage=DistributionStage.TEST_POOL_1,
        )
        db.videos.insert_one(video.model_dump(mode="python"))

        # --- Emit the event ---
        # If this fails we've already committed to Mongo + MinIO. The user's
        # upload succeeded; the analyzer just won't pick it up automatically.
        # We log via HTTPException-free path so the client still sees 201.
        try:
            event_bus.publish_video_uploaded(VideoUploadedEvent(
                video_id=video_id,
                author_id=author_id,
                url=video_url,
                thumbnail_url=thumbnail_url,
                duration_seconds=video.duration_seconds,
                uploaded_at=now,
            ))
        except Exception as e:
            print(f"[WARN] video.uploaded publish failed for {video_id}: {e}")

        return UploadResponse(
            video_id=video_id,
            url=video_url,
            thumbnail_url=thumbnail_url,
            duration_seconds=video.duration_seconds,
            moderation_status=video.moderation_status.value,
            distribution_stage=video.distribution_stage.value,
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@router.get("/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: str):
    db = get_mongo()
    doc = db.videos.find_one({"video_id": video_id}, {"_id": 0})
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    # Hand to Pydantic so enum -> str conversion and field whitelisting happen consistently.
    return VideoOut(
        video_id=doc["video_id"],
        author_id=doc["author_id"],
        url=doc["url"],
        thumbnail_url=doc.get("thumbnail_url"),
        duration_seconds=doc["duration_seconds"],
        caption=doc.get("caption", ""),
        hashtags=doc.get("hashtags", []),
        category=doc.get("category"),
        moderation_status=str(doc.get("moderation_status", "pending")),
        distribution_stage=str(doc.get("distribution_stage", "test_pool_1")),
        uploaded_at=doc["uploaded_at"],
    )


@router.get("/health", summary="Liveness + dependency check")
def health():
    """
    Returns 200 with a per-dependency status dict.
    We don't fail the whole endpoint when one dependency is down — that lets
    operators see exactly which one to investigate.
    """
    out: dict[str, str] = {}

    try:
        get_mongo().command("ping")
        out["mongo"] = "ok"
    except Exception as e:
        out["mongo"] = f"fail: {e}"

    try:
        get_redis().ping()
        out["redis"] = "ok"
    except Exception as e:
        out["redis"] = f"fail: {e}"

    try:
        # list_buckets is cheap and proves credentials work.
        get_minio().list_buckets()
        out["minio"] = "ok"
    except Exception as e:
        out["minio"] = f"fail: {e}"

    out["ffmpeg"] = "ok" if transcoder.ffmpeg_available() else "missing (passthrough mode)"
    return out
