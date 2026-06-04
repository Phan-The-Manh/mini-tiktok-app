"""
main.py — FastAPI entrypoint
============================

Run with (from project root, venv active):

    uvicorn upload_service.main:app --reload --port 8001

Or:

    python -m upload_service.main

On startup we ensure the MinIO buckets exist, so the very first upload
doesn't fail with NoSuchBucket.
"""

from __future__ import annotations

import os

import upload_service._path  # noqa: F401  configures sys.path + env first

from fastapi import FastAPI

from upload_service.routers import ui, videos
from upload_service.services import storage

app = FastAPI(
    title="Mini-TikTok Upload Service",
    description="Accepts video uploads, transcodes, stores in MinIO, and emits video.uploaded events.",
    version="0.1.0",
)

app.include_router(videos.router)
app.include_router(ui.router)


@app.on_event("startup")
def _on_startup() -> None:
    # Idempotent. Logs a clear message so first-run users see what happened.
    try:
        storage.ensure_buckets()
        print(f"[OK] MinIO buckets ready: {storage.video_bucket()}, {storage.thumbnail_bucket()}")
    except Exception as e:
        # Don't crash the app — health endpoint will surface the MinIO failure.
        print(f"[WARN] Could not ensure MinIO buckets on startup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "upload_service.main:app",
        host=os.getenv("UPLOAD_SERVICE_HOST", "0.0.0.0"),
        port=int(os.getenv("UPLOAD_SERVICE_PORT", "8001")),
        reload=False,
    )
