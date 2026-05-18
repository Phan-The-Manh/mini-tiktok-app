"""
HTTP request/response models for the Upload Service.

Form fields for POST /videos are described directly on the FastAPI route
(via `Form(...)`), so this module only defines response bodies.
"""

from datetime import datetime
from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Returned by POST /videos."""
    video_id: str
    url: str
    thumbnail_url: str | None = None
    duration_seconds: float
    moderation_status: str
    distribution_stage: str


class VideoOut(BaseModel):
    """Returned by GET /videos/{video_id} — a trimmed view of the Video doc."""
    video_id: str
    author_id: str
    url: str
    thumbnail_url: str | None = None
    duration_seconds: float
    caption: str
    hashtags: list[str]
    category: str | None
    moderation_status: str
    distribution_stage: str
    uploaded_at: datetime
