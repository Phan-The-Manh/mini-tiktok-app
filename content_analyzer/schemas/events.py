"""
events.py — Redis Stream event payloads (consumer side)
=======================================================

`VideoUploadedEvent` mirrors the Pydantic model the Upload Service publishes
on the `video.uploaded` stream. We deliberately re-declare it here (rather
than importing from `upload_service`) so the Content Analyzer can be built
and tested without taking a hard Python dependency on the upload service.
The shared contract is the wire format — the set of string fields in the
Redis stream entry — not the Python class.

`VideoEmbeddedEvent` is reserved for later: a downstream consumer can fan
out on it once we add one. Nothing emits it today.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class VideoUploadedEvent(BaseModel):
    """Emitted by the Upload Service on Redis Stream `video.uploaded`."""

    video_id: str
    author_id: str
    url: str
    thumbnail_url: Optional[str] = None
    duration_seconds: float
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_stream_fields(cls, fields: dict[str, str]) -> "VideoUploadedEvent":
        """Inverse of the publisher's `to_stream_fields`: cast strings back to
        typed fields. Missing required keys raise ValidationError; missing
        optional keys are tolerated."""
        data: dict = dict(fields)
        if "duration_seconds" in data:
            data["duration_seconds"] = float(data["duration_seconds"])
        if "uploaded_at" in data:
            data["uploaded_at"] = _parse_dt(data["uploaded_at"])
        return cls.model_validate(data)


class VideoEmbeddedEvent(BaseModel):
    """Optional event published once a video has a non-empty content_embedding.
    Reserved for downstream consumers (e.g., a recall warm-up cache)."""

    video_id: str
    analyzer_version: str
    embedding_dim: int
    embedded_at: datetime = Field(default_factory=datetime.utcnow)
