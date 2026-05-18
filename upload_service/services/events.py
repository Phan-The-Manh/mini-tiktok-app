"""
events.py — Redis Streams publisher
===================================

Publishes one event per successful upload:

    stream key:  $UPLOAD_EVENT_STREAM (default "video.uploaded")
    payload:     VideoUploadedEvent flattened to string fields

The Content Analyzer (Component #3) consumes this stream, fetches the video
from MinIO, runs CLIP/Whisper, and writes `content_embedding` back to Mongo.

We use a stream (XADD) rather than pub/sub so consumers can replay missed
events after a restart. This is the same pattern Kafka would give us.
"""

from __future__ import annotations

import os

import upload_service._path  # noqa: F401  side-effect: configures sys.path + env
from client import get_redis  # type: ignore[import-not-found]

from upload_service.schemas.events import VideoUploadedEvent


def stream_key() -> str:
    return os.getenv("UPLOAD_EVENT_STREAM", "video.uploaded")


def publish_video_uploaded(event: VideoUploadedEvent) -> str:
    """
    XADD the event to the stream. Returns the new entry's ID (e.g. "1700000000000-0").

    `maxlen=10000, approximate=True` caps stream growth — old entries are trimmed
    so the stream doesn't grow unbounded on a free-tier Redis. 10k events is far
    more than the demo will ever need.
    """
    r = get_redis()
    entry_id: str = r.xadd(
        name=stream_key(),
        fields=event.to_stream_fields(),
        maxlen=10_000,
        approximate=True,
    )
    return entry_id
