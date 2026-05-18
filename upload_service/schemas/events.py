"""
Event schemas
-------------
Payloads published to Redis Streams by this service.

Currently only one event: `video.uploaded`, consumed by the Content Analyzer
(Component #3) to generate CLIP/Whisper embeddings.

Why a dedicated schema (vs. just re-using Video)?
    The Video document is big (caption, hashtags, stats, embedding, ...).
    Stream consumers only need the minimum info to fetch + process the file.
    Keeping the event payload narrow makes the contract obvious and the
    stream cheap. Consumers re-read the full Video doc from Mongo when needed.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class VideoUploadedEvent(BaseModel):
    """Emitted on Redis Stream `video.uploaded` after a successful upload."""

    video_id: str
    author_id: str
    url: str                       # MinIO object URL for the (transcoded) video
    thumbnail_url: str | None = None
    duration_seconds: float
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    def to_stream_fields(self) -> dict[str, str]:
        """
        Redis Streams stores fields as string key/value pairs.
        We flatten the model into strings here so the publisher does not
        need to know which fields are datetimes / floats / None.
        """
        out: dict[str, str] = {}
        for k, v in self.model_dump(mode="python").items():
            if v is None:
                continue
            out[k] = v.isoformat() if isinstance(v, datetime) else str(v)
        return out
