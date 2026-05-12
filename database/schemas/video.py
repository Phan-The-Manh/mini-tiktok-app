"""
Video schema
------------
A piece of content uploaded by a creator. Stored in the `videos` collection.

The most important field is `content_embedding` — this is what
MongoDB Atlas Vector Search will index, and what powers the recall stage.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModerationStatus(str, Enum):
    """
    Tracks whether a video has passed safety/policy checks.

    Inheriting from (str, Enum) means values serialize as strings to MongoDB
    while still giving us autocomplete and typo protection in code.
    """
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class DistributionStage(str, Enum):
    """
    Cold-start distribution stages.

    A new video starts in TEST_POOL_1 (shown to ~50 users).
    If engagement passes thresholds, it gets promoted to TEST_POOL_2.
    Eventually reaches MAINSTREAM where it's eligible for full recall.

    This is how new creators can go viral despite having no audience.
    """
    TEST_POOL_1 = "test_pool_1"
    TEST_POOL_2 = "test_pool_2"
    MAINSTREAM = "mainstream"


class AITags(BaseModel):
    """
    Output of the Content Analyzer (Component #3).

    Filled in by CLIP (objects/scenes), Whisper (transcript), and
    optional models like YOLO (detection). Empty until processed.
    """
    objects: list[str] = Field(default_factory=list)        # ["coffee_cup", "person"]
    scene: Optional[str] = None                             # "indoor_kitchen"
    actions: list[str] = Field(default_factory=list)        # ["pouring", "drinking"]
    transcript: Optional[str] = None                        # whisper output
    detected_language: Optional[str] = None                 # "en"
    music_id: Optional[str] = None                          # ID of background music


class VideoStats(BaseModel):
    """
    Engagement counters. Updated by stream processors as events arrive.
    Denormalized for fast access during ranking (don't aggregate on every read).
    """
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    avg_watch_pct: float = 0.0      # 0.0 to 1.0
    completion_rate: float = 0.0    # fraction of viewers who watched to the end


class Video(BaseModel):
    """
    A video document in MongoDB.

    Maps directly to a document in the `videos` collection.
    """

    # --- Identity ---
    video_id: str                              # human-readable ID, e.g., "v_xyz789"
    author_id: str                             # references User.user_id
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    # --- Storage ---
    url: str                                   # MinIO/S3 URL of the video file
    thumbnail_url: Optional[str] = None
    duration_seconds: float

    # --- Creator-provided metadata ---
    caption: str = ""
    hashtags: list[str] = Field(default_factory=list)
    category: Optional[str] = None             # e.g., "cooking" — used for diversity rules

    # --- AI-extracted metadata (filled by Content Analyzer) ---
    ai_tags: AITags = Field(default_factory=AITags)

    # --- The all-important embedding ---
    # 384-dimensional vector. Empty until Content Analyzer processes the video.
    # This field is indexed by MongoDB Atlas Vector Search.
    content_embedding: list[float] = Field(default_factory=list)

    # --- Engagement (denormalized) ---
    stats: VideoStats = Field(default_factory=VideoStats)

    # --- Distribution (cold-start) ---
    distribution_stage: DistributionStage = DistributionStage.TEST_POOL_1
    next_review_at: Optional[datetime] = None  # when to re-evaluate stage promotion

    # --- Safety ---
    moderation_status: ModerationStatus = ModerationStatus.PENDING
