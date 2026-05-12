"""
Interaction schema
------------------
Every user action becomes one document in the `interactions` collection.

This is the highest-volume collection — every swipe, watch, like, comment,
share generates a document. Used for:
- Training the ranking model (each row = a labeled training example)
- Computing video engagement stats (aggregated by stream processors)
- Updating user embeddings (each action shifts the user's interests)

Because volume is huge, we apply a TTL index to auto-delete docs older than 90 days.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """
    The complete set of actions a user can take on a video.

    Watch is by far the most common (every video served generates a watch event).
    Skip is a special case: a watch event with very short watch_time.
    """
    WATCH = "watch"
    LIKE = "like"
    COMMENT = "comment"
    SHARE = "share"
    FOLLOW = "follow"
    SKIP = "skip"
    NOT_INTERESTED = "not_interested"
    REPORT = "report"


class DeviceType(str, Enum):
    MOBILE_IOS = "mobile_ios"
    MOBILE_ANDROID = "mobile_android"
    WEB = "web"


class NetworkType(str, Enum):
    WIFI = "wifi"
    CELLULAR_5G = "cellular_5g"
    CELLULAR_4G = "cellular_4g"
    UNKNOWN = "unknown"


class Interaction(BaseModel):
    """
    A single user-video interaction event.
    """

    # --- Identity ---
    interaction_id: str                       # unique ID, e.g., "i_..."
    user_id: str
    video_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # --- Session context ---
    session_id: str                           # groups events from one app open
    action: ActionType

    # --- Watch-specific fields (populated only when action == WATCH) ---
    watch_time_ms: Optional[int] = None       # how long the user watched
    video_duration_ms: Optional[int] = None   # full length of the video
    watch_pct: Optional[float] = None         # watch_time_ms / video_duration_ms (0.0-1.0)
    is_completion: bool = False               # user watched ≥95%
    is_loop: bool = False                     # user looped (replayed) the video

    # --- Device / network context ---
    device: DeviceType = DeviceType.WEB
    network: NetworkType = NetworkType.UNKNOWN

    # --- Position in feed (signal for ranking analysis) ---
    # Was this video served as #1 in the feed? #5? Position affects engagement.
    position_in_feed: Optional[int] = None

    # --- A/B testing ---
    # If this user is in an experiment, log the variant so we can analyze later.
    experiment_variant: Optional[str] = None
