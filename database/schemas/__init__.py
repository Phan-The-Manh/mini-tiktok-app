"""
schemas package
---------------
Pydantic models for every MongoDB collection.

Re-exports everything so other modules can do:

    from schemas import User, Video, Interaction
    # instead of:
    from schemas.user import User
    from schemas.video import Video
    from schemas.interaction import Interaction
"""

from schemas.user import (
    EMBEDDING_DIM,
    Demographics,
    RecentInteractions,
    User,
)
from schemas.video import (
    AITags,
    DistributionStage,
    ModerationStatus,
    Video,
    VideoStats,
)
from schemas.interaction import (
    ActionType,
    DeviceType,
    Interaction,
    NetworkType,
)
from schemas.experiment import (
    Experiment,
    TrafficAllocation,
)

__all__ = [
    # constants
    "EMBEDDING_DIM",
    # user
    "User",
    "Demographics",
    "RecentInteractions",
    # video
    "Video",
    "VideoStats",
    "AITags",
    "ModerationStatus",
    "DistributionStage",
    # interaction
    "Interaction",
    "ActionType",
    "DeviceType",
    "NetworkType",
    # experiment
    "Experiment",
    "TrafficAllocation",
]
