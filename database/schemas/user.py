"""
User schema
-----------
Represents a person using the app. Stored in the `users` collection.

Each user has:
- A stable identifier (user_id)
- Demographics (used for cold-start recommendations)
- Two embeddings: long-term (slow-moving interests) and short-term (current session)
- A summary of recent activity (denormalized for fast access during ranking)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# Embedding dimensionality — matches sentence-transformers/all-MiniLM-L6-v2.
# Defined here so other modules can `from schemas.user import EMBEDDING_DIM`.
EMBEDDING_DIM = 384


class Demographics(BaseModel):
    """
    Optional demographic info. Used for cold-start: when we have no
    interaction history yet, we recommend things popular within
    similar demographics.
    """
    age_range: Optional[str] = None       # e.g., "18-24", "25-34"
    country: Optional[str] = None         # ISO country code, e.g., "SG"
    language: Optional[str] = None        # ISO language code, e.g., "en"


class RecentInteractions(BaseModel):
    """
    Denormalized summary of recent activity.

    'Denormalized' = duplicated data, kept fresh by background workers.
    We store this on the user doc so the ranking service can read it
    in one DB call instead of running an aggregation over interactions.
    """
    last_50_video_ids: list[str] = Field(default_factory=list)
    liked_categories: dict[str, int] = Field(default_factory=dict)   # {"cooking": 12, "tech": 8}
    avg_watch_time_sec: float = 0.0


class User(BaseModel):
    """
    A user document in MongoDB.

    Maps directly to a document in the `users` collection.
    """

    # --- Identity ---
    user_id: str                           # human-readable ID, e.g., "u_abc123"
    username: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # --- Demographics (optional) ---
    demographics: Demographics = Field(default_factory=Demographics)

    # --- Embeddings ---
    # Long-term embedding: refreshed nightly via batch job.
    # Captures stable, durable interests built from weeks of activity.
    long_term_embedding: list[float] = Field(default_factory=list)

    # Short-term embedding: updated in near real-time as user interacts.
    # Captures current-session interests.
    short_term_embedding: list[float] = Field(default_factory=list)
    short_term_updated_at: Optional[datetime] = None

    # --- Activity summary (denormalized for fast ranking) ---
    recent_interactions: RecentInteractions = Field(default_factory=RecentInteractions)

    # --- Diversity helpers (used during re-ranking) ---
    # We keep last seen authors/categories to enforce diversity in the feed.
    recently_seen_authors: list[str] = Field(default_factory=list)
    recently_seen_categories: list[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "u_abc123",
                "username": "alice",
                "demographics": {"age_range": "25-34", "country": "SG", "language": "en"},
                "long_term_embedding": [0.21, -0.13],   # truncated for example
                "short_term_embedding": [0.45, 0.02],
            }
        }
    }
