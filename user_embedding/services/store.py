"""
store.py — MongoDB read/write adapter (step 4.3)
================================================

The only module that talks to MongoDB for the User Embedding Service. Keeps
all collection/field knowledge in one place so the consumer, the HTTP layer,
and the batch job never hand-roll queries.

Reuses the shared connection factory (`database/client.py`) via the _path
shim — no duplicated connection code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

import user_embedding._path  # noqa: F401  side-effect: sys.path + env
from client import get_mongo  # type: ignore[import-not-found]


def _to_list(vec: Sequence[float] | np.ndarray) -> list[float]:
    """Coerce to a list of plain Python floats (BSON-safe; no np.float64)."""
    if isinstance(vec, np.ndarray):
        return [float(x) for x in vec.tolist()]
    return [float(x) for x in vec]


def get_user_vectors(user_id: str) -> Optional[tuple[list[float], list[float]]]:
    """Return `(long_term, short_term)` for a user, or None if the user does
    not exist. Either vector may be an empty list (cold start)."""
    db = get_mongo()
    doc = db.users.find_one(
        {"user_id": user_id},
        projection={"_id": 0, "long_term_embedding": 1, "short_term_embedding": 1},
    )
    if doc is None:
        return None
    return (
        doc.get("long_term_embedding") or [],
        doc.get("short_term_embedding") or [],
    )


def get_video_embedding(video_id: str) -> Optional[list[float]]:
    """Return a video's `content_embedding`, or None if the video does not
    exist. An empty list means the video exists but is not embedded yet."""
    db = get_mongo()
    doc = db.videos.find_one(
        {"video_id": video_id},
        projection={"_id": 0, "content_embedding": 1},
    )
    if doc is None:
        return None
    return doc.get("content_embedding") or []


def set_short_term(
    user_id: str,
    vec: Sequence[float] | np.ndarray,
    *,
    updated_at: Optional[datetime] = None,
) -> bool:
    """Persist `short_term_embedding` + `short_term_updated_at`. Returns True
    if a user document matched."""
    ts = updated_at or datetime.now(timezone.utc)
    res = get_mongo().users.update_one(
        {"user_id": user_id},
        {"$set": {
            "short_term_embedding": _to_list(vec),
            "short_term_updated_at": ts,
        }},
    )
    return res.matched_count > 0


def set_long_term(user_id: str, vec: Sequence[float] | np.ndarray) -> bool:
    """Persist `long_term_embedding`. Returns True if a user matched."""
    res = get_mongo().users.update_one(
        {"user_id": user_id},
        {"$set": {"long_term_embedding": _to_list(vec)}},
    )
    return res.matched_count > 0


def iter_users(limit: Optional[int] = None):
    """Yield `user_id`s for the long-term recompute batch."""
    db = get_mongo()
    cursor = db.users.find({}, projection={"_id": 0, "user_id": 1})
    if limit is not None:
        cursor = cursor.limit(limit)
    for doc in cursor:
        yield doc["user_id"]


def get_positive_interaction_videos(
    user_id: str,
    *,
    since: datetime,
) -> list[tuple[str, str, Optional[float], bool]]:
    """Return `(video_id, action, watch_pct, is_completion)` tuples for a
    user's interactions since `since`. The batch decides which are positive
    via `math_core.action_weight`; this just fetches the raw rows."""
    db = get_mongo()
    cursor = db.interactions.find(
        {"user_id": user_id, "timestamp": {"$gte": since}},
        projection={
            "_id": 0, "video_id": 1, "action": 1,
            "watch_pct": 1, "is_completion": 1,
        },
    )
    out: list[tuple[str, str, Optional[float], bool]] = []
    for d in cursor:
        out.append((
            d["video_id"],
            d.get("action", ""),
            d.get("watch_pct"),
            bool(d.get("is_completion", False)),
        ))
    return out


def get_video_embeddings(video_ids: list[str]) -> dict[str, list[float]]:
    """Batch-fetch `content_embedding` for many videos at once (one query)."""
    if not video_ids:
        return {}
    db = get_mongo()
    cursor = db.videos.find(
        {"video_id": {"$in": video_ids}},
        projection={"_id": 0, "video_id": 1, "content_embedding": 1},
    )
    return {
        d["video_id"]: (d.get("content_embedding") or [])
        for d in cursor
    }
