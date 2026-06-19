"""
update.py — apply-interaction + read-query orchestration
========================================================

Ties the math core (4.2), the Mongo adapter (4.3), and the Redis cache (4.4)
together. Both the `user.action` consumer (4.6) and the dev HTTP endpoint
call `apply_interaction`; the GET endpoint calls `get_query_vector`. Keeping
both here means the wire layers stay thin and the read/write paths share one
implementation.

Write-through policy: every successful short-term update writes the cache AND
Mongo synchronously, so the cache is never the only copy of a user's drift.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

import user_embedding._path  # noqa: F401  side-effect: sys.path + env

from user_embedding.schemas.events import UserActionEvent
from user_embedding.services import cache, store
from user_embedding.services import math_core as mc

DECAY = float(os.getenv("USER_EMBEDDING_DECAY", "0.9"))
BETA = float(os.getenv("USER_EMBEDDING_BLEND_BETA", "0.5"))


# --- Errors ------------------------------------------------------------------

class UnknownUser(Exception):
    """The user_id has no document. Permanent — DLQ, do not retry."""


class UnknownVideo(Exception):
    """The video_id has no document (deleted). Permanent — DLQ, do not retry."""


class VideoNotEmbedded(Exception):
    """The video exists but has no content_embedding yet. Transient — the
    Content Analyzer may catch up, so this is retryable."""


# --- Helpers -----------------------------------------------------------------

def _load_short_term(user_id: str, mongo_short: list[float]) -> list[float]:
    """Prefer the cached short-term vector; fall back to the Mongo copy."""
    cached = cache.get_short_term(user_id)
    if cached is not None:
        return cached
    return mongo_short


def _load_video_vec(video_id: str) -> list[float]:
    """Cache-first read of a video's content_embedding. Raises UnknownVideo /
    VideoNotEmbedded so callers can classify retryability."""
    cached = cache.get_video(video_id)
    if cached is not None:
        if not cached:
            raise VideoNotEmbedded(f"video {video_id!r} has empty embedding")
        return cached

    vec = store.get_video_embedding(video_id)
    if vec is None:
        raise UnknownVideo(f"video {video_id!r} not found")
    if not vec:
        raise VideoNotEmbedded(f"video {video_id!r} has empty embedding")
    cache.set_video(video_id, vec)
    return vec


# --- Public API --------------------------------------------------------------

def apply_interaction(event: UserActionEvent) -> dict:
    """Fold one interaction into the user's short-term vector (write-through).

    Returns a small result dict. A zero-weight action (e.g. a watch with
    unknown duration) is a no-op: nothing is written and `updated` is False.
    """
    vectors = store.get_user_vectors(event.user_id)
    if vectors is None:
        raise UnknownUser(f"user {event.user_id!r} not found")
    _long, mongo_short = vectors

    video_vec = _load_video_vec(event.video_id)

    weight = mc.action_weight(
        event.action,
        watch_pct=event.watch_pct,
        is_completion=event.is_completion,
    )
    if weight == 0.0:
        return {
            "user_id": event.user_id,
            "video_id": event.video_id,
            "action": event.action,
            "weight": 0.0,
            "updated": False,
        }

    short = _load_short_term(event.user_id, mongo_short)
    new_short = mc.update_short_term(short, video_vec, weight, decay=DECAY)

    # Write-through: Mongo (durable) + cache (hot reads).
    store.set_short_term(event.user_id, new_short)
    cache.set_short_term(event.user_id, new_short)

    return {
        "user_id": event.user_id,
        "video_id": event.video_id,
        "action": event.action,
        "weight": weight,
        "updated": True,
        "similarity_to_video": mc.cosine(new_short, video_vec),
    }


def get_query_vector(user_id: str) -> dict:
    """Return the blended query vector served to Recall, plus metadata.

    Cold start (no long- or short-term signal) returns an empty `embedding`
    and `cold_start: true` so Recall can branch to demographics / trending.
    """
    vectors = store.get_user_vectors(user_id)
    if vectors is None:
        raise UnknownUser(f"user {user_id!r} not found")
    long_v, mongo_short = vectors
    short_v = _load_short_term(user_id, mongo_short)

    q = mc.blend_query(long_v, short_v, beta=BETA)
    cold = mc.is_zero(q)

    return {
        "user_id": user_id,
        "embedding": [] if cold else [float(x) for x in q.tolist()],
        "dim": 0 if cold else int(q.shape[0]),
        "cold_start": cold,
        "has_long_term": not mc.is_zero(long_v),
        "has_short_term": not mc.is_zero(short_v),
    }
