"""
cache.py — Redis vector cache (step 4.4)
========================================

Hot-path cache so reads/updates of a user's vectors don't hit Mongo every
time. We run **write-through**: every short-term update writes Redis AND Mongo
synchronously (see services/update.py), so the cache is never the sole copy.

Serialization
-------------
The shared Redis client (`database/client.py`) is created with
`decode_responses=True`, so values come back as `str`. We store vectors as a
JSON array of floats — compact enough for 384-d, human-inspectable, and it
avoids a second Redis client just for binary embeddings.

Keys
----
    ue:short:{user_id}   -> JSON float list   (short-term vector)
    ue:long:{user_id}    -> JSON float list   (long-term vector, read cache)
    ue:vid:{video_id}    -> JSON float list   (video content_embedding)
    ue:seen:{interaction_id} -> "1"           (idempotency marker)
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np

import user_embedding._path  # noqa: F401  side-effect: sys.path + env
from client import get_redis  # type: ignore[import-not-found]

_PREFIX = "ue"
_DEFAULT_TTL = int(os.getenv("USER_EMBEDDING_CACHE_TTL", "86400"))
# Idempotency markers live longer than a redelivery could plausibly take.
_SEEN_TTL = int(os.getenv("USER_EMBEDDING_SEEN_TTL", str(7 * 86400)))


def _key(kind: str, ident: str) -> str:
    return f"{_PREFIX}:{kind}:{ident}"


def _dumps(vec: Sequence[float] | np.ndarray) -> str:
    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    return json.dumps([float(x) for x in vec])


def _loads(raw: Optional[str]) -> Optional[list[float]]:
    if not raw:
        return None
    try:
        return [float(x) for x in json.loads(raw)]
    except (ValueError, TypeError):
        return None


# --- Short-term ---------------------------------------------------------------

def get_short_term(user_id: str) -> Optional[list[float]]:
    return _loads(get_redis().get(_key("short", user_id)))


def set_short_term(
    user_id: str,
    vec: Sequence[float] | np.ndarray,
    *,
    ttl: int = _DEFAULT_TTL,
) -> None:
    get_redis().set(_key("short", user_id), _dumps(vec), ex=ttl)


# --- Long-term (read cache; source of truth is Mongo via the batch) ----------

def get_long_term(user_id: str) -> Optional[list[float]]:
    return _loads(get_redis().get(_key("long", user_id)))


def set_long_term(
    user_id: str,
    vec: Sequence[float] | np.ndarray,
    *,
    ttl: int = _DEFAULT_TTL,
) -> None:
    get_redis().set(_key("long", user_id), _dumps(vec), ex=ttl)


# --- Video embeddings (stable; cache to avoid re-reading on hot users) -------

def get_video(video_id: str) -> Optional[list[float]]:
    return _loads(get_redis().get(_key("vid", video_id)))


def set_video(
    video_id: str,
    vec: Sequence[float] | np.ndarray,
    *,
    ttl: int = _DEFAULT_TTL,
) -> None:
    get_redis().set(_key("vid", video_id), _dumps(vec), ex=ttl)


# --- Idempotency markers ------------------------------------------------------

def is_processed(interaction_id: str) -> bool:
    return bool(get_redis().exists(_key("seen", interaction_id)))


def mark_processed(interaction_id: str, *, ttl: int = _SEEN_TTL) -> None:
    get_redis().set(_key("seen", interaction_id), "1", ex=ttl)


# --- Test/cleanup helpers -----------------------------------------------------

def forget_user(user_id: str) -> None:
    """Drop a user's cached vectors. Used by the smoke test cleanup."""
    r = get_redis()
    r.delete(_key("short", user_id), _key("long", user_id))


def forget_video(video_id: str) -> None:
    get_redis().delete(_key("vid", video_id))


def forget_interaction(interaction_id: str) -> None:
    get_redis().delete(_key("seen", interaction_id))
