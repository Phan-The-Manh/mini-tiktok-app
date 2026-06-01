"""
writer.py — MongoDB write-back for analyzed videos
==================================================

Step 3.9 of the Content Analyzer build.

Persists the output of the embedding pipeline (steps 3.4-3.8) back to the
`videos` collection so the video becomes searchable by Atlas Vector
Search:

    content_embedding    (384-d unit vector from fuse.fuse())
    ai_tags.transcript   (Whisper text from audio.transcribe_video();
                          may be the empty string for silent / audio-less
                          videos)
    analyzer_version     (idempotency key consulted by
                          consumers/video_uploaded.py)
    analyzed_at          (audit timestamp, UTC)
    moderation_status    (pending -> approved; other states preserved)

Atomicity
---------
The five fields land in a single aggregation-pipeline update so they
share one round-trip to Atlas and the conditional moderation flip
cannot race with another writer. `$cond` keeps `moderation_status`
untouched when it is already `approved` or `rejected`, so a later
re-embedding (bumped ANALYZER_VERSION) never resurrects a rejected
video.

Moderation stub
---------------
Real moderation lands in a later component. For now the writer flips
`pending` straight to `approved` — the no-op "auto-approve on analyze"
behaviour called out in TODO.md step 3.9. (TODO.md uses the word
"ready" as shorthand; the schema's `ModerationStatus` enum has no such
state, so we map onto `approved`, which is what Recall will filter
on.)

Embedding serialization
-----------------------
`fuse.fuse()` returns a `numpy.ndarray`; BSON wants plain Python
floats. We convert to `list[float]` once here so callers do not have
to remember.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import numpy as np
from pymongo.database import Database

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env
from client import get_mongo  # type: ignore[import-not-found]


log = logging.getLogger(__name__)


class WriteBackError(RuntimeError):
    """Raised when the write-back cannot complete (e.g., no matching video)."""


# --- Tunables (env-backed) ---------------------------------------------------

def _analyzer_version(explicit: str | None = None) -> str:
    return explicit or os.getenv("ANALYZER_VERSION", "v1")


# --- Helpers -----------------------------------------------------------------

def _to_float_list(embedding: "np.ndarray | list[float]") -> list[float]:
    """Coerce a 1-D embedding to a list of Python floats Mongo can store."""
    if isinstance(embedding, np.ndarray):
        if embedding.ndim != 1:
            raise WriteBackError(
                f"expected 1-D embedding, got shape {embedding.shape}"
            )
        return embedding.astype(float).tolist()
    return [float(x) for x in embedding]


# --- Public API --------------------------------------------------------------

def write_back(
    video_id: str,
    content_embedding: "np.ndarray | list[float]",
    transcript: str,
    *,
    analyzer_version: str | None = None,
    db: Database | None = None,
) -> None:
    """Persist the analyzer's output for `video_id`.

    Always sets `content_embedding`, `ai_tags.transcript`,
    `analyzer_version`, and `analyzed_at`. Conditionally flips
    `moderation_status` from `pending` to `approved` in the same
    atomic update; `approved` and `rejected` are preserved.

    Raises `WriteBackError` if no `videos` document has `video_id`.
    """
    db = db or get_mongo()
    version = _analyzer_version(analyzer_version)
    embedding_list = _to_float_list(content_embedding)
    now = datetime.now(timezone.utc)

    result = db.videos.update_one(
        {"video_id": video_id},
        [
            {
                "$set": {
                    "content_embedding": embedding_list,
                    "ai_tags.transcript": transcript,
                    "analyzer_version": version,
                    "analyzed_at": now,
                    "moderation_status": {
                        "$cond": [
                            {"$eq": ["$moderation_status", "pending"]},
                            "approved",
                            "$moderation_status",
                        ]
                    },
                }
            }
        ],
    )

    if result.matched_count == 0:
        raise WriteBackError(f"no videos document with video_id={video_id!r}")

    log.info(
        "[OK] wrote analyzer output to video %s "
        "(dim=%d, transcript_chars=%d, version=%s)",
        video_id, len(embedding_list), len(transcript), version,
    )
