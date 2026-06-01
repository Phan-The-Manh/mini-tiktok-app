"""Ad-hoc smoke check for MongoDB write-back (step 3.9).

Hits a live MongoDB Atlas cluster via `database/client.py`. Skips
gracefully if the cluster cannot be reached so the test does not
poison local-only runs.

The test inserts throw-away `videos` docs under the prefix
`_smoke_writer_*` and removes them on exit (success or failure). It
exercises:

  1. write_back on a `pending` doc -> all five fields set; status
     flips to `approved`.
  2. write_back on an `approved` doc -> fields refreshed; status
     stays `approved` (idempotent re-embed).
  3. write_back on a `rejected` doc -> fields refreshed; status
     stays `rejected` (never resurrects a rejected video).
  4. write_back on a missing video_id -> `WriteBackError`.
  5. Embedding accepted as both `np.ndarray` and `list[float]`; the
     stored value is a list of plain floats with the correct length.
  6. Wrong-shape ndarray raises `WriteBackError` before any write.

Run: `python -m content_analyzer.tests._smoke_writer`
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

import content_analyzer._path  # noqa: F401
from client import get_mongo  # type: ignore[import-not-found]
from pymongo.errors import ServerSelectionTimeoutError  # type: ignore[import-not-found]

from content_analyzer.services.writer import (
    WriteBackError,
    write_back,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


DIM = 384
ANALYZER_VERSION = "smoke-v1"


def _insert_stub(db, video_id: str, moderation_status: str) -> None:
    db.videos.insert_one(
        {
            "video_id": video_id,
            "author_id": "_smoke_writer_author",
            "url": "http://localhost:9000/videos/_smoke_writer.mp4",
            "duration_seconds": 1.0,
            "caption": "",
            "hashtags": [],
            "ai_tags": {},
            "content_embedding": [],
            "stats": {},
            "distribution_stage": "test_pool_1",
            "moderation_status": moderation_status,
            "uploaded_at": datetime.now(timezone.utc),
        }
    )


def main() -> int:
    db = get_mongo()
    try:
        db.command("ping")
    except ServerSelectionTimeoutError as e:
        log.warning("[WARN] MongoDB unreachable, skipping smoke: %s", e)
        return 0

    embedding = np.random.default_rng(0).standard_normal(DIM).astype(np.float32)

    ids = [
        "_smoke_writer_pending",
        "_smoke_writer_approved",
        "_smoke_writer_rejected",
        "_smoke_writer_listinput",
    ]
    # Defensive: remove any leftovers from a previous failed run.
    db.videos.delete_many({"video_id": {"$in": ids}})

    try:
        # === Case 1: pending -> approved ===
        _insert_stub(db, "_smoke_writer_pending", "pending")
        write_back(
            "_smoke_writer_pending",
            embedding,
            "hello world",
            analyzer_version=ANALYZER_VERSION,
        )
        doc = db.videos.find_one({"video_id": "_smoke_writer_pending"})
        assert doc is not None, "doc disappeared after write_back"
        assert doc["moderation_status"] == "approved", (
            f"expected approved, got {doc['moderation_status']}"
        )
        assert doc["analyzer_version"] == ANALYZER_VERSION
        assert isinstance(doc["analyzed_at"], datetime)
        assert doc["ai_tags"]["transcript"] == "hello world"
        assert len(doc["content_embedding"]) == DIM
        assert all(isinstance(x, float) for x in doc["content_embedding"][:8])
        log.info("[OK] pending -> approved with all fields populated")

        # === Case 2: approved stays approved on re-embed ===
        write_back(
            "_smoke_writer_pending",
            embedding,
            "second pass",
            analyzer_version="smoke-v2",
        )
        doc = db.videos.find_one({"video_id": "_smoke_writer_pending"})
        assert doc["moderation_status"] == "approved"
        assert doc["analyzer_version"] == "smoke-v2"
        assert doc["ai_tags"]["transcript"] == "second pass"
        log.info("[OK] re-embed on approved doc preserves status, refreshes fields")

        # === Case 3: rejected stays rejected ===
        _insert_stub(db, "_smoke_writer_rejected", "rejected")
        write_back(
            "_smoke_writer_rejected",
            embedding,
            "should not change status",
            analyzer_version=ANALYZER_VERSION,
        )
        doc = db.videos.find_one({"video_id": "_smoke_writer_rejected"})
        assert doc["moderation_status"] == "rejected", (
            f"rejected video must not be resurrected; got {doc['moderation_status']}"
        )
        assert len(doc["content_embedding"]) == DIM
        log.info("[OK] rejected doc keeps rejected status; embedding still refreshed")

        # === Case 4: missing video_id -> WriteBackError ===
        try:
            write_back(
                "_smoke_writer_does_not_exist",
                embedding,
                "",
                analyzer_version=ANALYZER_VERSION,
            )
        except WriteBackError as e:
            log.info("[OK] missing video raised WriteBackError: %s", e)
        else:
            log.error("[FAIL] expected WriteBackError on missing video_id")
            return 1

        # === Case 5: list input is accepted, stored as floats ===
        _insert_stub(db, "_smoke_writer_listinput", "pending")
        write_back(
            "_smoke_writer_listinput",
            [float(x) for x in embedding.tolist()],
            "",
            analyzer_version=ANALYZER_VERSION,
        )
        doc = db.videos.find_one({"video_id": "_smoke_writer_listinput"})
        assert len(doc["content_embedding"]) == DIM
        assert doc["ai_tags"]["transcript"] == ""
        log.info("[OK] list[float] embedding accepted and stored")

        # === Case 6: wrong-shape ndarray rejected without writing ===
        _insert_stub(db, "_smoke_writer_approved", "approved")
        original = db.videos.find_one({"video_id": "_smoke_writer_approved"})
        bad = np.zeros((2, DIM), dtype=np.float32)
        try:
            write_back(
                "_smoke_writer_approved",
                bad,
                "should not be written",
                analyzer_version=ANALYZER_VERSION,
            )
        except WriteBackError as e:
            log.info("[OK] 2-D embedding raised WriteBackError: %s", e)
        else:
            log.error("[FAIL] expected WriteBackError on 2-D embedding")
            return 1
        after = db.videos.find_one({"video_id": "_smoke_writer_approved"})
        assert after.get("analyzer_version") is None, (
            "bad-shape write_back must not have touched the document"
        )
        assert after["content_embedding"] == original["content_embedding"]
        log.info("[OK] failed write_back left the document untouched")

        log.info("[OK] all writer cases passed")
        return 0
    finally:
        db.videos.delete_many({"video_id": {"$in": ids}})


if __name__ == "__main__":
    raise SystemExit(main())
