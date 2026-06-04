"""smoke_test.py - Content Analyzer end-to-end smoke test (step 3.12)
====================================================================

Proves the whole Content Analyzer pipeline works wired up: an event on
the Redis Stream turns into a 384-d unit vector on the Mongo doc and
the same vector retrieves the video from Atlas Vector Search.

Prereqs:
    1. docker compose up -d              Redis + MinIO healthy
    2. database/.env points at Atlas     vector index ACTIVE
    3. ffmpeg on PATH                    needed to synthesize the fixture
    4. database seeded (optional)        we fall back to a throwaway author

Run from project root, venv active:
    python -m content_analyzer.tests.smoke_test

Total wall time on a CPU laptop: ~30-60 s (model load 5-15 s + per-video
embedding 10-40 s + a few seconds of vector-index lag).

What it does:

    1. Preflight Mongo / Redis / MinIO / ffmpeg.
    2. Synthesize a 2-second silent-blue mp4 with ffmpeg.
    3. Upload the mp4 to MinIO and insert a `videos` doc
       (status=pending, embedding=[]) under a unique smoke prefix.
    4. Override CONTENT_ANALYZER_* env so the consumer uses a
       throwaway stream/group/dlq and a bumped ANALYZER_VERSION.
       The real `video.uploaded` stream is untouched.
    5. Pre-create the consumer group (so the event we publish lands in
       the group's PEL instead of falling off the back of `$`).
    6. Publish a synthetic `video.uploaded` event for the fixture.
    7. Run `content_analyzer.main.main(["--once"])` - real startup
       path: loads CLIP + Whisper-tiny + MiniLM, joins the (already
       existing) group, processes the one pending message, acks.
    8. Assert on the Mongo doc:
         - content_embedding length == 384
         - it is a unit vector (norm ~= 1)
         - analyzer_version matches the smoke override
         - analyzed_at is a datetime
         - ai_tags.transcript is a string (possibly empty - the
           silence gate should kick in on an anullsrc audio track)
         - moderation_status flipped pending -> approved
    9. Atlas Vector Search round-trip: query with the same embedding,
       expect the smoke video as the top hit. Polled to absorb the
       1-5 s index-update lag.
   10. Cleanup, best-effort: Mongo doc + author (if we created it),
       MinIO object, Redis stream + group + DLQ, temp dir.

The smoke test bypasses the Upload Service deliberately - the
Component #3 contract is "consume an event, write an embedding", and
that is what this verifies. Component #2 has its own smoke test.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env

import numpy as np
from pymongo.errors import ServerSelectionTimeoutError  # type: ignore[import-not-found]

from client import get_minio, get_mongo, get_redis  # type: ignore[import-not-found]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("smoke")


# --- Per-run identifiers -----------------------------------------------------
# A fresh suffix per invocation keeps concurrent / leftover runs isolated.
SUFFIX = uuid.uuid4().hex[:8]
TEST_VIDEO_ID = f"_smoke_e2e_{SUFFIX}"
TEST_OBJECT_KEY = f"{TEST_VIDEO_ID}.mp4"
TEST_STREAM = f"video.uploaded.smoke.{SUFFIX}"
TEST_GROUP = f"content_analyzer.smoke.{SUFFIX}"
TEST_CONSUMER = f"smoke-{SUFFIX}"
TEST_DLQ = f"video.uploaded.dlq.smoke.{SUFFIX}"
TEST_ANALYZER_VERSION = "smoke-v1"

# Atlas index expects 384-d unit vectors; this is the contract we assert on.
INDEX_DIM = 384

# Atlas Search indexes are eventually consistent. The video usually appears
# within 1-5 s of the write; we poll longer to absorb the long tail.
VECTOR_POLL_TIMEOUT_S = 30
VECTOR_POLL_INTERVAL_S = 2


# --- Fixture helpers ---------------------------------------------------------

def _ffmpeg_bin() -> str:
    return os.getenv("FFMPEG_BIN", "ffmpeg")


def _make_silent_mp4(dst: Path) -> bool:
    """Generate a 2-second silent blue mp4 via ffmpeg. Returns True on success.

    We add a true-silence audio track (anullsrc, peak ~= -91 dBFS) rather
    than dropping audio entirely. That exercises the silence gate in
    services/audio.py - the gate should suppress Whisper and yield an
    empty transcript, but we still want to prove the gate triggers
    rather than relying on the no-audio-stream short-circuit.
    """
    cmd = [
        _ffmpeg_bin(), "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000:d=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        "-shortest",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _pick_or_create_author(db) -> tuple[str, bool]:
    """Return (author_id, created_here). Prefer a seeded user; create a
    throwaway one if the DB has not been seeded yet."""
    doc = db.users.find_one({"user_id": {"$regex": "^seed_"}}, {"user_id": 1})
    if doc:
        return doc["user_id"], False
    uid = f"_smoke_e2e_author_{SUFFIX}"
    db.users.insert_one({
        "user_id": uid,
        "username": uid,
        "_smoke": True,
    })
    return uid, True


def _ensure_group(r, stream: str, group: str) -> None:
    """Create the group from the head of the stream so any later XADD lands
    in the group's PEL. The consumer's own ensure_group uses id='$' which
    would skip past anything we publish before it joins."""
    try:
        r.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise


def _publish_event(r, stream: str, video_id: str, author_id: str, url: str) -> str:
    fields = {
        "video_id": video_id,
        "author_id": author_id,
        "url": url,
        "duration_seconds": "2.0",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    return r.xadd(stream, fields)


def _vector_search_top(db, query_vec: list[float]) -> tuple[str, float] | None:
    """Run a $vectorSearch and return (video_id, score) of the top hit."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": "video_content_index",
                "path": "content_embedding",
                "queryVector": query_vec,
                "numCandidates": 50,
                "limit": 5,
            }
        },
        {
            "$project": {
                "_id": 0,
                "video_id": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    try:
        results = list(db.videos.aggregate(pipeline))
    except Exception as e:
        log.warning("[WARN] vector search query failed: %s", e)
        return None
    if not results:
        return None
    return results[0]["video_id"], float(results[0]["score"])


# --- Main --------------------------------------------------------------------

def main() -> int:
    log.info("=" * 60)
    log.info("Content Analyzer end-to-end smoke (suffix=%s)", SUFFIX)
    log.info("=" * 60)

    # --- Preflight ---
    db = get_mongo()
    try:
        db.command("ping")
    except ServerSelectionTimeoutError as e:
        log.error("[FAIL] MongoDB unreachable: %s", e)
        return 1

    r = get_redis()
    try:
        r.ping()
    except Exception as e:
        log.error("[FAIL] Redis unreachable: %s", e)
        return 1

    minio = get_minio()
    try:
        minio.list_buckets()
    except Exception as e:
        log.error("[FAIL] MinIO unreachable: %s", e)
        return 1

    if shutil.which(_ffmpeg_bin()) is None:
        log.error("[FAIL] ffmpeg not on PATH (set FFMPEG_BIN if installed elsewhere)")
        return 1

    log.info("[OK] preflight: mongo + redis + minio + ffmpeg reachable")

    # --- Test isolation: override env BEFORE the consumer reads ConsumerConfig ---
    # The consumer reads these once via ConsumerConfig.from_env() during main(),
    # so they must be set before we import/run the worker.
    os.environ["CONTENT_ANALYZER_STREAM"] = TEST_STREAM
    os.environ["CONTENT_ANALYZER_GROUP"] = TEST_GROUP
    os.environ["CONTENT_ANALYZER_CONSUMER"] = TEST_CONSUMER
    os.environ["CONTENT_ANALYZER_DLQ"] = TEST_DLQ
    # Short block keeps the test fast; the message is already pending when
    # XREADGROUP fires, so this only bounds the no-message case.
    os.environ["CONTENT_ANALYZER_BLOCK_MS"] = "2000"
    os.environ["ANALYZER_VERSION"] = TEST_ANALYZER_VERSION

    bucket = os.getenv("MINIO_BUCKET", "videos")
    public_prefix = os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000").rstrip("/")
    url = f"{public_prefix}/{bucket}/{TEST_OBJECT_KEY}"

    work_dir = Path(tempfile.mkdtemp(prefix=f"smoke_e2e_{SUFFIX}_"))
    cleanup_objects: list[tuple[str, str]] = []  # (bucket, key)
    author_to_cleanup: str | None = None

    try:
        # --- Stage: MinIO bucket + fixture object ---
        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)

        sample = work_dir / "sample.mp4"
        if not _make_silent_mp4(sample):
            log.error("[FAIL] ffmpeg refused to generate the fixture mp4")
            return 1
        minio.fput_object(
            bucket_name=bucket,
            object_name=TEST_OBJECT_KEY,
            file_path=str(sample),
            content_type="video/mp4",
        )
        cleanup_objects.append((bucket, TEST_OBJECT_KEY))
        log.info("[OK] uploaded fixture to %s/%s (%d bytes)",
                 bucket, TEST_OBJECT_KEY, sample.stat().st_size)

        # --- Stage: Mongo videos doc ---
        author_id, created_here = _pick_or_create_author(db)
        if created_here:
            author_to_cleanup = author_id

        db.videos.insert_one({
            "video_id": TEST_VIDEO_ID,
            "author_id": author_id,
            "url": url,
            "thumbnail_url": None,
            "duration_seconds": 2.0,
            # A non-empty caption so the text encoder has signal even
            # when the silence gate suppresses the transcript.
            "caption": "smoke test fixture caption about cooking",
            "hashtags": ["smoke", "cooking"],
            "category": "cooking",
            "ai_tags": {},
            "content_embedding": [],
            "stats": {},
            "distribution_stage": "test_pool_1",
            "moderation_status": "pending",
            "uploaded_at": datetime.now(timezone.utc),
            "_smoke": True,
        })
        log.info("[OK] inserted videos doc %s (status=pending, embedding=[])",
                 TEST_VIDEO_ID)

        # --- Pre-create group, then publish ---
        # Order matters: if we publish before the group exists, XREADGROUP
        # with `>` will never see the message (the consumer's own
        # ensure_group uses id='$', skipping past pre-existing entries).
        _ensure_group(r, TEST_STREAM, TEST_GROUP)
        msg_id = _publish_event(r, TEST_STREAM, TEST_VIDEO_ID, author_id, url)
        log.info("[OK] published %s on %s (id=%s)", TEST_VIDEO_ID, TEST_STREAM, msg_id)

        # --- Run the worker once ---
        # Import here so logging.basicConfig in worker_main is a no-op
        # (basicConfig only configures the root logger if no handlers are
        # attached yet, and ours already are).
        from content_analyzer.main import main as worker_main
        log.info("[INFO] starting worker - loads ~300 MB of weights (5-15 s)")
        t0 = time.time()
        rc = worker_main(["--once"])
        log.info("[INFO] worker --once returned rc=%d in %.1f s", rc, time.time() - t0)
        if rc != 0:
            log.error("[FAIL] worker exited non-zero")
            return 1

        # --- Assertions on the Mongo doc ---
        doc = db.videos.find_one({"video_id": TEST_VIDEO_ID})
        if doc is None:
            log.error("[FAIL] videos doc vanished during the test")
            return 1

        emb = doc.get("content_embedding") or []
        if len(emb) != INDEX_DIM:
            log.error("[FAIL] embedding dim mismatch: got %d expected %d",
                      len(emb), INDEX_DIM)
            return 1

        norm = float(np.linalg.norm(np.asarray(emb, dtype=np.float64)))
        if not (0.98 <= norm <= 1.02):
            log.error("[FAIL] embedding is not unit-norm: |v|=%.4f", norm)
            return 1

        if doc.get("analyzer_version") != TEST_ANALYZER_VERSION:
            log.error("[FAIL] analyzer_version=%r expected %r",
                      doc.get("analyzer_version"), TEST_ANALYZER_VERSION)
            return 1

        if not isinstance(doc.get("analyzed_at"), datetime):
            log.error("[FAIL] analyzed_at missing or not a datetime")
            return 1

        transcript = doc.get("ai_tags", {}).get("transcript")
        if not isinstance(transcript, str):
            log.error("[FAIL] ai_tags.transcript missing or not a string (got %r)",
                      transcript)
            return 1

        if doc.get("moderation_status") != "approved":
            log.error("[FAIL] moderation_status=%r expected 'approved'",
                      doc.get("moderation_status"))
            return 1

        log.info("[OK] doc fields: dim=%d |v|=%.4f version=%s "
                 "transcript_chars=%d status=%s",
                 len(emb), norm, doc["analyzer_version"],
                 len(transcript), doc["moderation_status"])

        # --- Vector search round-trip ---
        log.info("[INFO] polling Atlas Vector Search (timeout=%ds)",
                 VECTOR_POLL_TIMEOUT_S)
        started = time.time()
        hit: tuple[str, float] | None = None
        while time.time() - started < VECTOR_POLL_TIMEOUT_S:
            hit = _vector_search_top(db, emb)
            if hit and hit[0] == TEST_VIDEO_ID:
                break
            time.sleep(VECTOR_POLL_INTERVAL_S)

        if hit is None or hit[0] != TEST_VIDEO_ID:
            log.error(
                "[FAIL] vector search did not return %s as top hit (got %r). "
                "Possible causes: index not ACTIVE, index-update lag exceeded "
                "%ds, or fixture filtered out (check moderation_status / "
                "distribution_stage filters on the index).",
                TEST_VIDEO_ID, hit, VECTOR_POLL_TIMEOUT_S,
            )
            return 1
        log.info("[OK] vector search top hit: %s (score=%.4f)", hit[0], hit[1])

        log.info("=" * 60)
        log.info("[OK] end-to-end smoke test passed")
        log.info("=" * 60)
        return 0

    finally:
        # Best-effort cleanup. We swallow errors here so a partial run
        # doesn't mask the real failure above.
        try:
            db.videos.delete_one({"video_id": TEST_VIDEO_ID})
        except Exception as e:
            log.warning("[WARN] mongo videos cleanup failed: %s", e)
        if author_to_cleanup:
            try:
                db.users.delete_one({"user_id": author_to_cleanup, "_smoke": True})
            except Exception as e:
                log.warning("[WARN] mongo users cleanup failed: %s", e)
        for b, k in cleanup_objects:
            try:
                minio.remove_object(b, k)
            except Exception as e:
                log.warning("[WARN] minio cleanup %s/%s failed: %s", b, k, e)
        try:
            r.xgroup_destroy(TEST_STREAM, TEST_GROUP)
        except Exception:
            pass
        for key in (TEST_STREAM, TEST_DLQ):
            try:
                r.delete(key)
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
