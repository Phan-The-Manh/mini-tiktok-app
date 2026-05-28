"""Ad-hoc smoke check for the Redis Streams consumer (step 3.2).
Not part of the final smoke test (step 3.12) — this only exercises the
loop contract: group creation, XACK on success, idempotent skip, and
DLQ after exceeding retries."""

from __future__ import annotations

import logging
import time
import uuid

import content_analyzer._path  # noqa: F401
from client import get_mongo, get_redis  # type: ignore[import-not-found]

from content_analyzer.consumers.video_uploaded import (
    ConsumerConfig,
    VideoUploadedConsumer,
)
from content_analyzer.schemas.events import VideoUploadedEvent


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


def _make_cfg(suffix: str) -> ConsumerConfig:
    # Use a fresh stream + group + dlq per run so we don't collide with the
    # real `video.uploaded` stream the upload service uses.
    return ConsumerConfig(
        stream=f"video.uploaded.test.{suffix}",
        group=f"content_analyzer.test.{suffix}",
        consumer="smoke",
        dlq_list=f"video.uploaded.dlq.test.{suffix}",
        max_retries=2,
        block_ms=500,        # short blocks so the smoke completes quickly
        claim_idle_ms=100,   # treat anything pending >100ms as stale
        analyzer_version="vtest",
    )


def _publish(r, stream: str, video_id: str) -> str:
    ev = VideoUploadedEvent(
        video_id=video_id,
        author_id="u_smoke",
        url=f"http://localhost:9000/videos/{video_id}.mp4",
        duration_seconds=12.5,
    )
    fields: dict = {}
    for k, v in ev.model_dump(mode="python").items():
        if v is None:
            continue
        fields[k] = v.isoformat() if hasattr(v, "isoformat") else str(v)
    return r.xadd(stream, fields)


def _stream_len(r, stream: str) -> int:
    try:
        return r.xlen(stream)
    except Exception:
        return -1


def _pending_count(r, stream: str, group: str) -> int:
    try:
        info = r.xpending(stream, group)
        # info is a dict-like with 'pending' key in redis-py
        return int(info["pending"]) if isinstance(info, dict) else int(info[0])
    except Exception:
        return -1


def main() -> int:
    r = get_redis()
    db = get_mongo()
    suffix = uuid.uuid4().hex[:8]

    # === Case 1: success path ===
    cfg = _make_cfg(suffix + "-ok")
    seen: list[str] = []

    def ok_handler(ev: VideoUploadedEvent) -> None:
        seen.append(ev.video_id)

    consumer = VideoUploadedConsumer(handler=ok_handler, config=cfg)
    consumer.ensure_group()
    vid_ok = f"v_smoke_{suffix}_ok"
    msg_id = _publish(r, cfg.stream, vid_ok)
    consumer.run(once=True)
    assert seen == [vid_ok], f"handler should have run once: {seen}"
    assert _pending_count(r, cfg.stream, cfg.group) == 0, "expected no pending after ack"
    log.info("[OK] success path: handler ran, message acked, no pending")

    # === Case 2: idempotency ===
    # Seed the Mongo doc with the current analyzer_version so the consumer skips it.
    cfg2 = _make_cfg(suffix + "-idem")
    seen2: list[str] = []

    def ok_handler2(ev: VideoUploadedEvent) -> None:
        seen2.append(ev.video_id)

    vid_idem = f"v_smoke_{suffix}_idem"
    db.videos.insert_one({
        "video_id": vid_idem,
        "author_id": "u_smoke",
        "analyzer_version": cfg2.analyzer_version,
        "_smoke": True,
    })
    try:
        consumer2 = VideoUploadedConsumer(handler=ok_handler2, config=cfg2)
        consumer2.ensure_group()
        _publish(r, cfg2.stream, vid_idem)
        consumer2.run(once=True)
        assert seen2 == [], f"handler should NOT have run (idempotent skip), got: {seen2}"
        assert _pending_count(r, cfg2.stream, cfg2.group) == 0, "skipped msg should be acked"
        log.info("[OK] idempotency: handler skipped, message still acked")
    finally:
        db.videos.delete_one({"video_id": vid_idem, "_smoke": True})

    # === Case 3: DLQ after exceeding retries ===
    cfg3 = _make_cfg(suffix + "-dlq")
    attempts = {"n": 0}

    def bad_handler(ev: VideoUploadedEvent) -> None:
        attempts["n"] += 1
        raise RuntimeError(f"forced failure #{attempts['n']}")

    consumer3 = VideoUploadedConsumer(handler=bad_handler, config=cfg3)
    consumer3.ensure_group()
    vid_bad = f"v_smoke_{suffix}_bad"
    _publish(r, cfg3.stream, vid_bad)

    # First pass: read new, handler raises, message stays pending (delivery=1).
    consumer3.run(once=True)
    # Subsequent passes: wait > claim_idle_ms, then reclaim and retry.
    # max_retries=2 means we DLQ once times_delivered > 2 (i.e., on the 3rd pass).
    for i in range(5):
        time.sleep(cfg3.claim_idle_ms / 1000 + 0.05)
        consumer3.run(once=True)
        if r.llen(cfg3.dlq_list) > 0:
            break

    dlq_len = r.llen(cfg3.dlq_list)
    assert dlq_len == 1, f"expected 1 DLQ entry, got {dlq_len}"
    assert _pending_count(r, cfg3.stream, cfg3.group) == 0, "DLQ'd msg should be acked"
    raw = r.lrange(cfg3.dlq_list, 0, -1)[0]
    log.info("[OK] DLQ path: handler raised %d times, then message DLQ'd", attempts["n"])
    log.info("     DLQ entry: %s", raw[:140])

    # === Cleanup ===
    for c in (cfg, cfg2, cfg3):
        try:
            r.xgroup_destroy(c.stream, c.group)
        except Exception:
            pass
        r.delete(c.stream)
        r.delete(c.dlq_list)

    log.info("[OK] all 3 cases passed; cleaned up test streams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
