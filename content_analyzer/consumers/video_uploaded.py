"""
video_uploaded.py — Redis Streams consumer for `video.uploaded`
===============================================================

Consumes the stream emitted by the Upload Service (Component #2) and hands
each event to an injected `handler`. The handler is a callable that does the
actual embedding work — wired up in main.py once steps 3.3-3.9 are in.

Design choices:

- **Named consumer group** (`content_analyzer`). Scaling horizontally is as
  simple as starting more workers with the same group but distinct consumer
  names; Redis load-balances entries between them.
- **Idempotency.** Before doing work we look up the Mongo `videos` doc and
  skip if it already carries the current `ANALYZER_VERSION`. This makes
  re-deliveries safe — e.g., a worker that crashes between writing the
  embedding to Mongo and XACK'ing the message will see the second delivery
  as a no-op on restart.
- **Retry + DLQ.** Messages that the handler raises on are deliberately NOT
  ack'd, so Redis keeps them in the group's PEL (pending entries list).
  Each loop iteration calls `XAUTOCLAIM` to pick up pending entries idle
  for more than `claim_idle_ms`. For each, we look up `times_delivered`
  via `XPENDING`: above `max_retries` it goes to the `video.uploaded.dlq`
  list (`LPUSH` of a JSON record) and is ack'd; otherwise the handler runs
  again. Unparseable payloads go straight to the DLQ on first sight — they
  can never succeed and shouldn't burn retries.

The consumer is a class with an injectable handler so it can be unit-tested
with synthetic events, and so steps 3.3-3.9 can build the real handler in
isolation without touching the loop.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import redis  # type: ignore[import-untyped]
from pymongo.database import Database

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env
from client import get_mongo, get_redis  # type: ignore[import-not-found]

from content_analyzer.schemas.events import VideoUploadedEvent


log = logging.getLogger(__name__)


Handler = Callable[[VideoUploadedEvent], None]


@dataclass(frozen=True)
class ConsumerConfig:
    stream: str
    group: str
    consumer: str
    dlq_list: str
    max_retries: int
    block_ms: int
    claim_idle_ms: int
    analyzer_version: str
    batch_size: int = 10

    @classmethod
    def from_env(cls) -> "ConsumerConfig":
        return cls(
            stream=os.getenv("CONTENT_ANALYZER_STREAM", "video.uploaded"),
            group=os.getenv("CONTENT_ANALYZER_GROUP", "content_analyzer"),
            consumer=os.getenv("CONTENT_ANALYZER_CONSUMER", "worker-1"),
            dlq_list=os.getenv("CONTENT_ANALYZER_DLQ", "video.uploaded.dlq"),
            max_retries=int(os.getenv("CONTENT_ANALYZER_MAX_RETRIES", "3")),
            block_ms=int(os.getenv("CONTENT_ANALYZER_BLOCK_MS", "5000")),
            claim_idle_ms=int(os.getenv("CONTENT_ANALYZER_CLAIM_IDLE_MS", "60000")),
            analyzer_version=os.getenv("ANALYZER_VERSION", "v1"),
        )


class VideoUploadedConsumer:
    def __init__(
        self,
        handler: Handler,
        config: ConsumerConfig | None = None,
        redis_client: "redis.Redis | None" = None,
        mongo_db: Database | None = None,
    ):
        self.handler = handler
        self.cfg = config or ConsumerConfig.from_env()
        # `is not None` rather than `or`: pymongo Database raises
        # NotImplementedError on bool() to prevent the common
        # `if db:` mistake, so `mongo_db or get_mongo()` would crash
        # when an explicit db is passed in (which main.py does).
        self.r = redis_client if redis_client is not None else get_redis()
        self.db = mongo_db if mongo_db is not None else get_mongo()
        self._stopping = False

    # --- Lifecycle ---

    def ensure_group(self) -> None:
        """Create the consumer group if it doesn't exist. MKSTREAM also creates
        the underlying stream so the worker can start before any publisher
        has emitted an event."""
        try:
            self.r.xgroup_create(
                name=self.cfg.stream,
                groupname=self.cfg.group,
                id="$",
                mkstream=True,
            )
            log.info("[OK] created consumer group %s on %s",
                     self.cfg.group, self.cfg.stream)
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return  # already exists — normal
            raise

    def stop(self) -> None:
        """Signal the run loop to exit at the next iteration boundary."""
        self._stopping = True

    def run(self, once: bool = False) -> None:
        """Main loop. `once=True` performs a single pass (reclaim + read) and
        returns; useful for the smoke test and the `--once` flag the main
        entry point will expose in step 3.10."""
        self.ensure_group()
        while not self._stopping:
            self._reclaim_stale()
            self._read_new()
            if once:
                return

    # --- Stale-message reclaim + DLQ routing ---

    def _reclaim_stale(self) -> None:
        """XAUTOCLAIM any messages idle longer than `claim_idle_ms` from any
        consumer in the group. Route each to DLQ (over the retry budget) or
        retry-process (under)."""
        next_start = "0-0"
        while True:
            try:
                next_start, claimed, _deleted = self.r.xautoclaim(
                    name=self.cfg.stream,
                    groupname=self.cfg.group,
                    consumername=self.cfg.consumer,
                    min_idle_time=self.cfg.claim_idle_ms,
                    start_id=next_start,
                    count=self.cfg.batch_size,
                )
            except redis.ResponseError as e:
                # XAUTOCLAIM requires Redis 6.2+. New messages still flow; log once.
                log.warning("[WARN] xautoclaim unavailable: %s", e)
                return

            if not claimed:
                break

            for msg_id, fields in claimed:
                self._route_claimed(_to_str(msg_id), _decode_fields(fields))

            if _to_str(next_start) == "0-0":
                break

    def _route_claimed(self, msg_id: str, fields: dict) -> None:
        delivered = self._delivery_count(msg_id)
        if delivered > self.cfg.max_retries:
            self._send_to_dlq(msg_id, fields, delivered, reason="exceeded max retries")
            self.r.xack(self.cfg.stream, self.cfg.group, msg_id)
            log.warning("[WARN] DLQ %s after %d deliveries", msg_id, delivered)
            return
        log.info("[INFO] retrying %s (delivery %d/%d)",
                 msg_id, delivered, self.cfg.max_retries)
        self._handle_one(msg_id, fields)

    def _delivery_count(self, msg_id: str) -> int:
        info = self.r.xpending_range(
            name=self.cfg.stream,
            groupname=self.cfg.group,
            min=msg_id,
            max=msg_id,
            count=1,
        )
        if not info:
            return 0
        return int(info[0].get("times_delivered", 0))

    def _send_to_dlq(self, msg_id: str, fields: dict, delivered: int, reason: str) -> None:
        payload = {
            "message_id": msg_id,
            "fields": fields,
            "times_delivered": delivered,
            "reason": reason,
            "dlq_at": datetime.now(timezone.utc).isoformat(),
        }
        self.r.lpush(self.cfg.dlq_list, json.dumps(payload))

    # --- New-message read + dispatch ---

    def _read_new(self) -> None:
        try:
            resp = self.r.xreadgroup(
                groupname=self.cfg.group,
                consumername=self.cfg.consumer,
                streams={self.cfg.stream: ">"},
                count=self.cfg.batch_size,
                block=self.cfg.block_ms,
            )
        except redis.ResponseError as e:
            log.error("[FAIL] xreadgroup error: %s", e)
            return

        for _stream_name, messages in (resp or []):
            for msg_id, fields in messages:
                self._handle_one(_to_str(msg_id), _decode_fields(fields))

    def _handle_one(self, msg_id: str, fields: dict) -> None:
        """Parse, idempotency-check, then invoke the handler. Ack only on
        successful completion (or on idempotent skip / unparseable payload).
        Exceptions leave the entry pending for the next reclaim pass."""
        try:
            event = VideoUploadedEvent.from_stream_fields(fields)
        except Exception as e:
            # An unparseable payload can never succeed. DLQ + ack immediately
            # so it doesn't poison the retry budget.
            log.error("[FAIL] unparseable event %s: %s", msg_id, e)
            self._send_to_dlq(msg_id, fields, 1, reason=f"unparseable: {e}")
            self.r.xack(self.cfg.stream, self.cfg.group, msg_id)
            return

        if self._already_processed(event.video_id):
            log.info("[OK] skip %s (video %s already at analyzer_version=%s)",
                     msg_id, event.video_id, self.cfg.analyzer_version)
            self.r.xack(self.cfg.stream, self.cfg.group, msg_id)
            return

        try:
            self.handler(event)
        except Exception as e:
            # Don't ack — message remains pending and will be reclaimed.
            log.exception("[FAIL] handler raised on %s (video %s): %s",
                          msg_id, event.video_id, e)
            return

        self.r.xack(self.cfg.stream, self.cfg.group, msg_id)
        log.info("[OK] processed %s (video %s)", msg_id, event.video_id)

    def _already_processed(self, video_id: str) -> bool:
        """A video is considered processed if its Mongo doc carries the current
        ANALYZER_VERSION. Steps 3.5-3.9 will set this field after embedding;
        for now it's never present, so this returns False and every event
        flows through to the handler."""
        doc = self.db.videos.find_one(
            {"video_id": video_id, "analyzer_version": self.cfg.analyzer_version},
            projection={"_id": 1},
        )
        return doc is not None


# --- Helpers ---

def _to_str(x) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _decode_fields(fields) -> dict:
    """Redis Streams fields come back as either dict[bytes,bytes] or
    dict[str,str] depending on the client's `decode_responses` setting.
    Normalize to dict[str,str]."""
    out: dict[str, str] = {}
    items = fields.items() if isinstance(fields, dict) else fields
    for k, v in items:
        out[_to_str(k)] = _to_str(v)
    return out
