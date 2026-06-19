"""
user_action.py — Redis Streams consumer for `user.action` (step 4.6)
===================================================================

Consumes interaction events and folds each into the acting user's short-term
vector. Mirrors the Content Analyzer consumer's reliability model:

- **Named consumer group** (`user_embedding`); scale by starting more workers
  with the same group and distinct consumer names.
- **Idempotency.** EMA updates are NOT naturally idempotent (applying twice
  shifts twice), so we gate on `interaction_id` via a Redis "seen" marker set
  only AFTER a successful apply. A redelivery of an already-applied event is a
  no-op.
- **Retry + DLQ.** Handler errors leave the message pending; `XAUTOCLAIM`
  reclaims stale entries each pass and routes them to `user.action.dlq` once
  past `max_retries`. Permanent errors (unknown user / deleted video /
  unparseable payload) go straight to the DLQ without burning retries.
  `VideoNotEmbedded` is treated as transient (the Content Analyzer may catch
  up), so it retries.

Run:
    python -m user_embedding.consumers.user_action            # forever loop
    python -m user_embedding.consumers.user_action --once     # one pass, exit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import redis  # type: ignore[import-untyped]

import user_embedding._path  # noqa: F401  side-effect: sys.path + env
from client import get_redis  # type: ignore[import-not-found]

from user_embedding.schemas.events import UserActionEvent
from user_embedding.services import cache
from user_embedding.services import update as update_svc

log = logging.getLogger("user_embedding.consumer")

Applier = Callable[[UserActionEvent], dict]


@dataclass(frozen=True)
class ConsumerConfig:
    stream: str
    group: str
    consumer: str
    dlq_list: str
    max_retries: int
    block_ms: int
    claim_idle_ms: int
    batch_size: int = 10

    @classmethod
    def from_env(cls) -> "ConsumerConfig":
        return cls(
            stream=os.getenv("USER_ACTION_STREAM", "user.action"),
            group=os.getenv("USER_EMBEDDING_GROUP", "user_embedding"),
            consumer=os.getenv("USER_EMBEDDING_CONSUMER", "worker-1"),
            dlq_list=os.getenv("USER_EMBEDDING_DLQ", "user.action.dlq"),
            max_retries=int(os.getenv("USER_EMBEDDING_MAX_RETRIES", "3")),
            block_ms=int(os.getenv("USER_EMBEDDING_BLOCK_MS", "5000")),
            claim_idle_ms=int(os.getenv("USER_EMBEDDING_CLAIM_IDLE_MS", "60000")),
        )


# Errors that can never succeed on retry -> DLQ immediately.
_PERMANENT = (update_svc.UnknownUser, update_svc.UnknownVideo)


class UserActionConsumer:
    def __init__(
        self,
        config: ConsumerConfig | None = None,
        applier: Applier | None = None,
        redis_client: "redis.Redis | None" = None,
    ):
        self.cfg = config or ConsumerConfig.from_env()
        self.apply = applier or update_svc.apply_interaction
        self.r = redis_client if redis_client is not None else get_redis()
        self._stopping = False

    # --- Lifecycle ---

    def ensure_group(self) -> None:
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
                return
            raise

    def stop(self) -> None:
        self._stopping = True

    def run(self, once: bool = False) -> None:
        self.ensure_group()
        while not self._stopping:
            self._reclaim_stale()
            self._read_new()
            if once:
                return

    # --- Stale reclaim + DLQ routing ---

    def _reclaim_stale(self) -> None:
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
            name=self.cfg.stream, groupname=self.cfg.group,
            min=msg_id, max=msg_id, count=1,
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
        for _stream, messages in (resp or []):
            for msg_id, fields in messages:
                self._handle_one(_to_str(msg_id), _decode_fields(fields))

    def _handle_one(self, msg_id: str, fields: dict) -> None:
        try:
            event = UserActionEvent.from_stream_fields(fields)
        except Exception as e:
            log.error("[FAIL] unparseable event %s: %s", msg_id, e)
            self._send_to_dlq(msg_id, fields, 1, reason=f"unparseable: {e}")
            self.r.xack(self.cfg.stream, self.cfg.group, msg_id)
            return

        if cache.is_processed(event.interaction_id):
            log.info("[OK] skip %s (interaction %s already applied)",
                     msg_id, event.interaction_id)
            self.r.xack(self.cfg.stream, self.cfg.group, msg_id)
            return

        try:
            result = self.apply(event)
        except _PERMANENT as e:
            log.error("[FAIL] permanent error on %s: %s", msg_id, e)
            self._send_to_dlq(msg_id, fields, self._delivery_count(msg_id),
                              reason=f"permanent: {e}")
            self.r.xack(self.cfg.stream, self.cfg.group, msg_id)
            return
        except Exception as e:
            # Transient (incl. VideoNotEmbedded) — leave pending for reclaim.
            log.warning("[WARN] transient error on %s (interaction %s): %s",
                        msg_id, event.interaction_id, e)
            return

        # Success (including no-op zero-weight) — mark seen + ack.
        cache.mark_processed(event.interaction_id)
        self.r.xack(self.cfg.stream, self.cfg.group, msg_id)
        log.info("[OK] processed %s (interaction %s, updated=%s)",
                 msg_id, event.interaction_id, result.get("updated"))


# --- Helpers -----------------------------------------------------------------

def _to_str(x) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _decode_fields(fields) -> dict:
    out: dict[str, str] = {}
    items = fields.items() if isinstance(fields, dict) else fields
    for k, v in items:
        out[_to_str(k)] = _to_str(v)
    return out


# --- Entry point -------------------------------------------------------------

def _configure_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="user_embedding.consumers.user_action")
    parser.add_argument("--once", action="store_true",
                        help="One reclaim + read pass, then exit (smoke test).")
    args = parser.parse_args(argv)

    _configure_logging()
    log.info("[INFO] starting user.action consumer (once=%s)", args.once)

    consumer = UserActionConsumer()

    def _stop(signum, _frame):
        log.info("[INFO] received signal %d, stopping", signum)
        consumer.stop()

    signal.signal(signal.SIGINT, _stop)
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        try:
            signal.signal(sigterm, _stop)
        except (ValueError, OSError):
            pass

    try:
        consumer.run(once=args.once)
    except KeyboardInterrupt:
        log.info("[INFO] KeyboardInterrupt, exiting")

    log.info("[OK] user.action consumer stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
