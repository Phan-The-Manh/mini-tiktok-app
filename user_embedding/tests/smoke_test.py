"""
smoke_test.py — end-to-end User Embedding Service check (step 4.10)
==================================================================

Requires live Redis + MongoDB (Atlas). Run:

    python -m user_embedding.tests.smoke_test

What it does (all under a unique `_smoke_{uuid}` prefix so it never collides
with real data or a running worker):

    1. Stage a fresh user (cold start) + two videos with known 384-d unit
       content_embeddings directly in Mongo.
    2. Assert the user reads as cold-start (empty query vector) to begin.
    3. Publish two synthetic `user.action` "like" events to an isolated
       stream and run the real consumer once (--once path).
    4. Assert short_term moved toward the liked videos (cosine > 0), is
       unit-norm and 384-d, and that the Redis cache agrees with Mongo.
    5. Assert the served query vector is now warm (cold_start=false).
    6. Re-deliver the same events and assert the short_term is unchanged
       (idempotency on interaction_id).
    7. Assert the DLQ is empty.

Cleanup (Mongo docs, Redis stream/group/DLQ, cache + seen keys) runs in a
`finally` regardless of outcome.
"""

from __future__ import annotations

import uuid

import numpy as np

import user_embedding._path  # noqa: F401  side-effect: sys.path + env
from client import get_mongo, get_redis  # type: ignore[import-not-found]

from user_embedding.consumers.user_action import ConsumerConfig, UserActionConsumer
from user_embedding.schemas.events import UserActionEvent
from user_embedding.services import cache, store
from user_embedding.services import math_core as mc
from user_embedding.services import update as update_svc

DIM = mc.EMBEDDING_DIM


def _unit(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x) for x in mc.l2_normalize(rng.standard_normal(DIM)).tolist()]


def main() -> int:
    uid = uuid.uuid4().hex[:8]
    db = get_mongo()
    r = get_redis()

    user_id = f"_smoke_u_{uid}"
    video_ids = [f"_smoke_v_{uid}_0", f"_smoke_v_{uid}_1"]
    video_vecs = [_unit(101), _unit(202)]
    interaction_ids = [f"_smoke_i_{uid}_0", f"_smoke_i_{uid}_1"]

    cfg = ConsumerConfig(
        stream=f"_smoke_user.action_{uid}",
        group=f"_smoke_grp_{uid}",
        consumer="smoke",
        dlq_list=f"_smoke_user.action.dlq_{uid}",
        max_retries=3,
        block_ms=1000,
        claim_idle_ms=10**9,  # effectively disable reclaim during the smoke
    )

    try:
        # 1. Stage user + videos.
        db.users.insert_one({
            "user_id": user_id,
            "username": user_id,
            "long_term_embedding": [],
            "short_term_embedding": [],
        })
        for vid, vec in zip(video_ids, video_vecs):
            db.videos.insert_one({"video_id": vid, "content_embedding": vec})
        print(f"[OK] staged user {user_id} + {len(video_ids)} videos")

        # 2. Cold start to begin with.
        q0 = update_svc.get_query_vector(user_id)
        assert q0["cold_start"] is True, q0
        assert q0["embedding"] == [], q0
        print("[OK] user starts cold (empty query vector)")

        # 3. Publish synthetic 'like' events to the isolated stream.
        consumer = UserActionConsumer(config=cfg)
        consumer.ensure_group()
        for iid, vid in zip(interaction_ids, video_ids):
            ev = UserActionEvent(
                interaction_id=iid, user_id=user_id, video_id=vid, action="like",
            )
            r.xadd(cfg.stream, ev.to_stream_fields())
        print(f"[OK] published {len(interaction_ids)} synthetic user.action events")

        consumer.run(once=True)

        # 4. Short-term moved toward the liked videos.
        vectors = store.get_user_vectors(user_id)
        assert vectors is not None
        _long, short = vectors
        assert len(short) == DIM, f"dim={len(short)}"
        assert abs(np.linalg.norm(short) - 1.0) < 1e-6, "short-term not unit-norm"
        for vec in video_vecs:
            assert mc.cosine(short, vec) > 0.0, "short-term not pulled toward a liked video"
        # The last-processed video dominates a 0.9-decay EMA.
        assert mc.cosine(short, video_vecs[-1]) > 0.05
        print(f"[OK] short_term is {DIM}-d unit-norm and pulled toward liked videos")

        # Cache agrees with Mongo (write-through).
        cached_short = cache.get_short_term(user_id)
        assert cached_short is not None, "short-term missing from cache"
        assert np.allclose(np.asarray(cached_short), np.asarray(short), atol=1e-9), \
            "cache and Mongo disagree"
        print("[OK] Redis cache matches Mongo (write-through verified)")

        # 5. Served query vector is now warm.
        q1 = update_svc.get_query_vector(user_id)
        assert q1["cold_start"] is False, q1
        assert q1["dim"] == DIM and len(q1["embedding"]) == DIM, q1
        assert q1["has_short_term"] is True and q1["has_long_term"] is False, q1
        print("[OK] query vector is warm after interactions")

        # 6. Idempotency: re-deliver the same events; short_term unchanged.
        for iid, vid in zip(interaction_ids, video_ids):
            ev = UserActionEvent(
                interaction_id=iid, user_id=user_id, video_id=vid, action="like",
            )
            r.xadd(cfg.stream, ev.to_stream_fields())
        consumer.run(once=True)
        _long2, short2 = store.get_user_vectors(user_id)
        assert np.allclose(np.asarray(short2), np.asarray(short), atol=1e-9), \
            "re-delivery changed short_term (idempotency broken)"
        print("[OK] re-delivering the same events is a no-op (idempotent)")

        # 7. DLQ empty.
        assert r.llen(cfg.dlq_list) == 0, "unexpected DLQ entries"
        print("[OK] DLQ is empty")

        print("\n[OK] user_embedding smoke test passed")
        return 0

    finally:
        # --- Cleanup (best-effort) ---
        try:
            db.users.delete_one({"user_id": user_id})
            db.videos.delete_many({"video_id": {"$in": video_ids}})
        except Exception as e:
            print(f"[WARN] mongo cleanup: {e}")
        try:
            r.delete(cfg.stream, cfg.dlq_list)
            try:
                r.xgroup_destroy(cfg.stream, cfg.group)
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] redis cleanup: {e}")
        try:
            cache.forget_user(user_id)
            for vid in video_ids:
                cache.forget_video(vid)
            for iid in interaction_ids:
                cache.forget_interaction(iid)
        except Exception as e:
            print(f"[WARN] cache cleanup: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
