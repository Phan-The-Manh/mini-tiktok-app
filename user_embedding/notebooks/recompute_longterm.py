"""
recompute_longterm.py — long-term embedding batch (step 4.7)
===========================================================

Rebuilds each user's `long_term_embedding` from a window of their positive
interactions. Runs the same math core as the live consumer, so long- and
short-term vectors stay in the same 384-d space.

This is a batch job, not a streaming one — `long_term_embedding` captures
durable interests and only needs refreshing periodically (nightly cron, or a
free Colab/Kaggle GPU run for a large user base). It pulls work from Mongo,
so a re-invoke after interruption simply recomputes from scratch and is safe
to run repeatedly.

Run:
    python -m user_embedding.notebooks.recompute_longterm --limit 100
    python -m user_embedding.notebooks.recompute_longterm --user u_abc123
    python -m user_embedding.notebooks.recompute_longterm --dry-run

In a Colab cell, set MONGO_URI in os.environ first (see content_analyzer's
batch_embed for the pattern), then call the module the same way.

Algorithm per user
------------------
1. Fetch interactions in the last `--window-days` days.
2. Keep only positive-signal rows (action_weight > 0); use that weight.
3. Batch-fetch the videos' content_embeddings; drop un-embedded ones.
4. `aggregate_long_term` -> weighted, re-normalized mean.
5. Persist via store.set_long_term (skipped on --dry-run).
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import user_embedding._path  # noqa: F401  side-effect: sys.path + env

from user_embedding.services import store
from user_embedding.services import math_core as mc


def recompute_one(user_id: str, since: datetime, *, dry_run: bool) -> dict:
    rows = store.get_positive_interaction_videos(user_id, since=since)

    # Keep positive-signal interactions and their weights.
    weighted: list[tuple[str, float]] = []
    for video_id, action, watch_pct, is_completion in rows:
        w = mc.action_weight(action, watch_pct=watch_pct, is_completion=is_completion)
        if w > 0:
            weighted.append((video_id, w))

    if not weighted:
        return {"user_id": user_id, "interactions": len(rows),
                "positive": 0, "embedded": 0, "written": False}

    video_ids = [v for v, _ in weighted]
    embs = store.get_video_embeddings(video_ids)

    vecs: list[list[float]] = []
    weights: list[float] = []
    for video_id, w in weighted:
        e = embs.get(video_id)
        if e:  # exists and non-empty
            vecs.append(e)
            weights.append(w)

    if not vecs:
        return {"user_id": user_id, "interactions": len(rows),
                "positive": len(weighted), "embedded": 0, "written": False}

    long_vec = mc.aggregate_long_term(vecs, weights=weights)
    if not dry_run:
        store.set_long_term(user_id, long_vec)

    return {"user_id": user_id, "interactions": len(rows),
            "positive": len(weighted), "embedded": len(vecs),
            "dim": int(long_vec.shape[0]), "written": not dry_run}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recompute_longterm")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max users to process (default: all).")
    parser.add_argument("--user", type=str, default=None,
                        help="Recompute a single user_id and exit.")
    parser.add_argument("--window-days", type=int,
                        default=int(os.getenv("USER_EMBEDDING_LONGTERM_WINDOW_DAYS", "30")),
                        help="Interaction lookback window in days.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report, but do not write to Mongo.")
    args = parser.parse_args(argv)

    since = datetime.now(timezone.utc) - timedelta(days=args.window_days)
    print(f"[INFO] recompute long-term since {since.isoformat()} "
          f"(window={args.window_days}d, dry_run={args.dry_run})")

    user_ids = [args.user] if args.user else list(store.iter_users(limit=args.limit))
    written = 0
    for uid in user_ids:
        res = recompute_one(uid, since, dry_run=args.dry_run)
        if res["written"]:
            written += 1
        if res.get("embedded", 0) > 0:
            print(f"[OK] {uid}: positive={res['positive']} embedded={res['embedded']} "
                  f"written={res['written']}")

    print(f"[OK] processed {len(user_ids)} users, wrote {written} long-term vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
