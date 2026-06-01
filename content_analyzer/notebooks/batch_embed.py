"""
batch_embed.py — Colab / Kaggle batch embedder
==============================================

Step 3.11 of the Content Analyzer build.

The local laptop worker (`python -m content_analyzer.main`) is fine for
trickling new uploads through CLIP + Whisper + MiniLM, but a cold backfill
of hundreds of videos is painful on CPU. This script runs the *same*
encoder code path on a free Colab T4 / Kaggle P100 GPU and writes the
embeddings back to the same MongoDB Atlas cluster — no schema changes,
no separate pipeline.

It is **pull-based, not consumer-based**: rather than reading
`video.uploaded` events, it queries `videos` directly for docs whose
`analyzer_version` is missing or stale, then runs the per-video handler
for each. Reasons:

  * Colab can't easily reach a local Redis (the stream lives there), and
    we don't want to require an ngrok / public Redis just for a one-off
    GPU burst.
  * Backfill is a natural pull workload — "embed everything that hasn't
    been embedded yet" — so pulling from Mongo is simpler than rewinding
    a stream.

Cost / quota
------------
Free Colab: ~12 hrs/day on T4, often pre-empted earlier. The script is
restartable: anything we already wrote to Mongo is filtered out of the
next query, so partial runs compose. Set `--limit` to bound a run and
re-invoke to continue.

Object storage caveat
---------------------
For Colab to download the video bytes the `videos.url` host must be
reachable from the public internet. The local `minio` container at
`localhost:9000` is not. Two options when running on Colab:

  1. Expose MinIO via `ngrok http 9000` and set `MINIO_ENDPOINT` to the
     tunnel host, OR
  2. Migrate the bucket to a public S3-compatible store (Cloudflare R2 /
     Backblaze B2) — the `client.py` factory needs no code change, only
     env vars.

Usage (Colab cell)
------------------
    !git clone https://github.com/Phan-The-Manh/mini-tiktok-app.git
    %cd mini-tiktok-app
    !pip install -q -r content_analyzer/requirements.txt

    import os
    os.environ["MONGO_URI"]         = "mongodb+srv://..."
    os.environ["MINIO_ENDPOINT"]    = "<ngrok-or-r2-host>"
    os.environ["MINIO_ACCESS_KEY"]  = "..."
    os.environ["MINIO_SECRET_KEY"]  = "..."
    os.environ["MINIO_SECURE"]      = "true"   # if HTTPS
    os.environ["ANALYZER_VERSION"]  = "v1"

    !python -m content_analyzer.notebooks.batch_embed --limit 100 --device cuda

Locally (CPU) it works too — same flags, leave `--device` off. Useful
for dry-running the script before paying for the Colab boot time.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Iterable

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env
from client import get_mongo  # type: ignore[import-not-found]

from content_analyzer.main import build_handler
from content_analyzer.schemas.events import VideoUploadedEvent
from content_analyzer.services.audio import WhisperTranscriber
from content_analyzer.services.text import TextEncoder
from content_analyzer.services.visual import CLIPVisualEncoder


log = logging.getLogger("batch_embed")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="content_analyzer.notebooks.batch_embed",
        description=(
            "Pull videos that need embedding from MongoDB and run the same "
            "encoder pipeline used by the streaming worker. Designed to run "
            "on Colab / Kaggle GPUs but also works locally on CPU."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of videos to process in this run (default 100).",
    )
    p.add_argument(
        "--analyzer-version",
        default=None,
        help=(
            "Override `ANALYZER_VERSION`. Defaults to the env value (or 'v1'). "
            "Videos whose stored `analyzer_version` does not match this are "
            "selected for re-embedding."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help=(
            "Torch device for all three encoders. 'auto' picks 'cuda' when "
            "available, else 'cpu'."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the video_ids that would be processed and exit.",
    )
    return p.parse_args(argv)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _find_pending(db, version: str, limit: int) -> Iterable[dict]:
    """Yield up to `limit` `videos` docs whose embedding is missing or stale.

    A doc is "pending" if its `analyzer_version` is missing/different OR
    its `content_embedding` is an empty array. We project only the fields
    `build_handler`'s closure actually needs (video_id, author_id, url,
    duration_seconds). Caption is re-fetched inside the handler.
    """
    query = {
        "$or": [
            {"analyzer_version": {"$ne": version}},
            {"content_embedding": {"$size": 0}},
        ]
    }
    projection = {
        "_id": 0,
        "video_id": 1,
        "author_id": 1,
        "url": 1,
        "duration_seconds": 1,
        "uploaded_at": 1,
    }
    return db.videos.find(query, projection=projection).limit(limit)


def _to_event(doc: dict) -> VideoUploadedEvent:
    """Synthesize the event the streaming handler expects from a Mongo doc."""
    payload: dict = {
        "video_id": doc["video_id"],
        "author_id": doc["author_id"],
        "url": doc["url"],
        "duration_seconds": float(doc.get("duration_seconds") or 0.0),
    }
    # Only pass uploaded_at if Mongo had one — passing an explicit None
    # bypasses the model's default_factory and trips validation.
    if doc.get("uploaded_at") is not None:
        payload["uploaded_at"] = doc["uploaded_at"]
    return VideoUploadedEvent(**payload)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    device = _resolve_device(args.device)
    log.info("[INFO] batch_embed starting (device=%s, limit=%d)", device, args.limit)

    db = get_mongo()

    # Resolve the analyzer version the same way the writer does — env first,
    # CLI flag wins if provided. We also pass it down to the writer via env
    # so the streaming handler picks it up without rewiring its signature.
    import os

    version = args.analyzer_version or os.getenv("ANALYZER_VERSION", "v1")
    os.environ["ANALYZER_VERSION"] = version

    docs = list(_find_pending(db, version=version, limit=args.limit))
    log.info("[INFO] %d videos to embed at analyzer_version=%s", len(docs), version)

    if not docs:
        log.info("[OK] nothing to do")
        return 0

    if args.dry_run:
        for d in docs:
            log.info("[DRY] would embed %s (%s)", d["video_id"], d["url"])
        return 0

    # Load models on the chosen device. Each `load()` is idempotent.
    visual = CLIPVisualEncoder(device=device); visual.load()
    text = TextEncoder(device=device); text.load()
    whisper = WhisperTranscriber(device=device); whisper.load()

    handler = build_handler(visual, text, whisper, db)

    ok = 0
    fail = 0
    started = time.time()
    for i, doc in enumerate(docs, start=1):
        vid = doc.get("video_id", "?")
        try:
            handler(_to_event(doc))
            ok += 1
        except Exception as e:
            # Batch must not abort on a single bad video — log + continue.
            log.exception("[FAIL] %s (%d/%d): %s", vid, i, len(docs), e)
            fail += 1

        # Periodic progress so a long Colab run gives feedback.
        if i % 10 == 0 or i == len(docs):
            elapsed = time.time() - started
            rate = i / elapsed if elapsed > 0 else 0.0
            log.info(
                "[INFO] progress %d/%d (ok=%d fail=%d, %.1f vid/min)",
                i, len(docs), ok, fail, rate * 60,
            )

    log.info("[OK] batch_embed done: ok=%d fail=%d total=%d", ok, fail, len(docs))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
