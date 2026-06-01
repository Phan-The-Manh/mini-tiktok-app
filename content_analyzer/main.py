"""
main.py — Content Analyzer worker entry point
=============================================

Step 3.10 of the Content Analyzer build.

Wires the consumer (step 3.2) to the embedding pipeline (steps 3.3-3.8)
and the MongoDB write-back (step 3.9). One process per worker; horizontal
scaling is "start another `python -m content_analyzer.main` with a
different `CONTENT_ANALYZER_CONSUMER` name".

Run:
    python -m content_analyzer.main             # forever loop
    python -m content_analyzer.main --once      # one reclaim + one read pass, then exit

Startup order
-------------
1. Configure logging.
2. Pre-load CLIP, Whisper-tiny, and MiniLM. We load *before* joining the
   consumer group so that a missing weight file fails the process
   immediately rather than DLQ'ing the first batch of events. The three
   weights total ~300 MB and a cold load is 5-15 s on CPU.
3. Build the per-video handler as a closure over the loaded models.
4. Start the consumer loop. Graceful shutdown on SIGINT/SIGTERM is
   wired to `consumer.stop()`, which exits at the next iteration
   boundary (worst case ~`CONTENT_ANALYZER_BLOCK_MS` later).

Per-video pipeline
------------------
For each `video.uploaded` event the handler:

    1. Looks up the Mongo doc to read the creator-supplied `caption`
       (the event payload deliberately does not carry it; caption is
       editable post-upload and the doc is the source of truth).
    2. Downloads the file from MinIO to a temp dir.
    3. Samples N frames with ffmpeg and runs CLIP -> 512-d visual.
    4. Extracts a 16 kHz mono wav and runs Whisper-tiny -> transcript
       (empty string for silent / audio-less videos).
    5. Encodes `caption + transcript` with MiniLM -> 384-d text.
    6. Fuses visual + text -> 384-d unit vector.
    7. Writes the embedding + transcript + analyzer_version +
       analyzed_at back to Mongo and flips `pending` to `approved`.

Anything that raises out of the handler is intentionally not ack'd by
the consumer; the message stays pending and is retried / DLQ'd by the
existing reclaim path.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env
from client import get_mongo  # type: ignore[import-not-found]

from content_analyzer.consumers.video_uploaded import (
    VideoUploadedConsumer,
)
from content_analyzer.schemas.events import VideoUploadedEvent
from content_analyzer.services.audio import (
    WhisperTranscriber,
    transcribe_video,
)
from content_analyzer.services.downloader import downloaded
from content_analyzer.services.frames import frames
from content_analyzer.services.fuse import fuse
from content_analyzer.services.text import TextEncoder
from content_analyzer.services.visual import CLIPVisualEncoder
from content_analyzer.services.writer import write_back


log = logging.getLogger("content_analyzer")


# --- Pipeline handler --------------------------------------------------------

class HandlerError(RuntimeError):
    """Raised when the per-video pipeline cannot complete."""


def build_handler(
    visual_encoder: CLIPVisualEncoder,
    text_encoder: TextEncoder,
    transcriber: WhisperTranscriber,
    db,
):
    """Return a `(event) -> None` callable closed over the preloaded models.

    The closure keeps the encoders out of module-level globals so that
    main() can be re-entered with different models in tests, and so that
    the consumer never accidentally triggers a lazy load on the hot path.
    """

    def handle(event: VideoUploadedEvent) -> None:
        vid = event.video_id
        log.info("[INFO] processing video %s", vid)

        # 1. Caption from the Mongo doc (event payload doesn't carry it).
        doc = db.videos.find_one(
            {"video_id": vid},
            projection={"caption": 1},
        )
        if doc is None:
            # No retry can fix a deleted-mid-flight upload. Raise so the
            # consumer's DLQ path takes the message after max retries.
            raise HandlerError(f"video {vid!r} not found in mongo")
        caption = doc.get("caption") or ""

        # 2-4. Download -> frames -> visual; audio -> transcript.
        # Both temp dirs are cleaned up by their context managers, on
        # success and on any exception raised inside the `with`.
        with downloaded(event.url) as video_path:
            with frames(video_path) as frame_paths:
                visual_vec = visual_encoder.encode(frame_paths)
            transcript = transcribe_video(video_path, transcriber=transcriber)

        # 5-6. Text + fusion.
        text_vec = text_encoder.encode(caption, transcript)
        embedding = fuse(visual_vec, text_vec)

        # 7. Persist.
        write_back(vid, embedding, transcript)

        log.info(
            "[OK] video %s embedded (frames=%d, transcript_chars=%d, dim=%d)",
            vid, len(frame_paths), len(transcript), embedding.shape[0],
        )

    return handle


# --- Setup -------------------------------------------------------------------

def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="content_analyzer",
        description=(
            "Consume video.uploaded events and embed each video into "
            "MongoDB.videos.content_embedding."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Perform a single reclaim + read pass and exit. Used by the "
            "smoke test; not for production deployment."
        ),
    )
    return parser.parse_args(argv)


def _install_signal_handlers(consumer: VideoUploadedConsumer) -> None:
    def _stop(signum, _frame):
        log.info("[INFO] received signal %d, stopping at next iteration", signum)
        consumer.stop()

    signal.signal(signal.SIGINT, _stop)
    # SIGTERM exists on POSIX; on Windows the symbol is defined but only fires
    # for the current process via `os.kill(pid, signal.SIGTERM)`.
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        try:
            signal.signal(sigterm, _stop)
        except (ValueError, OSError):
            # Some Windows shells refuse SIGTERM installation; not fatal.
            pass


# --- Entry point -------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging()

    log.info("[INFO] starting content_analyzer worker (once=%s)", args.once)

    # Pre-load all three models. Each `load()` is idempotent.
    visual_encoder = CLIPVisualEncoder()
    visual_encoder.load()

    text_encoder = TextEncoder()
    text_encoder.load()

    transcriber = WhisperTranscriber()
    transcriber.load()

    db = get_mongo()
    handler = build_handler(visual_encoder, text_encoder, transcriber, db)

    consumer = VideoUploadedConsumer(handler=handler, mongo_db=db)
    _install_signal_handlers(consumer)

    try:
        consumer.run(once=args.once)
    except KeyboardInterrupt:
        # Belt-and-suspenders: the signal handler already calls stop(),
        # but if it runs *between* iterations the loop sees stopping=True
        # immediately. If it fires inside a blocking XREADGROUP, redis-py
        # raises KeyboardInterrupt instead — handle that cleanly too.
        log.info("[INFO] KeyboardInterrupt, exiting")

    log.info("[OK] content_analyzer worker stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
