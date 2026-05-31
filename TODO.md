# TODO / Build Progress

> Live tracker for build order, the current stage, and completed-stage history.
> Architecture, rules, and component boundaries live in `CLAUDE.md` — keep those there.

---

## Build Order

| Order | Component | Why This Order | Status |
|-------|-----------|---------------|--------|
| 1 | **Database Layer** | Foundation — every other component reads/writes here | `[DONE]` |
| 2 | **Upload Service** | Need a way to get videos into the system | `[DONE]` |
| 3 | **Content Analyzer** | Videos must be embedded before they can be recommended | **CURRENT FOCUS** |
| 4 | User Embedding Service | Needed before Recall can use user vectors | Pending |
| 5 | Recall Service | First half of the recommendation funnel | Pending |
| 6 | Feed API (skeleton) | End-to-end "random feed" working before adding ranking | Pending |
| 7 | Frontend (basic swipe UI) | Need to actually see the feed working | Pending |
| 8 | Event Service | Capture user actions to drive learning | Pending |
| 9 | Stream Processors | Close the feedback loop: events to embedding updates | Pending |
| 10 | Ranking Service (LightGBM) | Upgrade the funnel from "similar" to "personalized score" | Pending |
| 11 | Re-ranker | Diversity + cold-start polish | Pending |
| 12 | A/B Testing Framework | Final layer — measure improvements | Pending |

---

## CURRENT STAGE: Content Analyzer (Component #3)

The Content Analyzer is a **background worker**, not an HTTP service. It consumes `video.uploaded` events emitted by the Upload Service, fetches each video from MinIO, runs multi-modal embedding (CLIP + Whisper + MiniLM), fuses the result into a single vector, and writes it back to the `videos` collection so the video becomes searchable by Atlas Vector Search.

### Goals (broken down into small, testable steps)

1. **Project skeleton.** Standalone `content_analyzer/` package with its own `requirements.txt`, `.env.example`, and `_path.py` that reuses `database/client.py` and `database/schemas/`.
2. **Redis Streams consumer.** Read from stream `video.uploaded` using a named consumer group (`content_analyzer`), with `XACK` on success and a dead-letter list (`video.uploaded.dlq`) after N failed retries. Idempotent: re-processing the same `video_id` must not duplicate work.
3. **Video download.** Given the MinIO object key from the event payload, stream the file to a local temp path. Clean up temp files on success and failure.
4. **Frame extraction.** Use ffmpeg to sample N evenly-spaced frames (default 8) as JPEGs into a temp directory. Falls back gracefully if the video is shorter than expected.
5. **Visual encoding (CLIP).** Load `openai/clip-vit-base-patch32` once at startup. Encode each frame, mean-pool the per-frame embeddings into a single 512-d visual vector. CPU-only.
6. **Audio extraction + transcription (Whisper).** Use ffmpeg to extract a mono 16 kHz wav from the video. Run `whisper-tiny` over it. If the audio is silent or under ~0.5 s, skip transcription cleanly and use an empty transcript. `[DONE]` — `services/audio.py` (extract + silence gate + lazy-loaded `WhisperTranscriber`) + `tests/_smoke_audio.py`.
7. **Text encoding (MiniLM).** Concatenate caption + transcript (truncated to model limit), encode with `sentence-transformers/all-MiniLM-L6-v2` to get a 384-d text vector. `[DONE]` — `services/text.py` (lazy-loaded `TextEncoder`, caption-first concat, zero-vector for empty input) + `tests/_smoke_text.py`.
8. **Modal fusion.** Combine visual (512-d) + text (384-d) into a single `content_embedding` whose dimension matches the Atlas Vector Search index on `videos.content_embedding`. Initial strategy: L2-normalize each, concatenate or project to the index dim (decide based on `database/scripts/vector_index_def.json`). Document the chosen strategy in the README. `[DONE]` — `services/fuse.py` (deterministic Gaussian projection 512->384, per-modality L2-normalize, alpha-weighted sum, final L2-normalize) + `tests/_smoke_fuse.py`. Index is 384-d cosine, so concatenation was not viable; projection chosen over learned reduction because no training data exists yet.
9. **MongoDB write-back.** Update the `videos` document with `content_embedding`, `transcript`, `analyzer_version`, `analyzed_at`, and flip `moderation_status` from `pending` to `ready` (moderation itself is still a no-op stub at this stage).
10. **Worker entry point.** `python -m content_analyzer.main` starts the consumer loop with graceful shutdown on Ctrl+C / SIGTERM. Structured `[OK] / [WARN] / [FAIL]` log lines (no emojis).
11. **Colab fallback.** A small `notebooks/batch_embed.ipynb` (or `.py`) that can be run on Colab/Kaggle to batch-embed many videos against the same MongoDB cluster, using the same encoder code path. Documented in the README.
12. **Smoke test.** End-to-end: pick a seeded video (or upload one via the Upload Service), publish a synthetic `video.uploaded` event if needed, run the worker once with `--once` mode, and assert that `content_embedding` is non-empty, has the correct dimension, and that a vector-search query for the same vector returns the video as the top hit.

### Out of Scope (deferred to later components)
- Real moderation logic (still flips status to `ready` blindly; a separate moderation step lands later)
- Backfill of pre-existing un-embedded videos (separate batch script, post-MVP)
- GPU acceleration on the local laptop (Colab burst is the documented fallback)
- Fine-tuning or distilling any of the encoder models
- Re-embedding when a video's caption is edited

### Deliverables for This Stage
```
content_analyzer/
├── README.md
├── requirements.txt
├── .env.example
├── _path.py                # adds database/ to sys.path, loads .env
├── main.py                 # consumer loop entry point (--once flag for tests)
├── consumers/
│   └── video_uploaded.py   # Redis Streams XREADGROUP loop + ack/dlq
├── services/
│   ├── downloader.py       # MinIO object -> local temp file
│   ├── frames.py           # ffmpeg frame sampling
│   ├── audio.py            # ffmpeg audio extract + Whisper-tiny transcribe
│   ├── visual.py           # CLIP frame encoder (lazy-loaded)
│   ├── text.py             # MiniLM caption+transcript encoder
│   ├── fuse.py             # modal fusion into final content_embedding
│   └── writer.py           # MongoDB update + status flip
├── schemas/
│   └── events.py           # VideoUploadedEvent (re-used) + optional VideoEmbeddedEvent
├── notebooks/
│   └── batch_embed.py      # Colab-runnable batch path
└── tests/
    └── smoke_test.py
```

### Definition of Done
- [ ] `python -m content_analyzer.main` starts the worker and joins consumer group `content_analyzer` on stream `video.uploaded`
- [ ] Models load once at startup; per-video processing reuses them
- [ ] Posting a new video via the Upload Service results in a non-empty `content_embedding` on the Mongo doc within ~60s on CPU
- [ ] `content_embedding` length matches the Atlas Vector Search index dimension
- [ ] `transcript`, `analyzer_version`, and `analyzed_at` are populated on the doc
- [ ] `moderation_status` transitions from `pending` to `ready` after successful embedding
- [ ] Failed messages land in `video.uploaded.dlq` after the configured retry count, with the failure reason
- [ ] Re-delivering the same `video.uploaded` event is a no-op (idempotency check on `video_id` + `analyzer_version`)
- [ ] `python -m content_analyzer.tests.smoke_test` passes end-to-end and asserts vector-search retrieves the embedded video
- [ ] No duplication of MongoDB/Redis/MinIO connection code — uses `database/client.py`
- [ ] README documents: ffmpeg requirement, expected CPU runtime per video, the Colab batch fallback, and the chosen fusion strategy

---

## COMPLETED STAGES

### Component #2: Upload Service `[DONE]`

FastAPI service that accepts uploads, transcodes to mp4 via FFmpeg (with passthrough fallback), stores files in MinIO, writes a `Video` doc to MongoDB, and emits a `video.uploaded` event on Redis Streams. Full contract and run instructions: `upload_service/README.md`.

Definition of Done (all met):
- [x] `uvicorn upload_service.main:app --port 8001` starts the service
- [x] `GET /health` returns `ok` for mongo+redis+minio (and reports whether ffmpeg is available)
- [x] `POST /videos` with a real mp4 returns `201` + `video_id`
- [x] `GET /videos/{id}` returns the persisted Video doc
- [x] The file is present in MinIO under `videos/{video_id}.mp4`
- [x] A new entry exists in Redis Stream `video.uploaded` carrying `video_id`, `author_id`, `url`, `duration_seconds`, `uploaded_at`
- [x] `python -m upload_service.tests.smoke_test` passes end-to-end
- [x] No duplication of MongoDB/Redis/MinIO connection code — uses `database/client.py`
- [x] README documents how a teammate could run the service standalone, including the ffmpeg PATH requirement and the passthrough fallback

### Component #1: Database Layer `[DONE]`

MongoDB Atlas M0 + local Redis + local MinIO via Docker Compose, four core collections (`users`, `videos`, `interactions`, `experiments`), Pydantic schemas, seed script (50 users + 200 videos with random embeddings), Atlas Vector Search index on `videos.content_embedding`, and a shared `client.py` factory reused by every later component. Full setup and the manual Atlas UI steps for the vector index: `database/README.md`.

Definition of Done (all met):
- [x] MongoDB Atlas M0 cluster created, connection string in `.env`
- [x] Docker Compose runs Redis + MinIO locally with `docker compose up -d`
- [x] All four collections exist with documents
- [x] All standard indexes created and verified via `db.<collection>.getIndexes()`
- [x] Vector Search index `video_content_index` is **ACTIVE** in Atlas UI
- [x] `python scripts/seed_data.py` populates the DB with 50 users + 200 videos
- [x] `python scripts/smoke_test.py` passes all checks (Mongo / Redis / MinIO connectivity, indexes present, vector search returns results)
- [x] `schemas/` package is importable from a sibling directory
- [x] `README.md` documents how a teammate could rebuild this from scratch, including the manual Atlas UI steps for the vector index
