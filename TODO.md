# TODO / Build Progress

> Live tracker for build order, the current stage, and completed-stage history.
> Architecture, rules, and component boundaries live in `CLAUDE.md` — keep those there.

---

## Build Order

| Order | Component | Why This Order | Status |
|-------|-----------|---------------|--------|
| 1 | **Database Layer** | Foundation — every other component reads/writes here | `[DONE]` |
| 2 | **Upload Service** | Need a way to get videos into the system | `[DONE]` |
| 3 | **Content Analyzer** | Videos must be embedded before they can be recommended | `[DONE]` |
| 4 | **User Embedding Service** | Needed before Recall can use user vectors | `[DONE]` |
| 5 | Recall Service | First half of the recommendation funnel | **CURRENT FOCUS** |
| 6 | Feed API (skeleton) | End-to-end "random feed" working before adding ranking | Pending |
| 7 | Frontend (basic swipe UI) | Need to actually see the feed working | Pending |
| 8 | Event Service | Capture user actions to drive learning | Pending |
| 9 | Stream Processors | Close the feedback loop: events to embedding updates | Pending |
| 10 | Ranking Service (LightGBM) | Upgrade the funnel from "similar" to "personalized score" | Pending |
| 11 | Re-ranker | Diversity + cold-start polish | Pending |
| 12 | A/B Testing Framework | Final layer — measure improvements | Pending |

---

## CURRENT STAGE: User Embedding Service (Component #4)

A small FastAPI microservice plus a shared embedding-math library. It maintains two vectors per user in the **same 384-d space as `videos.content_embedding`** (so Recall can vector-search a user's vector directly against videos):

- `short_term_embedding` — current-session interests, updated in near real-time from interaction events via an exponential moving average (EMA).
- `long_term_embedding` — durable interests, recomputed in batch (nightly / Colab) from a long window of positive interactions.

It also serves a single blended **query vector** for the Recall Service (Component #5) and handles cold-start (a user with no history). The schema fields already exist on the `User` doc (`long_term_embedding`, `short_term_embedding`, `short_term_updated_at`).

> Sequencing note: the live `user.action` event stream is owned by the Event Service (Component #8), which is built later. Component #4 builds and tests its consumer against **synthetic events** so it stays developable in isolation; the wire-up to real events lands with #8/#9.

### Goals (broken down into small, testable steps)

1. **Project skeleton.** Standalone `user_embedding/` package with its own `requirements.txt`, `.env.example`, and `_path.py` that reuses `database/client.py` and `database/schemas/`.
2. **Embedding math core (pure NumPy, no I/O).** Action weights (like / complete-watch / watch / skip / not_interested), EMA short-term update `s' = normalize(decay*s + (1-decay)*weight(action)*v)`, long-term aggregation (weighted mean of positively-interacted video vectors), and the query-vector blend `q = normalize(beta*long + (1-beta)*short)`. All L2-normalized, deterministic, unit-tested with synthetic vectors.
3. **Mongo adapter.** Read a user's two embeddings; persist `short_term_embedding` + `short_term_updated_at` and `long_term_embedding` via partial `$set` updates. Read a video's `content_embedding` (needed to apply an interaction). Idempotent.
4. **Redis vector cache.** Hot-path cache of a user's short-term (and blended query) vector so reads/updates don't hit Mongo every time. Floats serialized (JSON or base64) because the shared Redis client uses `decode_responses=True`. TTL + write-through to Mongo.
5. **FastAPI service.** `GET /users/{user_id}/embedding` returns the blended query vector + metadata (dim, freshness, cold_start flag); `GET /health` checks Mongo + Redis. Optional dev-only `POST /users/{user_id}/interactions` to apply a synthetic interaction by hand.
6. **`user.action` Redis Streams consumer.** Consumer group `user_embedding` on stream `user.action`, mirroring the Content Analyzer's reliability pattern: XREADGROUP + XACK on success, idempotency on `interaction_id`, retry + `user.action.dlq` after N failures. For each event: look up the video's `content_embedding`, apply the EMA update, write through cache + Mongo. Tested with synthetic events (no Event Service required).
7. **Long-term recompute batch.** A Colab/Kaggle-runnable `notebooks/recompute_longterm.py` that recomputes `long_term_embedding` for all (or stale) users from their positive interactions over a window, reusing the same math core. Flags: `--limit`, `--window-days`, `--dry-run`.
8. **Cold-start handling.** When a user has neither embedding, return a defined fallback (empty vector + `cold_start=true`) so Recall can branch to demographics/trending instead of vector search. Document the contract.
9. **README.** Document the HTTP contract, the `user.action` event shape, env vars, how to run standalone with mocked deps, the chosen EMA decay + action weights + blend beta, the cache strategy, and the cold-start behavior.
10. **Smoke test.** End-to-end: seed a user, publish a few synthetic `user.action` events referencing videos with known `content_embedding`s, run the consumer once, and assert the user's `short_term_embedding` moved toward those videos (cosine similarity increased), is unit-norm and 384-d, that cache and Mongo agree, and that `GET /users/{id}/embedding` returns a valid query vector. Cleanup in a `finally`.

### Out of Scope (deferred to later components)
- Wiring to the real `user.action` stream (Event Service #8 / Stream Processors #9 own production fan-out)
- Using the user vector for retrieval (that is Recall, Component #5)
- A learned (trained) fusion of long/short term — fixed blend until interaction data exists
- Per-user real-time long-term updates (long-term stays a batch job for now)
- Negative-feedback modeling beyond a simple negative action weight

### Deliverables for This Stage
```
user_embedding/
├── README.md
├── requirements.txt
├── .env.example
├── _path.py                  # adds database/ to sys.path, loads .env
├── main.py                   # FastAPI app entry point
├── routers/
│   └── embeddings.py         # GET /users/{id}/embedding, GET /health, dev POST
├── services/
│   ├── math_core.py          # pure-NumPy EMA / aggregation / blend (no I/O)
│   ├── store.py              # Mongo read/write adapter
│   ├── cache.py              # Redis write-through vector cache
│   └── update.py             # apply-interaction + read-query orchestration
├── consumers/
│   └── user_action.py        # Redis Streams XREADGROUP loop + ack/dlq
├── schemas/
│   └── events.py             # UserActionEvent
├── notebooks/
│   └── recompute_longterm.py # Colab-runnable long-term batch path
└── tests/
    └── smoke_test.py
```

### Definition of Done

> Status: **all of steps 4.1-4.10 are implemented and verified against live
> Redis + MongoDB Atlas (2026-06-19).** The math core is unit-tested with no
> infra; the end-to-end smoke test and every infra-gated check below were run
> and pass. Component complete.

- [x] `uvicorn user_embedding.main:app --port 8002` starts the service
- [x] `GET /health` returns `ok` for mongo + redis
- [x] `GET /users/{id}/embedding` returns a unit-norm 384-d query vector for a user with history (`seed_u_000`: dim 384, norm 1.0), and `cold_start=true` with an empty vector for a user with none
- [x] The math core is pure (no I/O), deterministic, and unit-tested with synthetic vectors (`tests/_smoke_math.py`, 7 checks pass)
- [x] `python -m user_embedding.consumers.user_action` joins consumer group `user_embedding` on stream `user.action`
- [x] Applying a synthetic `user.action` event moves `short_term_embedding` toward the interacted video (cosine similarity increases) and persists to both cache and Mongo
- [x] Re-delivering the same event is a no-op (idempotency on `interaction_id`)
- [x] Failed messages land in `user.action.dlq` after the configured retry count, with the failure reason (verified: unknown-user event DLQ'd with reason)
- [x] `long_term_embedding` recompute batch runs over the `interactions` collection and writes a unit-norm 384-d vector (verified with staged positive interactions: dim 384, norm 1.0)
- [x] `python -m user_embedding.tests.smoke_test` passes end-to-end
- [x] No duplication of MongoDB/Redis connection code — uses `database/client.py`
- [x] README documents the HTTP + event contract, the EMA decay / action weights / blend beta, the cache strategy, and cold-start behavior

---

## COMPLETED STAGES

### Component #3: Content Analyzer `[DONE]`

Background worker (not an HTTP service) that consumes `video.uploaded` events from the Upload Service, fetches each video from MinIO, runs multi-modal embedding (CLIP ViT-B/32 frames + Whisper-tiny transcript + MiniLM text), fuses the modalities into a single 384-d `content_embedding`, and writes it back to the `videos` collection so the video becomes searchable by Atlas Vector Search. Full contract, runtime budget, and the fusion strategy: `content_analyzer/README.md`.

Key design points:
- **Reliability:** named consumer group `content_analyzer` on stream `video.uploaded`; XACK on success; idempotency on `video_id` + `analyzer_version`; XAUTOCLAIM-based reclaim of stale pending messages; dead-letter list `video.uploaded.dlq` after `CONTENT_ANALYZER_MAX_RETRIES` (default 3), with unparseable payloads DLQ'd on first sight so they don't burn the retry budget.
- **Fusion:** CLIP gives 512-d, MiniLM gives 384-d, but the Atlas index is fixed at 384-d cosine, so concatenation (896) was not viable. `services/fuse.py` projects visual 512->384 with a fixed-seed (42) Gaussian random matrix (Johnson-Lindenstrauss), L2-normalizes each modality, blends `alpha*visual + (1-alpha)*text` (default alpha=0.6, a mild visual lean), then final L2-normalizes. Changing the seed invalidates every embedding, so `ANALYZER_VERSION` must be bumped if it changes.
- **Write-back:** one atomic aggregation-pipeline `update_one` sets `content_embedding`, `ai_tags.transcript`, `analyzer_version`, `analyzed_at`, and flips `moderation_status` `pending -> approved` while preserving `approved`/`rejected` (re-embedding never resurrects a rejected video). Moderation itself is still a no-op auto-approve stub.
- **Entry point:** `python -m content_analyzer.main` preloads all three models once (fail-fast on missing weights), runs `download -> frames -> visual -> audio+transcript -> text -> fuse -> write_back` per event, graceful shutdown on SIGINT/SIGTERM, `--once` flag for the smoke test.
- **Colab fallback:** `notebooks/batch_embed.py` pull-based batch runner reuses `main.build_handler` so the encoder code path is identical to the streaming worker; auto-detects CUDA; query is `analyzer_version != current OR content_embedding == []` so partial runs compose.

Definition of Done (all met):
- [x] `python -m content_analyzer.main` starts the worker and joins consumer group `content_analyzer` on stream `video.uploaded`
- [x] Models load once at startup; per-video processing reuses them
- [x] Posting a new video via the Upload Service results in a non-empty `content_embedding` on the Mongo doc within ~60s on CPU
- [x] `content_embedding` length matches the Atlas Vector Search index dimension (384)
- [x] `transcript`, `analyzer_version`, and `analyzed_at` are populated on the doc
- [x] `moderation_status` transitions from `pending` to `ready` after successful embedding (schema maps to `approved`)
- [x] Failed messages land in `video.uploaded.dlq` after the configured retry count, with the failure reason
- [x] Re-delivering the same `video.uploaded` event is a no-op (idempotency check on `video_id` + `analyzer_version`)
- [x] `python -m content_analyzer.tests.smoke_test` passes end-to-end and asserts vector-search retrieves the embedded video
- [x] No duplication of MongoDB/Redis/MinIO connection code — uses `database/client.py`
- [x] README documents: ffmpeg requirement, expected CPU runtime per video, the Colab batch fallback, and the chosen fusion strategy

### Component #2: Upload Service `[DONE]`

FastAPI service that accepts uploads, transcodes to mp4 via FFmpeg (with passthrough fallback), stores files in MinIO, writes a `Video` doc to MongoDB, and emits a `video.uploaded` event on Redis Streams. Full contract and run instructions: `upload_service/README.md`.

**Addendum: dev UI (throwaway).** A single embedded HTML page lives at `GET /` on the Upload Service, backed by three helpers in `upload_service/routers/ui.py`: `GET /ui/users` (author dropdown), `GET /ui/videos` (recent library with `moderation_status` + `embedding_dim`), and `GET /ui/videos/{id}/stream` (MinIO proxy with HTTP Range support so the HTML5 `<video>` tag can seek). Lets you upload a real video from the browser, watch the Content Analyzer flip `pending` -> `approved`, then replay it. Not the production UI - that is Component #7 (Next.js). When Component #7 lands, delete `routers/ui.py` and the `app.include_router(ui.router)` line in `main.py`.

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
