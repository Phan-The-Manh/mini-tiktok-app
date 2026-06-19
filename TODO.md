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

## CURRENT STAGE: Recall Service (Component #5)

The first half of the recommendation funnel. Given a `user_id`, return ~500 candidate videos (each tagged with the recall source that surfaced it) for the Ranking Service to score. Recall optimizes for **high recall over precision** — cast a wide, cheap net now; precision comes later in Rank (#10) and Re-rank (#11).

It runs several **parallel recall strategies** with asyncio and merges their outputs:

- **Vector recall** — fetch the user's blended query vector from the User Embedding Service (#4) and run Atlas `$vectorSearch` against `videos.content_embedding` (384-d cosine, index `video_content_index`). The personalized core.
- **Trending recall** — time-decayed popularity from `videos.stats`. Covers freshness and gives cold-start users something good.
- **Category / affinity recall** — videos in the user's liked categories and from recently-engaged authors, read from the denormalized `User.recent_interactions` / `recently_seen_*` fields.

Candidates are deduped, tagged with their recall source(s), filtered to `moderation_status == approved`, stripped of already-seen videos, capped at ~500, and cached in Redis per user.

> Dependency note: Recall consumes the query vector via the User Embedding Service's documented HTTP contract (`GET /users/{id}/embedding`, port 8002). That call sits behind a small client interface so Recall stays developable in isolation against a mocked provider; the same interface surfaces the `cold_start` flag so Recall can branch.

### Goals (broken down into small, testable steps)

1. **Project skeleton.** Standalone `recall_service/` package with its own `requirements.txt`, `.env.example`, and `_path.py` reusing `database/client.py` and `database/schemas/`.
2. **Query-vector client.** Fetch a user's blended query vector + `cold_start` flag from the User Embedding Service (#4) over HTTP. Behind a `QueryVectorProvider` interface with an HTTP impl and an in-process/mock impl, so strategy tests don't need #4 running. Timeout + fallback to cold-start on failure.
3. **Vector recall strategy.** Atlas `$vectorSearch` over `videos.content_embedding` using the query vector; `numCandidates` / `limit` tunable; returns `(video_id, score)` with `moderation_status == approved` filtered in the pipeline. Skipped for cold-start users.
4. **Trending recall strategy.** Mongo aggregation ranking by a time-decayed engagement score from `videos.stats` (views / likes / completion_rate) and `uploaded_at`. Pure DB, no user vector — the cold-start backbone.
5. **Category / affinity recall strategy.** Pull the user's top liked categories and recently-engaged authors from the denormalized `User` doc; fetch recent approved videos matching them. Degrades to empty cleanly for a brand-new user.
6. **Merge + dedup + exclusions.** Union candidates across strategies, dedup by `video_id` keeping all contributing `recall_source` tags and the max per-strategy score, drop already-seen videos (`recent_interactions.last_50_video_ids` plus an optional Redis seen-set), cap at `RECALL_LIMIT` (default 500).
7. **Parallel orchestration (asyncio).** Run the enabled strategies concurrently with a per-strategy timeout; a slow or failing strategy is logged and skipped rather than failing the whole recall (graceful degradation). Deterministic merge order for stable tests.
8. **Cold-start path.** When the query vector is `cold_start`, skip vector recall and lean on trending + category/demographics so a new user still gets a full candidate set. Document the contract.
9. **Redis result cache.** Cache the per-user candidate list under a short TTL (`RECALL_CACHE_TTL`, default ~60s) so repeated feed pulls within a session don't re-query Atlas. The cache key carries a strategy/version tag so tuning invalidates cleanly.
10. **Service + library API.** `GET /recall/{user_id}` returns the candidate list + per-candidate sources + which strategies ran; `GET /health` checks Mongo + Redis + the #4 dependency. Also expose an importable `recall(user_id) -> list[Candidate]` so the Feed API (#6) can call Recall in-process without HTTP.
11. **README.** Document the contract (request/response, the `Candidate` shape), each strategy and its tunables, the recall cap, the cold-start behavior, the #4 dependency + how to mock it, and the cache strategy.
12. **Smoke test.** End-to-end against seeded data: a warm user returns a non-empty candidate set whose top entries carry the `vector` source; a cold user returns a `trending`-dominated set; dedup holds (no repeated `video_id`); already-seen exclusions are respected; every candidate is `approved`. Cleanup in a `finally`.

### Out of Scope (deferred to later components)
- Scoring / ranking candidates beyond their raw recall score (Ranking, Component #10)
- Diversity, freshness balancing, cold-start injection, business rules (Re-ranker, Component #11)
- A real follow / social graph (use the denormalized `User` fields until an Event-driven graph exists)
- Personalized or per-region trending (global trending for now)
- Orchestrating the full feed (Feed API, Component #6 — it calls Recall)

### Deliverables for This Stage
```
recall_service/
├── README.md
├── requirements.txt
├── .env.example
├── _path.py                  # adds database/ to sys.path, loads .env
├── main.py                   # FastAPI app entry point
├── routers/
│   └── recall.py             # GET /recall/{user_id}, GET /health
├── clients/
│   └── query_vector.py       # QueryVectorProvider (HTTP impl + mock)
├── strategies/
│   ├── vector.py             # Atlas $vectorSearch recall
│   ├── trending.py           # time-decayed popularity recall
│   └── affinity.py           # category / author affinity recall
├── services/
│   ├── merge.py              # dedup + source tagging + exclusions + cap
│   ├── orchestrator.py       # asyncio parallel run + graceful degradation
│   └── cache.py              # Redis per-user candidate cache
├── schemas/
│   └── candidate.py          # Candidate + RecallResponse models
└── tests/
    └── smoke_test.py
```

### Definition of Done

> Status: not started. Build the strategies behind the mock query-vector
> provider first (no #4 process needed), then wire the HTTP client and run the
> infra-gated checks once Docker + Atlas are up and the User Embedding Service
> is running on :8002.

- [ ] `uvicorn recall_service.main:app --port 8003` starts the service
- [ ] `GET /health` returns `ok` for mongo + redis and reports the #4 dependency reachability
- [ ] `GET /recall/{user_id}` returns up to ~500 deduped candidates, each tagged with its recall source(s)
- [ ] A warm user's candidates include vector-recall hits near the top; a cold-start user gets a trending-backed set with no vector recall
- [ ] Already-seen videos are excluded and every returned candidate is `moderation_status == approved`
- [ ] A failing / slow individual strategy degrades gracefully (logged, skipped) without failing the whole recall
- [ ] The candidate list is cached in Redis under a short TTL and a strategy-version tag
- [ ] Strategies are unit-testable behind a mocked `QueryVectorProvider` (no #4 process required)
- [ ] `python -m recall_service.tests.smoke_test` passes end-to-end against seeded data
- [ ] No duplication of MongoDB/Redis connection code — uses `database/client.py`
- [ ] README documents the contract, the strategies + tunables, cold-start behavior, the #4 dependency + mocking, and the cache strategy

---

## COMPLETED STAGES

### Component #4: User Embedding Service `[DONE]`

FastAPI microservice plus a shared pure-NumPy embedding-math library that maintains two 384-d vectors per user in the same cosine space as `videos.content_embedding`, and serves a single blended query vector to Recall (#5) with cold-start handling. Full contract, the EMA / blend parameters, and the cache strategy: `user_embedding/README.md`.

Key design points:
- **Math core (no I/O):** signed action weights (like / share +1.0, follow +0.9, comment +0.7, skip -0.5, not_interested / report -1.0, graded watch `-0.3 + 1.3*watch_pct`), short-term EMA `s' = normalize(decay*s + (1-decay)*weight*v)` (decay 0.9), long-term weighted-mean aggregation, and the query blend `q = normalize(beta*long + (1-beta)*short)` (beta 0.5). Everything L2-normalized; the zero vector is the cold-start sentinel. Unit-tested with no infra (`tests/_smoke_math.py`, 7 checks).
- **Two update paths:** short-term drifts live per interaction (the `user.action` consumer, write-through to Mongo + Redis); long-term is rebuilt in batch (`notebooks/recompute_longterm.py`) from a window of positive interactions. Both reuse the same math core.
- **Reliability:** consumer group `user_embedding` on stream `user.action`; idempotency on `interaction_id` (marked only after a successful apply, because EMA is not naturally idempotent); XAUTOCLAIM reclaim of stale pending messages; `user.action.dlq` after `USER_EMBEDDING_MAX_RETRIES` (default 3). Permanent errors (unknown user / deleted video) DLQ immediately; `VideoNotEmbedded` is retried (the Content Analyzer may catch up).
- **Read path:** `GET /users/{id}/embedding` returns the blended query vector + metadata (`dim`, `cold_start`, `has_long_term`, `has_short_term`); cold-start returns an empty vector so Recall can branch to trending / demographics. Dev-only `POST /users/{id}/interactions` shares the same apply path as the consumer.
- **Sequencing:** the live `user.action` stream is owned by the Event Service (#8), built later; #4 is exercised with synthetic events until then.

Definition of Done (all met, verified against live Redis + MongoDB Atlas 2026-06-19):
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
