# Mini-TikTok Recommendation Engine

> A laptop-friendly, end-to-end video recommendation system inspired by TikTok's For-You feed.
> Built component-by-component with isolated boundaries so each piece can be developed and tested independently.
>
> **Cost target: $0/month. Every tool below is verified free-forever or free-on-laptop.**

> Build progress, current focus, and completed-stage history live in `TODO.md`. This file is the
> stable reference for project rules, architecture, and component boundaries.

---

## SYSTEM RULES

- **No emojis.** Do not write emojis or other pictographic Unicode characters into any file in this project (source code, Markdown, configs, commit messages, comments, log/print output, anywhere). Use plain ASCII tags such as `[OK]`, `[FAIL]`, `[WARN]`, `[TODO]` instead. Reason: the Windows console (cp1252) crashes on emoji output and emojis add visual noise without information. Box-drawing characters (`│ ▼ ► ─ ├ └`) and em dashes (`—`) are allowed because they are not emojis.

---

## PART 1: BASIC ARCHITECTURE

### System Goal
Build a swipe-style short-video app where the recommendation feed learns from user behavior in near real-time. The system must demonstrate:
- Vector similarity search (MongoDB Atlas Vector Search)
- Multi-modal AI for content understanding (CLIP + Whisper)
- A multi-stage recommendation funnel: **Recall → Rank → Re-rank**
- Real-time user embedding updates from interaction events
- Cold-start handling for new videos

### Component Map

| # | Component | Purpose / Function | Tools & Stack | Free? | GPU/CPU Demand |
|---|-----------|--------------------|---------------|-------|----------------|
| 1 | **Database Layer** | Store users, videos, interactions, embeddings; serve vector search | **MongoDB Atlas M0** (512 MB free forever, 1 vector index allowed — sufficient), **Redis** (local Docker), **MinIO** (local Docker, S3-compatible object storage for video files) | Yes, 100% free | None |
| 2 | **Upload Service** | Accept video uploads, transcode, store, emit `video.uploaded` event | FastAPI, FFmpeg, boto3 (talks to MinIO), **Redis Streams** (message queue) | Yes | Very low |
| 3 | **Content Analyzer** | Generate video embeddings from frames + audio + caption | CLIP ViT-B/32 (HuggingFace), **Whisper-tiny** (39M params, CPU-friendly), `all-MiniLM-L6-v2` (22M params), PyTorch | Free weights | HIGH on CPU (~20-40s/video). Mitigation: batch overnight or burst to **Google Colab** (free T4 GPU, 12 hr/day) |
| 4 | **User Embedding Service** | Maintain short-term + long-term user vectors; update from events | FastAPI microservice, NumPy, Redis cache, Redis Streams consumer | Yes | Low |
| 5 | **Recall Service** | Retrieve ~500 candidate videos per user via parallel strategies | MongoDB Atlas Vector Search, Redis, asyncio | Yes (within M0 limits) | Medium |
| 6 | **Ranking Service** | Score candidates with multi-task model predicting watch/like/share/skip | **LightGBM** (CPU-friendly, MIT license) — fast training, near-DNN quality | Yes | Inference fast on CPU. Training also fast (seconds-minutes) |
| 7 | **Re-ranker** | Apply diversity, freshness, cold-start injection, business rules | Pure Python rules engine | Yes | Trivial |
| 8 | **Event Service** | Capture every user action, fan out to consumers | FastAPI, Redis Streams | Yes | Low |
| 9 | **Stream Processors** | Update user embeddings, video stats, feature cache from events | Python Redis Streams consumers | Yes | Low |
| 10 | **Feed API** | Orchestrate Recall → Rank → Re-rank → return feed to client | FastAPI, asyncio | Yes | Low |
| 11 | **Frontend** | Swipe-style video UI, event tracking | Next.js, Tailwind, HTML5 video, **Vercel** hosting | Yes | None |
| 12 | **A/B Testing** | Hash-based bucketing, log variants | Python, MongoDB `experiments` collection | Yes | Trivial |

### Stack Changes from Original Plan

Two swaps were made after the free-tier audit:

| Original Plan | Final Choice | Reason |
|--------------|--------------|--------|
| Cloudflare R2 / S3 for object storage | **MinIO (local Docker)** | R2 requires a credit card even for free tier. MinIO is S3-API-compatible, runs locally, no signup. Production migration to R2/B2 is a one-line config change. |
| Kafka for message queue | **Redis Streams** | Kafka is heavy (needs Zookeeper/KRaft, ~1-2 GB RAM, complex setup). Redis Streams gives the same pub/sub patterns, we already run Redis for caching. Architecture remains "Kafka-ready" — swap is one component (#8). |

### High-Level Data Flow

```
Creator uploads video
        │
        ▼
[Upload Service] ──► [MinIO] + [MongoDB.videos]
        │
        ▼ (Redis Stream: video.uploaded)
[Content Analyzer] ──► writes content_embedding to MongoDB
        │
        ▼
   Video is now searchable

──────────────────────────────────────────────────────────

User opens app
        │
        ▼
[Feed API] ──► [Recall] ──► MongoDB Atlas Vector Search
        │           │
        │           ├─► trending recall
        │           └─► social/category recall
        │
        ▼ (~500 candidates)
[Ranking (LightGBM)] ──► scores each candidate
        │
        ▼ (top 50)
[Re-ranker] ──► diversity + cold-start + business rules
        │
        ▼ (top 10)
Returned to client

──────────────────────────────────────────────────────────

User swipes/likes/skips
        │
        ▼
[Event Service] ──► Redis Stream: user.action
        │
        ├─► [Stream Processor 1] updates user short_term_embedding
        ├─► [Stream Processor 2] updates video.stats
        ├─► [Stream Processor 3] checks cold-start promotion thresholds
        └─► [Stream Processor 4] persists to MongoDB.interactions
```

### Cost & Compute Strategy

- **Local laptop:** runs everything except heavy ML training
- **Google Colab (free T4):** Content Analyzer batch runs + occasional model experiments
- **Kaggle Notebooks (free P100/T4, 30 hrs/week):** alternative GPU
- **MongoDB Atlas M0:** 512 MB DB + Vector Search, free forever
- **Vercel free tier:** frontend hosting (generous hobby plan)
- **Total monthly cost:** **$0**

### Known Free-Tier Constraints (and how we handle them)

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| MongoDB M0: 512 MB storage | Limits total videos + interactions | Cap demo at ~500 videos, use TTL on `interactions` (90 days), drop old data |
| MongoDB M0: 1 vector index only | Can't index multiple embedding fields | Only index `videos.content_embedding` — user embeddings are query vectors, not searched |
| MongoDB M0: vector index must be created via Atlas UI (not API) | Manual step | Document the UI clicks in `database/README.md` |
| Colab: disconnects after ~12 hrs | Long training jobs interrupted | Save checkpoints frequently; train models in <2hr chunks |
| Whisper-tiny: lower accuracy than -base/-large | Some transcripts will be noisy | Acceptable for demo; document tradeoff in README |
| Vercel free: 100 GB bandwidth/month | Plenty for portfolio demo | Monitor; switch to Cloudflare Pages if exceeded |

---

## PART 2: COMPONENT BOUNDARIES

The system is intentionally designed with strong boundaries between components so each can be built and tested in isolation. Each component exposes a clear contract (HTTP API, Redis Stream topic, or DB schema) and can be mocked when working on other parts.

### Isolation Principles
- **Each service is a standalone Python package** with its own `requirements.txt`
- **Communication contracts are explicit:** Pydantic schemas for HTTP, JSON schemas for Redis Stream events
- **Every component can run with mocked dependencies** (e.g., Recall can be tested with a fake MongoDB; Ranking can be tested with synthetic candidates)
- **Shared models live in `database/schemas/`** so changes are coordinated
- **Each component has its own `README.md`** documenting its contract, env vars, and how to run it standalone

---

## Notes for Future Sessions

- Each component will get its own folder under the project root with a similar structure to `database/`
- Build progress, current focus, and the per-stage "Definition of Done" checklists live in `TODO.md` — update there, not here
- Keep PART 1 architecture stable — only update if a fundamental design decision changes (and document the swap in the "Stack Changes" subsection)
- Always preserve the principle: **a component should be developable and testable without the others running**
- If a tool's free tier ever changes, check this doc first; we explicitly chose tools to be free
