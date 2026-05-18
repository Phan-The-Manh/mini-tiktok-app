# Mini-TikTok Recommendation Engine

A laptop-friendly, end-to-end video recommendation system inspired by TikTok's For-You feed. Built component-by-component with strict isolation so each piece can be developed and tested independently.

**Cost target: $0/month.** Every tool in the stack is verified free-forever or runs locally.

> Architecture, rules, and component boundaries live in [`CLAUDE.md`](./CLAUDE.md). Live build progress and the current-stage breakdown live in [`TODO.md`](./TODO.md).

---

## What this demonstrates

- Vector similarity search on MongoDB Atlas
- Multi-modal content understanding (CLIP + Whisper)
- A multi-stage recommendation funnel: **Recall → Rank → Re-rank**
- Real-time user-embedding updates from interaction events
- Cold-start handling for new videos

## Stack at a glance

| Layer | Tool |
|---|---|
| Vector DB | MongoDB Atlas M0 (free) |
| Cache + message queue | Redis (local Docker) |
| Object storage | MinIO (local Docker, S3-compatible) |
| Backend services | FastAPI |
| ML | CLIP ViT-B/32, Whisper-tiny, all-MiniLM-L6-v2, LightGBM |
| Frontend | Next.js + Tailwind (Vercel) |
| GPU bursts | Google Colab / Kaggle (free tiers) |

## Build status

| # | Component | Status |
|---|---|---|
| 1 | Database Layer | `[DONE]` |
| 2 | Upload Service | `[DONE]` |
| 3 | Content Analyzer | **Current focus** |
| 4 | User Embedding Service | Pending |
| 5 | Recall Service | Pending |
| 6 | Feed API | Pending |
| 7 | Frontend | Pending |
| 8 | Event Service | Pending |
| 9 | Stream Processors | Pending |
| 10 | Ranking Service | Pending |
| 11 | Re-ranker | Pending |
| 12 | A/B Testing | Pending |

See [`TODO.md`](./TODO.md) for the Content Analyzer breakdown and per-component "Definition of Done" checklists.

## Quickstart

```bash
git clone https://github.com/Phan-The-Manh/mini-tiktok-app.git
cd mini-tiktok-app

# 1. Local infra (Redis + MinIO)
docker compose up -d

# 2. Python env
python -m venv venv
venv\Scripts\Activate.ps1            # Windows PowerShell
pip install -r database/requirements.txt
pip install -r upload_service/requirements.txt

# 3. Configure MongoDB Atlas
cp database/.env.example database/.env
# edit database/.env and paste your Atlas SRV URI into MONGO_URI

# 4. Initialize the database
python database/scripts/create_indexes.py
python database/scripts/seed_data.py
python database/tests/smoke_test.py

# 5. Run the Upload Service (in a second terminal)
uvicorn upload_service.main:app --reload --port 8001
# OpenAPI docs:  http://localhost:8001/docs
# Smoke test:    python -m upload_service.tests.smoke_test
```

Per-component setup notes:
- Database Layer (incl. the manual Atlas Vector Search UI step): [`database/README.md`](./database/README.md)
- Upload Service (incl. the ffmpeg PATH requirement and passthrough fallback): [`upload_service/README.md`](./upload_service/README.md)

## Project rules

- **No emojis in any file in this repo.** Use plain ASCII tags (`[OK]`, `[FAIL]`, `[WARN]`, `[TODO]`). Reason: the Windows console (cp1252) crashes on emoji output.
- Each component is a standalone package with its own `README.md`, `requirements.txt`, and ability to run with mocked dependencies.
- Shared contracts (Pydantic schemas, Redis Stream event shapes) live in component packages and are imported by callers.

## Repository layout

```
mini-tiktok-app/
├── CLAUDE.md                  rules, architecture, component boundaries
├── TODO.md                    build progress, current focus, completed stages
├── README.md                  this file
├── docker-compose.yml         Redis + MinIO
├── database/                  Component #1 [DONE]
│   ├── README.md
│   ├── client.py              shared MongoDB + Redis + MinIO factory
│   ├── schemas/               Pydantic models (User, Video, Interaction, Experiment)
│   ├── scripts/               create_indexes, seed_data, vector_index_def
│   └── tests/                 smoke_test
└── upload_service/            Component #2 [DONE]
    ├── README.md
    ├── main.py                FastAPI app
    ├── routers/videos.py      POST /videos, GET /videos/{id}, GET /health
    ├── services/              storage (MinIO), transcoder (ffmpeg), events (Redis Streams)
    ├── schemas/               api + event Pydantic models
    └── tests/smoke_test.py
```
