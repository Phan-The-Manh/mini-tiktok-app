# Mini-TikTok Recommendation Engine

A laptop-friendly, end-to-end video recommendation system inspired by TikTok's For-You feed. Built component-by-component with strict isolation so each piece can be developed and tested independently.

**Cost target: $0/month.** Every tool in the stack is verified free-forever or runs locally.

> Full architecture, build order, and design decisions live in [`CLAUDE.md`](./CLAUDE.md).

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
| 1 | Database Layer | In progress |
| 2 | Upload Service | Pending |
| 3 | Content Analyzer | Pending |
| 4 | User Embedding Service | Pending |
| 5 | Recall Service | Pending |
| 6 | Feed API | Pending |
| 7 | Frontend | Pending |
| 8 | Event Service | Pending |
| 9 | Stream Processors | Pending |
| 10 | Ranking Service | Pending |
| 11 | Re-ranker | Pending |
| 12 | A/B Testing | Pending |

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

# 3. Configure MongoDB Atlas
cp database/.env.example database/.env
# edit database/.env and paste your Atlas SRV URI into MONGO_URI

# 4. Initialize the database
python database/scripts/create_indexes.py
python database/scripts/seed_data.py
python database/tests/smoke_test.py
```

Detailed setup, including the manual Atlas Vector Search UI step, lives in [`database/README.md`](./database/README.md).

## Project rules

- **No emojis in any file in this repo.** Use plain ASCII tags (`[OK]`, `[FAIL]`, `[WARN]`, `[TODO]`). Reason: the Windows console (cp1252) crashes on emoji output.
- Each component is a standalone package with its own `README.md`, `requirements.txt`, and ability to run with mocked dependencies.
- Shared contracts (Pydantic schemas, Redis Stream event shapes) live in component packages and are imported by callers.

## Repository layout

```
mini-tiktok-app/
├── CLAUDE.md                  full architecture + working notes
├── README.md                  this file
├── docker-compose.yml         Redis + MinIO
└── database/                  Component #1 (current focus)
    ├── README.md
    ├── client.py
    ├── schemas/
    ├── scripts/
    └── tests/
```
