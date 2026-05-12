# Database Layer (Component #1)

Foundation component for Mini-TikTok. Provides:

- **MongoDB Atlas (M0 free tier)** — primary store for `users`, `videos`, `interactions`, `experiments`, plus vector search on video embeddings
- **Redis (local Docker)** — cache and Redis Streams message queue
- **MinIO (local Docker)** — S3-compatible object storage for video files

All other components import the Pydantic schemas and the connection factory from this package.

---

## Prerequisites

- Python 3.10+
- Docker Desktop (for Redis + MinIO)
- A free MongoDB Atlas account: https://www.mongodb.com/cloud/atlas

---

## Setup

### 1. Create the MongoDB Atlas cluster

1. Sign up at https://www.mongodb.com/cloud/atlas and create a free **M0** cluster.
2. Under **Database Access**, create a database user with a strong password.
3. Under **Network Access**, allow your IP (or `0.0.0.0/0` for dev only).
4. Click **Connect → Drivers** and copy the SRV connection string.

### 2. Configure local environment

From the project root:

```bash
cp database/.env.example database/.env
```

Edit `database/.env` and paste your Atlas connection string into `MONGO_URI`.

### 3. Start local infrastructure (Redis + MinIO)

From the project root:

```bash
docker compose up -d
```

Verify both containers are healthy:

```bash
docker compose ps
```

MinIO console will be available at http://localhost:9001 (login: `minioadmin` / `minioadmin`).

### 4. Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r database/requirements.txt
```

### 5. Verify connections

```bash
python database/client.py
```

Expected output: three `[OK]` lines (MongoDB, Redis, MinIO).

### 6. Create standard MongoDB indexes

```bash
python database/scripts/create_indexes.py
```

### 7. Create the Atlas Vector Search index (manual UI step — M0 limitation)

The M0 free tier does not support creating vector indexes via the API, so this is a one-time UI step.

1. Open your cluster in the Atlas UI.
2. Go to **Atlas Search → Create Search Index → JSON Editor**.
3. Choose **Vector Search**, database `mini_tiktok`, collection `videos`.
4. Name the index **`video_content_index`** (exact spelling — the code depends on it).
5. Paste the contents of `database/scripts/vector_index_def.json` (only the `fields` array is needed; remove the `_comment_` / `_index_name_` / etc. keys, those are documentation).
6. Click **Create** and wait until status shows **ACTIVE** (usually under a minute).

### 8. Seed fake data

```bash
python database/scripts/seed_data.py
```

This inserts 50 fake users and 200 fake videos with random embeddings so downstream components have something to work against.

### 9. Run the smoke test

```bash
python database/tests/smoke_test.py
```

All checks should pass: MongoDB, Redis, MinIO, indexes present, vector search returns results.

---

## Layout

```
database/
├── README.md                  this file
├── requirements.txt
├── .env.example               template for .env (real .env is gitignored)
├── client.py                  shared MongoDB + Redis + MinIO client factory
├── schemas/                   Pydantic models for all four collections
│   ├── user.py
│   ├── video.py
│   ├── interaction.py
│   └── experiment.py
├── scripts/
│   ├── create_indexes.py      creates all standard MongoDB indexes
│   ├── vector_index_def.json  paste into Atlas UI when creating the vector index
│   ├── seed_data.py           populates 50 users + 200 videos
│   └── test_vector_search.py  ad-hoc vector search experiments
└── tests/
    └── smoke_test.py          end-to-end connectivity + correctness check
```

---

## Contract for other components

Other components depend on this layer through three surfaces:

1. **Pydantic schemas** in `database/schemas/` — import these instead of defining your own.
2. **Connection factory** in `database/client.py` — call `get_mongo()`, `get_redis()`, `get_minio()`.
3. **Index names** — the vector index is `video_content_index`; do not rename without updating all callers.

---

## Free-tier reminders

- M0 has 512 MB total storage. Demo is capped at ~500 videos.
- M0 allows only **one** vector index — it is reserved for `videos.content_embedding`.
- `interactions` should have a 90-day TTL once Stream Processors are wired up.
