# Upload Service (Component #2)

FastAPI service that accepts video uploads, transcodes them to a web-friendly
mp4, stores files in MinIO, writes a `Video` document to MongoDB, and emits a
`video.uploaded` event on Redis Streams for downstream consumers (the
Content Analyzer in Component #3).

---

## Contract

### HTTP

| Method | Path                  | Body / Form                                                                 | Returns                           |
|--------|-----------------------|-----------------------------------------------------------------------------|-----------------------------------|
| POST   | `/videos`             | multipart: `file`, `author_id`, `caption`, `hashtags`, `category` (optional)| `201 UploadResponse`              |
| GET    | `/videos/{video_id}`  | -                                                                           | `200 VideoOut` or `404`           |
| GET    | `/health`             | -                                                                           | `200 {mongo, redis, minio, ffmpeg}` |

OpenAPI docs are at `http://localhost:8001/docs` when the service is running.

### Redis Streams

Stream key: `video.uploaded` (override with `UPLOAD_EVENT_STREAM`).

Each entry's fields:

```
video_id          str
author_id         str
url               str        # MinIO URL of the (transcoded) video
thumbnail_url     str        # may be absent in passthrough mode
duration_seconds str         # float as string
uploaded_at      str        # ISO 8601
```

Consumers should treat all values as strings (Redis Streams stores them that way)
and parse as needed.

### MongoDB

Writes one document to `videos`. The schema is `database/schemas/video.py:Video`.
On insert:

- `moderation_status = "pending"` (Content Analyzer or a future moderation step flips this)
- `distribution_stage = "test_pool_1"` (cold-start funnel entry point)
- `content_embedding = []` (filled by Content Analyzer)

---

## Setup

### 1. Prereqs

- Database Layer (Component #1) configured and running (`database/.env` filled in,
  `docker compose up -d` for Redis + MinIO, seed data inserted).
- Python 3.10+ and the same venv used for the Database Layer.
- **ffmpeg** on `PATH` (Windows: <https://www.gyan.dev/ffmpeg/builds/>, add the
  `bin/` folder to `PATH`). Without ffmpeg the service still runs in passthrough
  mode — uploads are stored as-is and thumbnails are skipped.

### 2. Install deps

```powershell
pip install -r upload_service/requirements.txt
```

### 3. Configure

The Upload Service reads `database/.env` first, then `upload_service/.env`
(if present) for overrides. Most users only need the database `.env`. To
override per-service settings, copy the template:

```powershell
Copy-Item upload_service\.env.example upload_service\.env
```

### 4. Run

From the project root:

```powershell
uvicorn upload_service.main:app --reload --port 8001
```

Visit <http://localhost:8001/docs> for the interactive OpenAPI explorer.

### 5. Smoke test

In a second terminal (with the service running, `docker compose up -d` running,
and database seed data loaded):

```powershell
python -m upload_service.tests.smoke_test
```

The script generates a 2-second blank mp4 via ffmpeg, posts it to `/videos`,
then verifies the MinIO object and Redis stream entry exist.

---

## Layout

```
upload_service/
  README.md                this file
  requirements.txt
  .env.example
  _path.py                 adds database/ to sys.path and loads .env files
  main.py                  FastAPI app + startup bucket bootstrap
  routers/
    videos.py              POST /videos, GET /videos/{id}, GET /health
  services/
    storage.py             MinIO uploads
    transcoder.py          ffmpeg wrapper (probe + normalize + thumbnail)
    events.py              Redis Streams publisher
  schemas/
    api.py                 HTTP response models
    events.py              VideoUploadedEvent (stream payload)
  tests/
    smoke_test.py          end-to-end check against a running service
```

---

## Design notes

- **Sharing with the Database Layer.** This service does not duplicate the
  MongoDB / Redis / MinIO clients. `_path.py` adds `database/` to `sys.path`
  so we can `from client import get_mongo, get_redis, get_minio` and
  `from schemas import Video, ...`. Pydantic + connection settings stay in
  exactly one place.

- **Library choice (MinIO vs. boto3).** `CLAUDE.md` lists `boto3` for this
  component, but the Database Layer already wires the `minio` Python client
  (also S3-compatible). To avoid two parallel client implementations we reuse
  the existing factory. Swapping to `boto3` is a `client.py`-only change if
  we ever migrate to AWS S3 or Cloudflare R2.

- **Transcoding is optional.** ffmpeg is the right tool but it's a heavy
  dependency on a dev laptop. If ffmpeg is missing or `ENABLE_TRANSCODE=false`,
  the original file is uploaded unchanged and the thumbnail step is skipped.
  The video is still playable — just larger.

- **Event publish is best-effort.** If Redis is down when we finish writing
  to Mongo and MinIO, we log a `[WARN]` and still return `201` to the client.
  The upload itself succeeded; the worst case is the Content Analyzer never
  picks the video up. A re-emit script (TBD) can scan for videos with empty
  `content_embedding` and re-publish them.

- **No auth yet.** The author_id field is trusted and only checked against
  the `users` collection. Real authentication will land alongside the
  frontend integration (per CLAUDE.md PART 2).
