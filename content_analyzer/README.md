# Content Analyzer (Component #3)

Background worker that consumes `video.uploaded` events emitted by the Upload
Service, embeds each video into a 384-d vector via CLIP + Whisper-tiny +
MiniLM, fuses the modalities, and writes the result back to MongoDB so the
video becomes searchable by Atlas Vector Search.

This is **not** an HTTP service. There is no public API surface — only a
Redis Streams consumer and a Mongo write-back.

---

## Contract

### Input — Redis Streams

Stream: `video.uploaded` (override with `CONTENT_ANALYZER_STREAM`).
Consumer group: `content_analyzer` (override with `CONTENT_ANALYZER_GROUP`).
Field shape: see `content_analyzer/schemas/events.py:VideoUploadedEvent`.

Failed messages land in the list `video.uploaded.dlq` after
`CONTENT_ANALYZER_MAX_RETRIES` (default 3) deliveries, each as a JSON
record carrying the original fields and the failure reason.

### Output — MongoDB

For every successfully processed event, `videos.{video_id}` is updated in
one atomic aggregation-pipeline `update_one`:

| Field                   | Source                                          |
|-------------------------|-------------------------------------------------|
| `content_embedding`     | 384-d unit vector from `services/fuse.fuse()`   |
| `ai_tags.transcript`    | Whisper-tiny output (empty for silent videos)   |
| `analyzer_version`      | `ANALYZER_VERSION` env (default `v1`)           |
| `analyzed_at`           | UTC `datetime` at write time                    |
| `moderation_status`     | `pending` -> `approved`; other values preserved |

Re-embedding a video that is `rejected` refreshes the embedding but
preserves the rejection — moderation never gets resurrected by an
analyzer run.

---

## Setup

### 1. Prereqs

- Database Layer (Component #1) and Upload Service (Component #2) up.
- **ffmpeg** + **ffprobe** on `PATH` (Windows: <https://www.gyan.dev/ffmpeg/builds/>).
  Set `FFMPEG_BIN` if you need a non-default path; `ffprobe` is derived
  alongside it.
- Python 3.10+ in the project venv. The model weights (~300 MB total) are
  pulled on first run and cached under `~/.cache/`.

### 2. Install

```powershell
pip install -r content_analyzer/requirements.txt
```

CPU-only torch is fine for the laptop worker. The Colab fallback (see
below) gets GPU torch from the Colab base image.

### 3. Configure

Reuses `database/.env`. Per-worker overrides go in
`content_analyzer/.env` (loaded on top with `override=True`):

```
ANALYZER_VERSION=v1
CONTENT_ANALYZER_CONSUMER=worker-1
CONTENT_ANALYZER_MAX_RETRIES=3
CONTENT_ANALYZER_BLOCK_MS=5000
CONTENT_ANALYZER_CLAIM_IDLE_MS=60000
ANALYZER_FRAME_COUNT=8
ANALYZER_FUSION_ALPHA=0.6
ANALYZER_MIN_AUDIO_SECONDS=0.5
ANALYZER_SILENCE_DBFS=-45.0
WHISPER_MODEL=tiny
CLIP_MODEL=openai/clip-vit-base-patch32
TEXT_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 4. Run the worker

```powershell
python -m content_analyzer.main
```

Or one pass for tests:

```powershell
python -m content_analyzer.main --once
```

Ctrl+C exits cleanly at the next iteration boundary (worst case ~5 s,
the `XREADGROUP` block time).

---

## Per-video runtime (CPU)

Measured informally on a mid-range laptop (no GPU), 8 frames, ~15-30 s
clips:

| Stage                              | Time         |
|------------------------------------|--------------|
| Download from MinIO                | <0.5 s (LAN) |
| ffmpeg frame sampling (8 frames)   | 1-3 s        |
| CLIP visual encode (8 frames)      | 0.3-0.8 s    |
| ffmpeg audio extract + volumedetect| 1-2 s        |
| Whisper-tiny transcribe            | 5-30 s       |
| MiniLM text encode                 | <0.1 s       |
| Fusion + Mongo write-back          | <0.1 s       |
| **Total**                          | **~10-40 s** |

Whisper dominates and scales with audio length. For backfills of many
videos, use the Colab fallback below.

---

## Fusion strategy

CLIP gives a 512-d visual vector; MiniLM gives a 384-d text vector. The
Atlas index (`database/scripts/vector_index_def.json`) is fixed at
**384-d cosine**, so concatenation (512 + 384 = 896) was not viable.

`services/fuse.py` does:

1. Project visual 512 -> 384 with a fixed-seed (`42`) Gaussian random
   matrix (Johnson-Lindenstrauss; preserves distances approximately,
   no training data needed).
2. L2-normalize each modality independently.
3. Weighted sum: `alpha * visual + (1 - alpha) * text`. Default
   `alpha = 0.6` (`ANALYZER_FUSION_ALPHA`) — a mild visual lean because
   CLIP frame embeddings are more robust than Whisper-tiny transcripts
   on short, noisy, or non-English clips.
4. Final L2-normalize so downstream dot products behave.

Changing the projection seed invalidates every embedding ever written —
bump `ANALYZER_VERSION` if you do.

---

## Colab / Kaggle fallback (batch backfill)

`content_analyzer/notebooks/batch_embed.py` runs the exact same encoder
code path against the same Atlas cluster, but pulls work from Mongo
instead of consuming the Redis stream. Suitable for cold backfills on a
free Colab T4 / Kaggle P100.

In a Colab cell:

```
!git clone https://github.com/Phan-The-Manh/mini-tiktok-app.git
%cd mini-tiktok-app
!pip install -q -r content_analyzer/requirements.txt

import os
os.environ["MONGO_URI"]         = "mongodb+srv://..."
os.environ["MINIO_ENDPOINT"]    = "<public-host>"
os.environ["MINIO_ACCESS_KEY"]  = "..."
os.environ["MINIO_SECRET_KEY"]  = "..."
os.environ["MINIO_SECURE"]      = "true"
os.environ["ANALYZER_VERSION"]  = "v1"

!python -m content_analyzer.notebooks.batch_embed --limit 100 --device cuda
```

Object-storage caveat: local MinIO at `localhost:9000` is not reachable
from Colab. Either expose it with `ngrok http 9000` (and set
`MINIO_ENDPOINT` to the tunnel host) or migrate the bucket to a public
S3-compatible store (R2 / B2 / S3). `database/client.py` switches with
env vars only — no code change.

Flags:

| Flag                  | Default | Effect                                          |
|-----------------------|---------|-------------------------------------------------|
| `--limit N`           | 100     | Cap docs processed in this run                  |
| `--analyzer-version V`| env     | Override `ANALYZER_VERSION`                     |
| `--device cpu/cuda`   | auto    | Force device; `auto` picks cuda when available  |
| `--dry-run`           | off     | Print video_ids that would be processed, exit   |

The query is `analyzer_version != current OR content_embedding == []`,
so partial runs compose — a re-invoke after pre-emption picks up where
the last one stopped.

---

## Layout

```
content_analyzer/
|-- README.md
|-- requirements.txt
|-- _path.py                      sys.path + .env shim
|-- main.py                       worker entry point (--once flag for tests)
|-- consumers/
|   `-- video_uploaded.py         Redis Streams XREADGROUP + retry/DLQ
|-- services/
|   |-- downloader.py             MinIO object -> local temp file
|   |-- frames.py                 ffmpeg evenly-spaced frame sampling
|   |-- visual.py                 CLIP ViT-B/32 (lazy-loaded)
|   |-- audio.py                  ffmpeg wav extract + silence gate + Whisper
|   |-- text.py                   MiniLM caption+transcript encoder
|   |-- fuse.py                   modal fusion -> content_embedding
|   `-- writer.py                 MongoDB write-back
|-- schemas/
|   `-- events.py                 VideoUploadedEvent (consumer side)
|-- notebooks/
|   `-- batch_embed.py            Colab/Kaggle pull-based batch path
`-- tests/
    `-- _smoke_*.py               per-module smoke checks
```
