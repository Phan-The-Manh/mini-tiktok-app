"""
_path.py — sys.path shim
========================

The Content Analyzer is a background worker (Component #3). It depends on
two things from the Database Layer:

    from client import get_mongo, get_redis, get_minio, get_minio_bucket_name
    from schemas import Video, ModerationStatus

`database/` is a sibling folder (not an installed package), so importing this
module once at process start adds `database/` to sys.path. Any module in
content_analyzer can then `import _path` before importing from the Database
Layer.

We also load `database/.env` here so MONGO_URI / REDIS_URL / MINIO_* are
available to the connection factory without each service maintaining its own
copy of the credentials. `content_analyzer/.env` is layered on top for
per-service settings (stream name, model versions, retry counts, ...).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# content_analyzer/ and database/ are siblings under the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATABASE_DIR = _PROJECT_ROOT / "database"

if str(_DATABASE_DIR) not in sys.path:
    sys.path.insert(0, str(_DATABASE_DIR))

# Load database/.env first so MONGO_URI etc. are defined.
load_dotenv(_DATABASE_DIR / ".env")
# Then layer content_analyzer/.env on top (so it can override per-service settings).
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def project_root() -> Path:
    return _PROJECT_ROOT


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)
