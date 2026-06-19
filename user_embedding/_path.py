"""
_path.py — sys.path shim
========================

The User Embedding Service (Component #4) depends on the Database Layer:

    from client import get_mongo, get_redis
    from schemas import User
    from schemas.user import EMBEDDING_DIM

`database/` is a sibling folder (not an installed package), so importing this
module once at process start adds `database/` to sys.path. Any module in
user_embedding can then `import user_embedding._path` before importing from
the Database Layer.

We also load `database/.env` first so MONGO_URI / REDIS_URL are available to
the connection factory, then layer `user_embedding/.env` on top for
per-service settings (stream name, EMA decay, blend beta, retry counts).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# user_embedding/ and database/ are siblings under the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATABASE_DIR = _PROJECT_ROOT / "database"

if str(_DATABASE_DIR) not in sys.path:
    sys.path.insert(0, str(_DATABASE_DIR))

# Load database/.env first so MONGO_URI etc. are defined.
load_dotenv(_DATABASE_DIR / ".env")
# Then layer user_embedding/.env on top (so it can override per-service settings).
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def project_root() -> Path:
    return _PROJECT_ROOT


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)
