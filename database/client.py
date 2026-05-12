"""
client.py — Connection Factory
==============================

Single source of truth for connecting to MongoDB Atlas, Redis, and MinIO.

Every other script in this project should import from here:

    from client import get_mongo, get_redis, get_minio

    db = get_mongo()
    db.users.find_one({"user_id": "u_123"})

Why centralize?
- One place to change connection settings
- Connections are cached (singleton): we connect once, reuse everywhere
- Reads .env automatically so no one has to deal with env vars manually
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
import redis
from minio import Minio


# --- Load .env file ---
# We look for .env in the same folder as this file (database/.env).
# This makes the script work regardless of where you run it from.
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)


# --- Singleton cache ---
# We store created connections here so calling get_mongo() multiple times
# returns the same connection (instead of opening a new one every call).
_mongo_client: Optional[MongoClient] = None
_redis_client: Optional[redis.Redis] = None
_minio_client: Optional[Minio] = None


def get_mongo() -> Database:
    """
    Returns a MongoDB Database object connected to Atlas.

    Usage:
        db = get_mongo()
        db.users.insert_one({"user_id": "u_123", "username": "alice"})
        db.videos.find_one({"video_id": "v_456"})
    """
    global _mongo_client

    if _mongo_client is None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise RuntimeError("MONGO_URI is not set in .env")

        # serverSelectionTimeoutMS: fail fast (5s) if can't reach Atlas
        # instead of hanging for 30 seconds (the default).
        _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)

    db_name = os.getenv("MONGO_DB_NAME", "mini_tiktok")
    return _mongo_client[db_name]


def get_redis() -> redis.Redis:
    """
    Returns a Redis client (used for both caching and Redis Streams).

    Usage:
        r = get_redis()
        r.set("hello", "world")
        r.get("hello")  # returns b'world' (bytes)
    """
    global _redis_client

    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # decode_responses=True makes get() return str instead of bytes.
        # More convenient for human-readable values; for binary data
        # (like serialized embeddings) we'd set this to False.
        _redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

    return _redis_client


def get_minio() -> Minio:
    """
    Returns a MinIO client for S3-compatible object storage.

    Usage:
        client = get_minio()
        client.fput_object("videos", "video1.mp4", "/local/path/video1.mp4")
        client.fget_object("videos", "video1.mp4", "/download/path.mp4")

    Note: Same code works against AWS S3 or Cloudflare R2 in production
    by changing only the endpoint and credentials in .env.
    """
    global _minio_client

    if _minio_client is None:
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        # secure=True for HTTPS (production), False for HTTP (local dev)
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

        _minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    return _minio_client


def get_minio_bucket_name() -> str:
    """Returns the bucket name configured in .env (defaults to 'videos')."""
    return os.getenv("MINIO_BUCKET", "videos")


# --- Quick connectivity check ---
# Running this file directly (`python client.py`) will test all 3 connections.
# This is the fastest way to verify your .env is correct before writing more code.
if __name__ == "__main__":
    print("Testing connections...\n")

    # Test MongoDB
    try:
        db = get_mongo()
        # ping is the standard Mongo health check
        db.command("ping")
        print(f"[OK] MongoDB connected — database: {db.name}")
    except Exception as e:
        print(f"[FAIL] MongoDB failed: {e}")

    # Test Redis
    try:
        r = get_redis()
        pong = r.ping()
        print(f"[OK] Redis connected — ping returned: {pong}")
    except Exception as e:
        print(f"[FAIL] Redis failed: {e}")

    # Test MinIO
    try:
        m = get_minio()
        # list_buckets is a lightweight operation that requires auth
        buckets = m.list_buckets()
        print(f"[OK] MinIO connected — found {len(buckets)} bucket(s)")
    except Exception as e:
        print(f"[FAIL] MinIO failed: {e}")