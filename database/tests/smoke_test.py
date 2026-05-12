"""
smoke_test.py
=============

End-to-end verification that the database component is healthy.

Usage (from D:\\mini-tiktok-app\\database, with venv active):
    python -m tests.smoke_test

What this verifies:
    1. MongoDB Atlas is reachable
    2. Redis (Docker) is reachable
    3. MinIO (Docker) is reachable
    4. All required indexes exist
    5. Seed data is present
    6. Vector search works end-to-end
    7. Redis Streams can be written
    8. MinIO bucket operations work

Exit code 0 if everything passes, 1 if anything fails.
"""

import sys
from datetime import datetime
from io import BytesIO

from client import get_minio, get_minio_bucket_name, get_mongo, get_redis
from schemas import EMBEDDING_DIM


# --- Helpers for pretty output ---

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def section(title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}━━━ {title} ━━━{Colors.END}")


def passed(msg: str):
    print(f"  {Colors.GREEN}✅ PASS{Colors.END}  {msg}")


def failed(msg: str, error: str = ""):
    print(f"  {Colors.RED}❌ FAIL{Colors.END}  {msg}")
    if error:
        print(f"          {Colors.RED}{error}{Colors.END}")


def warn(msg: str):
    print(f"  {Colors.YELLOW}⚠️  WARN{Colors.END}  {msg}")


# --- Individual checks ---
# Each returns True on success, False on failure.


def check_mongo_connection() -> bool:
    section("1. MongoDB Atlas connection")
    try:
        db = get_mongo()
        result = db.command("ping")
        if result.get("ok") == 1.0:
            passed(f"Connected to database '{db.name}'")
            return True
        failed("Ping returned unexpected response", str(result))
        return False
    except Exception as e:
        failed("Could not connect to MongoDB", str(e))
        print("\n          Common causes:")
        print("          - MONGO_URI in .env is wrong")
        print("          - Atlas Network Access doesn't include your IP (or 0.0.0.0/0)")
        print("          - Password in URI has unencoded special characters")
        return False


def check_redis_connection() -> bool:
    section("2. Redis connection")
    try:
        r = get_redis()
        pong = r.ping()
        if pong:
            passed("Redis responded to PING")
            return True
        failed("Redis ping returned False")
        return False
    except Exception as e:
        failed("Could not connect to Redis", str(e))
        print("\n          Common causes:")
        print("          - Docker containers not running. Run: docker compose ps")
        print("          - Redis container unhealthy. Run: docker compose logs redis")
        return False


def check_minio_connection() -> bool:
    section("3. MinIO connection")
    try:
        m = get_minio()
        buckets = m.list_buckets()
        passed(f"Connected to MinIO ({len(buckets)} existing bucket(s))")
        return True
    except Exception as e:
        failed("Could not connect to MinIO", str(e))
        print("\n          Common causes:")
        print("          - Docker containers not running. Run: docker compose ps")
        print("          - MinIO credentials in .env don't match docker-compose.yml")
        return False


def check_indexes() -> bool:
    """Verify every required index exists."""
    section("4. Required indexes")

    db = get_mongo()
    required = {
        "users": {"user_id_unique"},
        "videos": {"video_id_unique", "author_recent", "cold_start_review"},
        "interactions": {"user_history", "video_engagement", "interactions_ttl"},
        "experiments": {"experiment_id_unique", "active_experiments"},
    }

    all_ok = True
    for coll_name, expected in required.items():
        existing_names = {idx["name"] for idx in db[coll_name].list_indexes()}
        missing = expected - existing_names
        if missing:
            failed(f"Collection '{coll_name}' missing indexes: {missing}")
            print(f"          Fix: run `python -m scripts.create_indexes`")
            all_ok = False
        else:
            passed(f"Collection '{coll_name}' has all required indexes")

    return all_ok


def check_seed_data() -> bool:
    section("5. Seed data presence")
    db = get_mongo()

    user_count = db.users.count_documents({"user_id": {"$regex": "^seed_"}})
    video_count = db.videos.count_documents({"video_id": {"$regex": "^seed_"}})

    if user_count == 0 and video_count == 0:
        failed("No seed data found")
        print("          Fix: run `python -m scripts.seed_data`")
        return False

    if user_count < 10 or video_count < 50:
        warn(f"Less seed data than expected: {user_count} users, {video_count} videos")
        return True  # not a hard failure

    passed(f"Found {user_count} seed users, {video_count} seed videos")
    return True


def check_vector_search() -> bool:
    """The critical test: does Atlas Vector Search actually return results?"""
    section("6. Vector search end-to-end")
    db = get_mongo()

    # Grab any seed video — we'll use its embedding as a query vector.
    sample = db.videos.find_one(
        {"content_embedding": {"$exists": True, "$ne": []}},
        {"video_id": 1, "content_embedding": 1},
    )
    if not sample:
        failed("No videos with embeddings found in DB")
        print("          Fix: run `python -m scripts.seed_data` first")
        return False

    if len(sample["content_embedding"]) != EMBEDDING_DIM:
        failed(
            f"Embedding dim mismatch: stored={len(sample['content_embedding'])} "
            f"expected={EMBEDDING_DIM}"
        )
        return False

    # Run a $vectorSearch using that video's embedding as the query.
    # We expect at least the video itself back (cosine sim to itself = 1.0).
    pipeline = [
        {
            "$vectorSearch": {
                "index": "video_content_index",
                "path": "content_embedding",
                "queryVector": sample["content_embedding"],
                "numCandidates": 50,
                "limit": 5,
            }
        },
        {
            "$project": {
                "_id": 0,
                "video_id": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        results = list(db.videos.aggregate(pipeline))
    except Exception as e:
        failed("Vector search query failed", str(e))
        print("          Common causes:")
        print("          - Vector index `video_content_index` not yet Active in Atlas UI")
        print("          - Index name mismatch (must be exactly 'video_content_index')")
        print("          - numDimensions in index def doesn't match 384")
        return False

    if not results:
        failed("Vector search returned 0 results")
        return False

    # Sanity check: the top result should be the video we used as the query
    # (since cosine similarity of a vector with itself = 1.0).
    top = results[0]
    if top["video_id"] != sample["video_id"]:
        warn(
            f"Top result {top['video_id']} is not the query video {sample['video_id']} — "
            f"unusual but not a hard failure"
        )
    passed(f"Vector search returned {len(results)} results")
    print(f"          Top match: {top['video_id']} (score: {top['score']:.4f})")
    return True


def check_redis_streams() -> bool:
    """
    Future components use Redis Streams as the message queue
    (alternative to Kafka). Verify we can write and read a stream.
    """
    section("7. Redis Streams")
    r = get_redis()
    stream_name = "_smoke_test_stream"

    try:
        # Write a test message
        msg_id = r.xadd(stream_name, {"event": "smoke_test", "ts": datetime.utcnow().isoformat()})

        # Read it back
        messages = r.xrange(stream_name, count=1)
        if not messages:
            failed("Wrote to stream but couldn't read back")
            return False

        # Clean up
        r.delete(stream_name)
        passed(f"Wrote and read message (id: {msg_id})")
        return True
    except Exception as e:
        failed("Redis Streams operation failed", str(e))
        return False


def check_minio_bucket() -> bool:
    """
    The Upload Service (Component #2) will store videos in a MinIO bucket.
    Verify we can create one and upload/delete a tiny file.
    """
    section("8. MinIO bucket operations")
    m = get_minio()
    bucket = get_minio_bucket_name()

    try:
        # Create bucket if missing (idempotent)
        if not m.bucket_exists(bucket):
            m.make_bucket(bucket)
            print(f"          Created bucket '{bucket}'")
        else:
            print(f"          Bucket '{bucket}' already exists")

        # Upload a tiny test object
        test_data = b"smoke_test"
        m.put_object(
            bucket_name=bucket,
            object_name="_smoke_test.txt",
            data=BytesIO(test_data),
            length=len(test_data),
            content_type="text/plain",
        )

        # Verify we can stat it
        stat = m.stat_object(bucket, "_smoke_test.txt")
        if stat.size != len(test_data):
            failed("Uploaded size mismatch")
            return False

        # Clean up
        m.remove_object(bucket, "_smoke_test.txt")
        passed(f"Bucket '{bucket}' is writable and readable")
        return True
    except Exception as e:
        failed("MinIO bucket operation failed", str(e))
        return False


# --- Main runner ---


def main():
    print(f"\n{Colors.BOLD}{'═' * 60}")
    print(f"  🩺 Mini-TikTok — Database Component Smoke Test")
    print(f"{'═' * 60}{Colors.END}")

    checks = [
        check_mongo_connection,
        check_redis_connection,
        check_minio_connection,
        check_indexes,
        check_seed_data,
        check_vector_search,
        check_redis_streams,
        check_minio_bucket,
    ]

    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            failed(f"Unexpected error in {check.__name__}", str(e))
            results.append(False)

    # --- Summary ---
    total = len(results)
    passed_count = sum(results)
    print(f"\n{Colors.BOLD}{'═' * 60}")
    if all(results):
        print(f"  {Colors.GREEN}✅ ALL CHECKS PASSED ({passed_count}/{total}){Colors.END}")
        print(f"  Database component is ready. Move on to Component #2.")
    else:
        print(f"  {Colors.RED}❌ {total - passed_count} CHECK(S) FAILED ({passed_count}/{total} passed){Colors.END}")
        print(f"  See errors above for fix instructions.")
    print(f"{Colors.BOLD}{'═' * 60}{Colors.END}\n")

    # Exit code: 0 if all pass, 1 if anything fails.
    # Useful for CI/CD or scripting.
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()