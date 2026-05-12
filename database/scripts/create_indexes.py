"""
create_indexes.py
=================

Run this ONCE to create all standard MongoDB indexes for the project.

Usage (from D:\\mini-tiktok-app\\database, with venv active):
    python -m scripts.create_indexes

Why `python -m`?
    The `-m` flag runs scripts/create_indexes.py as a module, which makes
    `from client import get_mongo` work correctly. Running `python scripts/create_indexes.py`
    directly would break imports because of how Python resolves paths.

Note:
    - This script is idempotent: running it multiple times is safe.
      MongoDB skips index creation if an identical index already exists.
    - The vector search index on `videos.content_embedding` is NOT created here.
      That requires the Atlas UI (M0 free tier limitation) — handled in Step 8.
"""

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import OperationFailure

from client import get_mongo


def create_user_indexes(db):
    """Indexes for the `users` collection."""
    print("\n[*] Creating indexes on `users`...")

    # Primary lookup: find user by user_id. Must be unique.
    db.users.create_index(
        [("user_id", ASCENDING)],
        unique=True,
        name="user_id_unique",
    )
    print("   [OK] user_id_unique")


def create_video_indexes(db):
    """Indexes for the `videos` collection."""
    print("\n[*] Creating indexes on `videos`...")

    # Primary lookup: find video by video_id.
    db.videos.create_index(
        [("video_id", ASCENDING)],
        unique=True,
        name="video_id_unique",
    )
    print("   [OK] video_id_unique")

    # "Show me this creator's videos, newest first."
    # Compound index: filters by author_id, sorts by uploaded_at DESC.
    db.videos.create_index(
        [("author_id", ASCENDING), ("uploaded_at", DESCENDING)],
        name="author_recent",
    )
    print("   [OK] author_recent (author_id + uploaded_at)")

    # "Which videos in test pools need re-evaluation?"
    # Used by the cold-start promotion job.
    db.videos.create_index(
        [("distribution_stage", ASCENDING), ("next_review_at", ASCENDING)],
        name="cold_start_review",
    )
    print("   [OK] cold_start_review (distribution_stage + next_review_at)")


def create_interaction_indexes(db):
    """Indexes for the `interactions` collection."""
    print("\n[*] Creating indexes on `interactions`...")

    # "Show me a user's recent activity."
    # Filters by user_id, sorts by timestamp DESC.
    db.interactions.create_index(
        [("user_id", ASCENDING), ("timestamp", DESCENDING)],
        name="user_history",
    )
    print("   [OK] user_history (user_id + timestamp)")

    # "Show me this video's recent engagement."
    # Used by stream processors when computing video stats.
    db.interactions.create_index(
        [("video_id", ASCENDING), ("timestamp", DESCENDING)],
        name="video_engagement",
    )
    print("   [OK] video_engagement (video_id + timestamp)")

    # TTL index: auto-delete interactions older than 90 days.
    # MongoDB checks this every 60 seconds and removes expired docs.
    # Critical for the M0 free tier's 512 MB storage limit.
    NINETY_DAYS_IN_SECONDS = 90 * 24 * 60 * 60
    db.interactions.create_index(
        [("timestamp", ASCENDING)],
        expireAfterSeconds=NINETY_DAYS_IN_SECONDS,
        name="interactions_ttl",
    )
    print("   [OK] interactions_ttl (auto-delete after 90 days)")


def create_experiment_indexes(db):
    """Indexes for the `experiments` collection."""
    print("\n[*] Creating indexes on `experiments`...")

    db.experiments.create_index(
        [("experiment_id", ASCENDING)],
        unique=True,
        name="experiment_id_unique",
    )
    print("   [OK] experiment_id_unique")

    db.experiments.create_index(
        [("is_active", ASCENDING)],
        name="active_experiments",
    )
    print("   [OK] active_experiments")


def list_all_indexes(db):
    """Print a summary of every index on every collection."""
    print("\n" + "=" * 60)
    print("INDEX SUMMARY")
    print("=" * 60)

    for coll_name in ["users", "videos", "interactions", "experiments"]:
        coll = db[coll_name]
        indexes = list(coll.list_indexes())
        print(f"\n  {coll_name} ({len(indexes)} indexes):")
        for idx in indexes:
            keys = ", ".join(f"{k}:{v}" for k, v in idx["key"].items())
            extras = []
            if idx.get("unique"):
                extras.append("unique")
            if idx.get("expireAfterSeconds") is not None:
                extras.append(f"TTL={idx['expireAfterSeconds']}s")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            print(f"    - {idx['name']}: ({keys}){extra_str}")


def main():
    print("=" * 60)
    print("Creating MongoDB indexes for mini-tiktok")
    print("=" * 60)

    db = get_mongo()
    print(f"\nConnected to database: {db.name}")

    try:
        create_user_indexes(db)
        create_video_indexes(db)
        create_interaction_indexes(db)
        create_experiment_indexes(db)
    except OperationFailure as e:
        print(f"\n[FAIL] Index creation failed: {e}")
        print("   This usually means an index with the same name but different config already exists.")
        print("   To reset: drop the collection in Atlas UI and re-run.")
        return

    list_all_indexes(db)

    print("\n" + "=" * 60)
    print("[OK] All indexes created successfully")
    print("=" * 60)
    print("\nNext: create the Vector Search index in the Atlas UI (Step 8).")


if __name__ == "__main__":
    main()