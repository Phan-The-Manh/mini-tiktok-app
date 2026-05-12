"""
seed_data.py
============

Populates MongoDB with realistic-looking fake data for development.

Usage (from D:\\mini-tiktok-app\\database, with venv active):
    python -m scripts.seed_data

What gets created:
    - 50 users with random demographics and embeddings
    - 200 videos with realistic captions, hashtags, and embeddings

Idempotent: running this multiple times wipes the previous seed and re-creates.
            (Identified by user_id/video_id prefix "seed_*")

Note: The embeddings here are RANDOM noise. The Content Analyzer (Component #3)
      will later overwrite videos with real CLIP/Whisper embeddings. Random
      vectors are fine for now — they let us test recall/ranking infrastructure.
"""

import random
from datetime import datetime, timedelta

from faker import Faker

from client import get_mongo
from schemas import (
    EMBEDDING_DIM,
    Demographics,
    DistributionStage,
    ModerationStatus,
    User,
    Video,
)

# Reproducibility: same seeds → same fake data on re-runs.
# This makes debugging much easier.
random.seed(42)
fake = Faker()
Faker.seed(42)


# --- Tunable constants ---
NUM_USERS = 50
NUM_VIDEOS = 200

# Realistic-ish content categories (used for hashtags + future diversity rules)
CATEGORIES = [
    "cooking", "comedy", "dance", "fitness", "fashion",
    "tech", "gaming", "music", "travel", "pets",
    "diy", "education", "beauty", "sports", "art",
]

CAPTION_TEMPLATES = [
    "My {time} routine ✨",
    "POV: {scenario}",
    "Tutorial: how to {action}",
    "{adjective} {noun} hack you NEED to try",
    "Day in the life of a {profession}",
    "Trying {thing} for the first time!",
    "{number} things I wish I knew about {topic}",
    "When you {action}... 😂",
    "Behind the scenes of my {thing}",
    "Quick {category} tip!",
]

COUNTRIES = ["SG", "US", "ID", "MY", "PH", "TH", "VN", "JP", "KR", "AU"]
AGE_RANGES = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
LANGUAGES = ["en", "zh", "id", "ms", "th", "vi", "ja", "ko"]


def random_embedding() -> list[float]:
    """
    Generate a random 384-dim unit vector (normalized, like real embeddings).

    Real embeddings from CLIP / sentence-transformers are typically L2-normalized,
    so cosine similarity is meaningful. We normalize here to match.
    """
    raw = [random.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    # L2 normalize
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


def make_caption(category: str) -> str:
    """Generate a TikTok-style caption with hashtags."""
    template = random.choice(CAPTION_TEMPLATES)
    filled = template.format(
        time=random.choice(["morning", "evening", "weekend"]),
        scenario=fake.sentence(nb_words=4).rstrip("."),
        action=fake.word(),
        adjective=random.choice(["amazing", "wild", "simple", "underrated"]),
        noun=fake.word(),
        profession=random.choice(["barista", "developer", "designer", "student"]),
        thing=fake.word(),
        number=random.randint(3, 10),
        topic=category,
        category=category,
    )
    return filled


def generate_user(idx: int) -> dict:
    """Build one fake User document (returned as dict for Mongo insert)."""
    user = User(
        user_id=f"seed_u_{idx:03d}",            # zero-padded for sortability: seed_u_001, seed_u_002...
        username=fake.user_name(),
        created_at=fake.date_time_between(start_date="-1y", end_date="now"),
        demographics=Demographics(
            age_range=random.choice(AGE_RANGES),
            country=random.choice(COUNTRIES),
            language=random.choice(LANGUAGES),
        ),
        long_term_embedding=random_embedding(),
        short_term_embedding=random_embedding(),
        short_term_updated_at=datetime.utcnow(),
    )
    # Pydantic's model_dump() converts the model to a plain dict for MongoDB.
    # mode="python" preserves datetime objects (which pymongo handles natively).
    return user.model_dump(mode="python")


def generate_video(idx: int, author_ids: list[str]) -> dict:
    """Build one fake Video document."""
    category = random.choice(CATEGORIES)
    uploaded_at = fake.date_time_between(start_date="-90d", end_date="now")

    # Most videos are approved+mainstream. Some are still in cold-start.
    # We model this distribution to make recall queries interesting.
    rand = random.random()
    if rand < 0.7:
        moderation = ModerationStatus.APPROVED
        stage = DistributionStage.MAINSTREAM
    elif rand < 0.9:
        moderation = ModerationStatus.APPROVED
        stage = random.choice([DistributionStage.TEST_POOL_1, DistributionStage.TEST_POOL_2])
    else:
        moderation = ModerationStatus.PENDING
        stage = DistributionStage.TEST_POOL_1

    video = Video(
        video_id=f"seed_v_{idx:03d}",
        author_id=random.choice(author_ids),
        uploaded_at=uploaded_at,
        url=f"http://localhost:9000/videos/seed_v_{idx:03d}.mp4",  # MinIO path (videos don't exist yet)
        thumbnail_url=f"http://localhost:9000/thumbnails/seed_v_{idx:03d}.jpg",
        duration_seconds=round(random.uniform(8, 60), 1),
        caption=make_caption(category),
        hashtags=[category] + random.sample(CATEGORIES, k=2),
        category=category,
        content_embedding=random_embedding(),
        moderation_status=moderation,
        distribution_stage=stage,
        next_review_at=uploaded_at + timedelta(hours=random.randint(1, 24))
            if stage != DistributionStage.MAINSTREAM else None,
    )
    return video.model_dump(mode="python")


def wipe_existing_seed(db):
    """Remove previous seed data so we can re-run cleanly."""
    print("🧹 Wiping previous seed data...")
    u = db.users.delete_many({"user_id": {"$regex": "^seed_"}})
    v = db.videos.delete_many({"video_id": {"$regex": "^seed_"}})
    print(f"   Removed {u.deleted_count} users, {v.deleted_count} videos")


def main():
    print("=" * 60)
    print("🌱 Seeding mini-tiktok with fake data")
    print("=" * 60)

    db = get_mongo()
    print(f"Connected to database: {db.name}\n")

    wipe_existing_seed(db)

    # --- Users ---
    print(f"\n👥 Generating {NUM_USERS} users...")
    user_docs = [generate_user(i) for i in range(NUM_USERS)]
    db.users.insert_many(user_docs)
    print(f"   ✅ Inserted {len(user_docs)} users")

    author_ids = [u["user_id"] for u in user_docs]

    # --- Videos ---
    print(f"\n🎬 Generating {NUM_VIDEOS} videos...")
    video_docs = [generate_video(i, author_ids) for i in range(NUM_VIDEOS)]
    db.videos.insert_many(video_docs)
    print(f"   ✅ Inserted {len(video_docs)} videos")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("📊 SEED SUMMARY")
    print("=" * 60)
    print(f"  users:        {db.users.count_documents({})}")
    print(f"  videos:       {db.videos.count_documents({})}")
    print(f"  interactions: {db.interactions.count_documents({})} (none seeded — will come later)")

    # Show a sample of each
    print("\n📄 Sample user:")
    sample_user = db.users.find_one({}, {"_id": 0, "long_term_embedding": 0, "short_term_embedding": 0})
    for k, v in sample_user.items():
        print(f"     {k}: {v}")

    print("\n📄 Sample video:")
    sample_video = db.videos.find_one({}, {"_id": 0, "content_embedding": 0})
    for k, v in sample_video.items():
        print(f"     {k}: {v}")

    print("\n" + "=" * 60)
    print("✅ Seeding complete")
    print("=" * 60)


if __name__ == "__main__":
    main()