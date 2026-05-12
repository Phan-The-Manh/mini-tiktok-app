# test_vector_search.py — DELETE AFTER TESTING
import random
import time
from client import get_mongo

db = get_mongo()

# 1. Generate a random 384-dim vector (in real life this would come from CLIP/sentence-transformers)
def random_vector(dim=384):
    return [random.gauss(0, 1) for _ in range(dim)]

# 2. Insert one fake video with a random embedding
fake_video = {
    "video_id": "v_test_vector",
    "author_id": "u_test",
    "url": "http://test",
    "duration_seconds": 10.0,
    "caption": "test",
    "content_embedding": random_vector(),
    "moderation_status": "approved",
    "distribution_stage": "mainstream",
}

# Delete existing if any (idempotent test)
db.videos.delete_one({"video_id": "v_test_vector"})
db.videos.insert_one(fake_video)
print("Inserted test video with 384-dim embedding")

# 3. Query the vector index with another random vector
query_vector = random_vector()

pipeline = [
    {
        "$vectorSearch": {
            "index": "video_content_index",
            "path": "content_embedding",
            "queryVector": query_vector,
            "numCandidates": 10,
            "limit": 5,
            "filter": {
                "moderation_status": {"$eq": "approved"}
            }
        }
    },
    {
        "$project": {
            "_id": 0,
            "video_id": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

# Poll: Atlas Search indexes are eventually consistent (1-5s lag after writes).
results = []
for attempt in range(1, 16):
    results = list(db.videos.aggregate(pipeline))
    print(f"   attempt {attempt}: {len(results)} result(s)")
    if results:
        break
    time.sleep(1)

print(f"\nVector search returned {len(results)} result(s):")
for r in results:
    print(f"   {r}")

# 4. Clean up
db.videos.delete_one({"video_id": "v_test_vector"})
print("\nCleaned up test video")
