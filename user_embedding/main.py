"""
main.py — FastAPI entrypoint (step 4.5)
=======================================

Run with (from project root, venv active):

    uvicorn user_embedding.main:app --reload --port 8002

Or:

    python -m user_embedding.main

This service is read-mostly for Recall (`GET /users/{id}/embedding`). The
write path (folding interactions into short-term vectors) is driven by the
`user.action` Redis Streams consumer in `consumers/user_action.py`, run as a
separate process; the dev `POST /users/{id}/interactions` endpoint exercises
the same update code without the Event Service.
"""

from __future__ import annotations

import os

import user_embedding._path  # noqa: F401  configures sys.path + env first

from fastapi import FastAPI

from user_embedding.routers import embeddings

app = FastAPI(
    title="Mini-TikTok User Embedding Service",
    description="Maintains short-term + long-term user vectors and serves a blended query vector to Recall.",
    version="0.1.0",
)

app.include_router(embeddings.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "user_embedding.main:app",
        host=os.getenv("USER_EMBEDDING_HOST", "0.0.0.0"),
        port=int(os.getenv("USER_EMBEDDING_PORT", "8002")),
        reload=False,
    )
