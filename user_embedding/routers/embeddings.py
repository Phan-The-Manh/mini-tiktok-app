"""
routers/embeddings.py — HTTP surface (step 4.5)
===============================================

GET  /users/{user_id}/embedding   blended query vector + metadata (for Recall)
GET  /health                       liveness + Mongo/Redis dependency check
POST /users/{user_id}/interactions dev-only: apply a synthetic interaction

The POST endpoint is a developer convenience for exercising the update path
without the Event Service; it shares `update.apply_interaction` with the
stream consumer, so both paths are identical.
"""

from __future__ import annotations

import uuid
from typing import Optional

import user_embedding._path  # noqa: F401  side-effect: sys.path + env

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from client import get_mongo, get_redis  # type: ignore[import-not-found]

from user_embedding.schemas.events import UserActionEvent
from user_embedding.services import update as update_svc

router = APIRouter()


class EmbeddingOut(BaseModel):
    user_id: str
    embedding: list[float]
    dim: int
    cold_start: bool
    has_long_term: bool
    has_short_term: bool


class InteractionIn(BaseModel):
    video_id: str
    action: str
    watch_pct: Optional[float] = None
    is_completion: bool = False
    interaction_id: Optional[str] = None


@router.get("/users/{user_id}/embedding", response_model=EmbeddingOut)
def get_embedding(user_id: str) -> EmbeddingOut:
    try:
        return EmbeddingOut(**update_svc.get_query_vector(user_id))
    except update_svc.UnknownUser:
        raise HTTPException(status_code=404, detail=f"user '{user_id}' not found")


@router.post("/users/{user_id}/interactions", summary="Dev: apply a synthetic interaction")
def post_interaction(user_id: str, body: InteractionIn) -> dict:
    event = UserActionEvent(
        interaction_id=body.interaction_id or f"i_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        video_id=body.video_id,
        action=body.action,
        watch_pct=body.watch_pct,
        is_completion=body.is_completion,
    )
    try:
        return update_svc.apply_interaction(event)
    except update_svc.UnknownUser:
        raise HTTPException(status_code=404, detail=f"user '{user_id}' not found")
    except update_svc.UnknownVideo:
        raise HTTPException(status_code=404, detail=f"video '{body.video_id}' not found")
    except update_svc.VideoNotEmbedded:
        raise HTTPException(
            status_code=409,
            detail=f"video '{body.video_id}' has no content_embedding yet",
        )


@router.get("/health", summary="Liveness + dependency check")
def health() -> dict:
    out: dict[str, str] = {}
    try:
        get_mongo().command("ping")
        out["mongo"] = "ok"
    except Exception as e:
        out["mongo"] = f"fail: {e}"
    try:
        get_redis().ping()
        out["redis"] = "ok"
    except Exception as e:
        out["redis"] = f"fail: {e}"
    return out
