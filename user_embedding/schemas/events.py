"""
events.py — Redis Stream event payloads (consumer side)
=======================================================

`UserActionEvent` is the wire contract the User Embedding Service consumes on
the `user.action` stream. The Event Service (Component #8) will publish this
later; until then the smoke test publishes synthetic events with the same
field shape. As with the Content Analyzer, the shared contract is the wire
format (string fields in the Redis entry), not a Python class import.

Only the fields the embedding update needs are modeled here — a subset of the
full `Interaction` document. `interaction_id` is the idempotency key.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class UserActionEvent(BaseModel):
    """Emitted on Redis Stream `user.action`; consumed to update user vectors."""

    interaction_id: str
    user_id: str
    video_id: str
    action: str
    watch_pct: Optional[float] = None
    is_completion: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_stream_fields(cls, fields: dict[str, str]) -> "UserActionEvent":
        """Inverse of a publisher's `to_stream_fields`: cast the flat string
        fields of a Redis stream entry back to typed values. Missing required
        keys raise ValidationError; missing optional keys are tolerated."""
        data: dict = dict(fields)
        if data.get("watch_pct") not in (None, ""):
            data["watch_pct"] = float(data["watch_pct"])
        else:
            data.pop("watch_pct", None)
        if "is_completion" in data:
            data["is_completion"] = str(data["is_completion"]).lower() in (
                "1", "true", "yes",
            )
        if "timestamp" in data:
            data["timestamp"] = _parse_dt(data["timestamp"])
        return cls.model_validate(data)

    def to_stream_fields(self) -> dict[str, str]:
        """Flatten to the string-only field map Redis Streams stores. Used by
        the smoke test (and, later, by any test publisher)."""
        out: dict[str, str] = {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "action": self.action,
            "is_completion": "true" if self.is_completion else "false",
            "timestamp": self.timestamp.isoformat(),
        }
        if self.watch_pct is not None:
            out["watch_pct"] = repr(float(self.watch_pct))
        return out
