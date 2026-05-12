"""
Experiment schema
-----------------
Configuration for A/B tests. Stored in the `experiments` collection.

We don't store *results* here — those are computed at analysis time
by aggregating from the `interactions` collection (filtered by experiment_variant).
This document just defines what the experiment is and how users are bucketed.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TrafficAllocation(BaseModel):
    """
    How traffic splits across variants.

    Example: {"control": 50, "treatment": 50} means a 50/50 split.
    Values must sum to 100.
    """
    control: int = 50
    treatment: int = 50


class Experiment(BaseModel):
    """An A/B test definition."""

    # --- Identity ---
    experiment_id: str                              # e.g., "exp_ranking_v2"
    name: str                                       # human-readable: "New ranking model"
    description: str = ""

    # --- Lifecycle ---
    is_active: bool = True
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    # --- Traffic split ---
    variants: list[str] = Field(default_factory=lambda: ["control", "treatment"])
    traffic_allocation: TrafficAllocation = Field(default_factory=TrafficAllocation)
 
    # --- What's being tested ---
    # Free-form: which component the experiment affects
    target_component: Optional[str] = None          # e.g., "ranking", "recall", "rerank"

    # --- Owner / metadata ---
    owner: Optional[str] = None                     # who's running this experiment
    primary_metric: Optional[str] = None            # e.g., "avg_watch_time"
