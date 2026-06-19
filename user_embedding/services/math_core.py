"""
math_core.py — pure-NumPy embedding math (no I/O)
=================================================

Step 4.2 of the User Embedding Service. Everything here is deterministic and
dependency-free apart from NumPy, so it can be unit-tested without Mongo,
Redis, or any network. The I/O layers (store.py, cache.py, the consumer)
call into these functions; they never reimplement the math.

Vector space
------------
User vectors live in the SAME space as `videos.content_embedding` (384-d,
cosine). That is what lets the Recall Service vector-search a user's query
vector directly against video embeddings. We L2-normalize everything so that
cosine similarity is just a dot product and no single update can blow up the
magnitude.

The three operations
--------------------
1. `update_short_term` — fold one interaction into the short-term vector with
   an exponential moving average (EMA). Positive actions pull the vector
   toward the video; negative actions (skip / not_interested / report) push
   it away.
2. `aggregate_long_term` — (re)build the long-term vector as a weighted mean
   of the videos a user engaged with positively over a long window.
3. `blend_query` — combine long + short into the single query vector served
   to Recall.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

# Must match schemas.user.EMBEDDING_DIM and the Atlas vector index dimension.
# Duplicated as a plain constant (rather than imported) so this module stays
# importable with zero project-path setup for fast unit testing.
EMBEDDING_DIM = 384


# --- Primitives --------------------------------------------------------------

def l2_normalize(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return `vec` scaled to unit L2 norm as a float64 array.

    A zero (or non-finite) vector normalizes to zeros rather than dividing by
    zero — this is the natural "no signal yet" / cold-start state.
    """
    v = np.asarray(vec, dtype=np.float64).ravel()
    norm = np.linalg.norm(v)
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros_like(v)
    return v / norm


def cosine(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either side is a zero vector."""
    na, nb = l2_normalize(a), l2_normalize(b)
    if na.size == 0 or nb.size == 0:
        return 0.0
    return float(np.dot(na, nb))


def is_zero(vec: Sequence[float] | np.ndarray) -> bool:
    """True if `vec` is empty or all-zeros (the cold-start sentinel)."""
    v = np.asarray(vec, dtype=np.float64).ravel()
    return v.size == 0 or not np.any(v)


def zero_vector(dim: int = EMBEDDING_DIM) -> np.ndarray:
    return np.zeros(dim, dtype=np.float64)


# --- Action weights ----------------------------------------------------------

# Base signed weight per action. `watch` is special-cased below because its
# strength depends on how much of the video was actually watched.
_ACTION_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "share": 1.0,
    "follow": 0.9,
    "comment": 0.7,
    "skip": -0.5,
    "not_interested": -1.0,
    "report": -1.0,
}


def action_weight(
    action: str,
    *,
    watch_pct: Optional[float] = None,
    is_completion: bool = False,
) -> float:
    """Map an interaction to a signed scalar in roughly [-1, 1].

    Positive = "more like this", negative = "less like this". The magnitude
    scales how hard the short-term EMA is pulled toward (or away from) the
    video.

    `watch` is graded by `watch_pct`: a near-full watch is a strong positive,
    a quick bail-out is a mild negative. A completion flag (>=95%) clamps to
    the strongest watch signal even if `watch_pct` rounding is slightly under.
    """
    a = (action or "").lower()

    if a == "watch":
        if is_completion:
            return 1.0
        if watch_pct is None:
            return 0.0
        pct = float(np.clip(watch_pct, 0.0, 1.0))
        # Map [0,1] -> [-0.3, 1.0]: bailing in the first moments is a soft
        # negative; finishing is a strong positive.
        return -0.3 + 1.3 * pct

    return _ACTION_WEIGHTS.get(a, 0.0)


# --- Short-term EMA update ---------------------------------------------------

def update_short_term(
    short_term: Sequence[float] | np.ndarray,
    video_vec: Sequence[float] | np.ndarray,
    weight: float,
    *,
    decay: float = 0.9,
) -> np.ndarray:
    """Fold one interaction into the short-term vector via EMA and re-normalize.

        new = normalize(decay * short_term + (1 - decay) * weight * v)

    where `v` is the unit-normalized video vector. A `weight` of 0 (e.g. a
    watch with unknown duration) leaves the direction unchanged. A negative
    weight steers the user vector away from the video.

    On a cold short-term vector (empty / zeros) the result is just the signed,
    normalized video vector — the first interaction defines the direction.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")

    v = l2_normalize(video_vec)
    if v.size == 0:
        raise ValueError("video_vec is empty")

    s = np.asarray(short_term, dtype=np.float64).ravel()
    if s.size == 0:
        s = np.zeros_like(v)
    elif s.size != v.size:
        raise ValueError(
            f"dim mismatch: short_term={s.size}, video_vec={v.size}"
        )

    blended = decay * s + (1.0 - decay) * float(weight) * v
    return l2_normalize(blended)


# --- Long-term aggregation ---------------------------------------------------

def aggregate_long_term(
    video_vecs: Iterable[Sequence[float] | np.ndarray],
    weights: Optional[Iterable[float]] = None,
    *,
    dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    """Weighted mean of unit-normalized video vectors, re-normalized.

    Used by the nightly / Colab batch to rebuild a user's durable interests
    from a long window of positive interactions. Pass only positive-signal
    videos (the batch filters on action weight before calling this). Returns a
    zero vector for an empty input — i.e. a user with no qualifying history
    stays cold-start.
    """
    vecs = [l2_normalize(v) for v in video_vecs]
    if not vecs:
        return zero_vector(dim)

    mat = np.vstack(vecs)
    if weights is None:
        w = np.ones(mat.shape[0], dtype=np.float64)
    else:
        w = np.asarray(list(weights), dtype=np.float64)
        if w.size != mat.shape[0]:
            raise ValueError(
                f"weights ({w.size}) must match video count ({mat.shape[0]})"
            )

    summed = (mat * w[:, None]).sum(axis=0)
    return l2_normalize(summed)


# --- Query-vector blend ------------------------------------------------------

def blend_query(
    long_term: Sequence[float] | np.ndarray,
    short_term: Sequence[float] | np.ndarray,
    *,
    beta: float = 0.5,
) -> np.ndarray:
    """Combine long + short into the single query vector served to Recall.

        q = normalize(beta * normalize(long) + (1 - beta) * normalize(short))

    If only one side has signal, the result is just that side (the zero side
    contributes nothing). If neither side has signal, the result is a zero
    vector — the caller should treat that as cold-start and fall back to
    demographics / trending.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    lt = l2_normalize(long_term)
    st = l2_normalize(short_term)

    if lt.size == 0 and st.size == 0:
        return zero_vector()
    if lt.size == 0:
        lt = np.zeros_like(st)
    if st.size == 0:
        st = np.zeros_like(lt)

    return l2_normalize(beta * lt + (1.0 - beta) * st)
