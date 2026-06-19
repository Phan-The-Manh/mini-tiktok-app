"""
_smoke_math.py — unit checks for the pure-NumPy math core (step 4.2)
===================================================================

No infra required (no Mongo / Redis / network). Run:

    python -m user_embedding.tests._smoke_math

Asserts the embedding math behaves: normalization, signed action weights,
the short-term EMA pulling toward / away from videos, long-term aggregation,
the query blend, and cold-start zero handling.
"""

from __future__ import annotations

import numpy as np

from user_embedding.services import math_core as mc


def _rand_unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mc.l2_normalize(rng.standard_normal(dim))


def test_normalize_and_cold_start() -> None:
    v = mc.l2_normalize([3.0, 4.0])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    # Zero / empty normalize to zeros, not NaN.
    assert mc.is_zero(mc.l2_normalize([0.0, 0.0]))
    assert mc.is_zero(mc.l2_normalize([]))
    assert mc.is_zero(mc.zero_vector())


def test_action_weights() -> None:
    assert mc.action_weight("like") == 1.0
    assert mc.action_weight("not_interested") == -1.0
    assert mc.action_weight("report") < 0
    assert mc.action_weight("unknown_action") == 0.0
    # Graded watch: completion strong positive, quick bail soft negative.
    assert mc.action_weight("watch", is_completion=True) == 1.0
    assert mc.action_weight("watch", watch_pct=1.0) > 0.9
    assert mc.action_weight("watch", watch_pct=0.05) < 0.0
    assert mc.action_weight("watch", watch_pct=None) == 0.0


def test_short_term_pulls_toward_positive() -> None:
    dim = mc.EMBEDDING_DIM
    video = _rand_unit(dim, seed=1)

    # Cold start: first positive interaction defines the direction.
    s1 = mc.update_short_term(mc.zero_vector(dim), video, weight=1.0, decay=0.9)
    assert abs(np.linalg.norm(s1) - 1.0) < 1e-9
    assert mc.cosine(s1, video) > 0.99  # points at the video

    # A second, unrelated positive video should move it toward that video,
    # i.e. similarity to the new video increases vs before the update.
    other = _rand_unit(dim, seed=2)
    before = mc.cosine(s1, other)
    s2 = mc.update_short_term(s1, other, weight=1.0, decay=0.9)
    after = mc.cosine(s2, other)
    assert after > before
    assert abs(np.linalg.norm(s2) - 1.0) < 1e-9


def test_short_term_pushes_away_on_negative() -> None:
    dim = mc.EMBEDDING_DIM
    liked = _rand_unit(dim, seed=3)
    s = mc.update_short_term(mc.zero_vector(dim), liked, weight=1.0, decay=0.9)

    disliked = _rand_unit(dim, seed=4)
    before = mc.cosine(s, disliked)
    s2 = mc.update_short_term(s, disliked, weight=-1.0, decay=0.9)
    after = mc.cosine(s2, disliked)
    assert after < before  # steered away from the disliked video


def test_long_term_aggregation() -> None:
    dim = mc.EMBEDDING_DIM
    vids = [_rand_unit(dim, seed=s) for s in (10, 11, 12)]
    lt = mc.aggregate_long_term(vids, weights=[1.0, 1.0, 1.0])
    assert abs(np.linalg.norm(lt) - 1.0) < 1e-9
    # The aggregate should be positively correlated with its constituents.
    assert all(mc.cosine(lt, v) > 0 for v in vids)
    # Empty history stays cold-start.
    assert mc.is_zero(mc.aggregate_long_term([]))


def test_blend_query() -> None:
    dim = mc.EMBEDDING_DIM
    lt = _rand_unit(dim, seed=20)
    st = _rand_unit(dim, seed=21)

    q = mc.blend_query(lt, st, beta=0.5)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    assert mc.cosine(q, lt) > 0 and mc.cosine(q, st) > 0

    # beta extremes collapse to a single side.
    q_long = mc.blend_query(lt, st, beta=1.0)
    assert mc.cosine(q_long, lt) > 0.99
    q_short = mc.blend_query(lt, st, beta=0.0)
    assert mc.cosine(q_short, st) > 0.99

    # One side cold -> result is the other side.
    only_short = mc.blend_query(mc.zero_vector(dim), st, beta=0.5)
    assert mc.cosine(only_short, st) > 0.99
    # Both cold -> cold-start zero vector.
    assert mc.is_zero(mc.blend_query(mc.zero_vector(dim), mc.zero_vector(dim)))


def test_dim_mismatch_raises() -> None:
    try:
        mc.update_short_term([0.1, 0.2, 0.3], [0.1, 0.2], weight=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on dim mismatch")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[OK] {t.__name__}")
    print(f"[OK] math_core: all {len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
