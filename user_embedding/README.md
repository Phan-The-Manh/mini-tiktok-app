# User Embedding Service (Component #4)

Maintains two vectors per user and serves a single blended **query vector** to
the Recall Service (Component #5). User vectors live in the **same 384-d
cosine space as `videos.content_embedding`**, so Recall can vector-search a
user's vector directly against videos.

- `short_term_embedding` — current-session interests, updated in near real
  time from interaction events via an exponential moving average (EMA).
- `long_term_embedding` — durable interests, recomputed in batch (nightly /
  Colab) from a long window of positive interactions.

Both fields already exist on the `User` document (`database/schemas/user.py`).

> Status: **done and verified (2026-06-19).** All of steps 4.1-4.10 are
> implemented and tested against live Redis + MongoDB Atlas: math core, Mongo
> adapter, Redis write-through cache, FastAPI service, the `user.action`
> consumer (retry/DLQ/idempotency), the long-term recompute batch, and the
> end-to-end smoke test. The math core is unit-tested with no infra
> (`tests/_smoke_math.py`, 7 checks); `tests/smoke_test.py` passes end-to-end.
> See `TODO.md` for the full Definition of Done.

---

## Contract

### HTTP

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/users/{user_id}/embedding` | Blended query vector + metadata (`dim`, freshness, `cold_start`) |
| `GET`  | `/health` | Liveness + Mongo/Redis dependency check |
| `POST` | `/users/{user_id}/interactions` | Dev-only: apply a synthetic interaction by hand |

Cold start: a user with neither embedding returns an empty vector and
`cold_start: true` so Recall can fall back to demographics / trending instead
of vector search.

### Input — Redis Streams

Stream `user.action` (override with `USER_ACTION_STREAM`), consumer group
`user_embedding`. Field shape: `user_embedding/schemas/events.py:UserActionEvent`.
`interaction_id` is the idempotency key. Failed messages land in
`user.action.dlq` after `USER_EMBEDDING_MAX_RETRIES` (default 3) deliveries.

> The live stream is owned by the Event Service (Component #8, built later).
> Until then the consumer is exercised with synthetic events.

---

## Embedding math (the design decisions)

All math lives in `services/math_core.py` — pure NumPy, no I/O, unit-tested by
`tests/_smoke_math.py`. Everything is L2-normalized so cosine similarity is a
plain dot product and no update can blow up the magnitude.

**Action weights** (`action_weight`) map an interaction to a signed scalar:

| Action | Weight |
|--------|--------|
| like / share | +1.0 |
| follow | +0.9 |
| comment | +0.7 |
| watch (completion) | +1.0 |
| watch (graded) | `-0.3 + 1.3 * watch_pct` |
| skip | -0.5 |
| not_interested / report | -1.0 |

**Short-term EMA** (`update_short_term`):

```
new = normalize(decay * short_term + (1 - decay) * weight * v)
```

`decay` defaults to `0.9` (`USER_EMBEDDING_DECAY`) — the vector keeps ~90% of
its prior state per update, so a session drifts over several interactions
rather than snapping to the latest video. Positive weights pull toward the
video; negative weights push away.

**Long-term aggregation** (`aggregate_long_term`): weighted mean of the
unit-normalized vectors of videos the user engaged with positively over a
window, re-normalized. Rebuilt by the batch job, not per-event.

**Query blend** (`blend_query`):

```
q = normalize(beta * long_term + (1 - beta) * short_term)
```

`beta` defaults to `0.5` (`USER_EMBEDDING_BLEND_BETA`) — equal weight on
durable and session interests. If only one side has signal, the query is just
that side; if neither does, the query is the cold-start zero vector.

---

## Setup

Reuses `database/.env` via the `_path.py` shim. Per-service overrides go in
`user_embedding/.env` (see `.env.example`).

```powershell
pip install -r user_embedding/requirements.txt
```

Run the math unit checks (no infra needed):

```powershell
python -m user_embedding.tests._smoke_math
```

Run the HTTP service (read path for Recall):

```powershell
uvicorn user_embedding.main:app --port 8002
# GET  http://localhost:8002/users/{user_id}/embedding
# GET  http://localhost:8002/health
# POST http://localhost:8002/users/{user_id}/interactions   (dev: apply a synthetic interaction)
```

Run the `user.action` consumer (write path; separate process):

```powershell
python -m user_embedding.consumers.user_action
```

Rebuild long-term vectors in batch (nightly / Colab):

```powershell
python -m user_embedding.notebooks.recompute_longterm --limit 100
```

End-to-end smoke test (needs Redis + MongoDB up):

```powershell
python -m user_embedding.tests.smoke_test
```

---

## Layout

```
user_embedding/
|-- README.md
|-- requirements.txt
|-- .env.example
|-- _path.py                      sys.path + .env shim
|-- main.py                       FastAPI app
|-- routers/
|   `-- embeddings.py             GET /users/{id}/embedding, /health, dev POST
|-- services/
|   |-- math_core.py              pure-NumPy EMA / aggregation / blend
|   |-- store.py                  Mongo read/write adapter
|   |-- cache.py                  Redis write-through vector cache
|   `-- update.py                 apply-interaction + read-query orchestration
|-- consumers/
|   `-- user_action.py            Redis Streams XREADGROUP + ack/dlq
|-- schemas/
|   `-- events.py                 UserActionEvent
|-- notebooks/
|   `-- recompute_longterm.py     long-term batch path
`-- tests/
    |-- _smoke_math.py            math core unit checks (no infra)
    `-- smoke_test.py             end-to-end (needs Redis + Mongo)
```
