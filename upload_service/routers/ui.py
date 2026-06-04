"""
routers/ui.py - Throwaway dev UI
================================

A single embedded HTML page + a few JSON helper endpoints so a developer
can upload a video from the browser and replay it once the Content
Analyzer has embedded it. Not the production frontend (that is the
planned Next.js swipe UI in Component #7) - this exists only to close
the upload -> analyze -> watch loop on localhost without touching curl
or Postman.

Endpoints:
    GET  /                          embedded HTML page (form + gallery)
    GET  /ui/users                  list of users for the author dropdown
    GET  /ui/videos                 recent videos (newest first)
    GET  /ui/videos/{id}/stream     proxy stream from MinIO with Range support

The stream proxy avoids requiring a public-read MinIO bucket: the browser
hits FastAPI, FastAPI signs the MinIO request server-side. Supports HTTP
Range so the HTML5 <video> tag can seek.
"""

from __future__ import annotations

import os
from typing import Iterator

import upload_service._path  # noqa: F401

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from client import get_minio, get_mongo  # type: ignore[import-not-found]

from upload_service.services import storage


router = APIRouter()


# --- HTML page ---------------------------------------------------------------

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Mini-TikTok dev UI</title>
<style>
  :root { color-scheme: dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #111; color: #eee; margin: 0; padding: 24px;
  }
  h1 { margin: 0 0 16px 0; font-size: 20px; font-weight: 600; }
  h2 { margin: 24px 0 8px 0; font-size: 14px; text-transform: uppercase;
       letter-spacing: 0.08em; color: #888; }
  .row { display: flex; gap: 24px; align-items: flex-start; }
  .col { flex: 1; min-width: 0; }
  .card {
    background: #1c1c1c; border: 1px solid #2a2a2a; border-radius: 8px;
    padding: 16px; margin-bottom: 16px;
  }
  label { display: block; font-size: 12px; color: #aaa; margin: 8px 0 4px; }
  input, select, textarea, button {
    width: 100%; box-sizing: border-box; padding: 8px 10px; font-size: 14px;
    background: #111; color: #eee; border: 1px solid #333; border-radius: 6px;
    font-family: inherit;
  }
  button {
    background: #2563eb; border-color: #2563eb; color: white; cursor: pointer;
    margin-top: 12px; font-weight: 600;
  }
  button:hover:not(:disabled) { background: #1d4ed8; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .status { font-size: 12px; color: #aaa; margin-top: 8px; min-height: 16px; }
  .status.ok { color: #4ade80; }
  .status.err { color: #f87171; }
  .badge {
    display: inline-block; padding: 2px 6px; border-radius: 4px;
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em; margin-left: 6px;
  }
  .badge.pending { background: #422006; color: #fbbf24; }
  .badge.approved { background: #052e16; color: #4ade80; }
  .badge.rejected { background: #450a0a; color: #f87171; }
  video { width: 100%; max-height: 60vh; background: black; border-radius: 6px; }
  ul.videos { list-style: none; padding: 0; margin: 0; max-height: 70vh;
              overflow-y: auto; }
  ul.videos li {
    padding: 8px 10px; border-radius: 6px; cursor: pointer;
    border: 1px solid transparent; margin-bottom: 4px;
  }
  ul.videos li:hover { background: #1c1c1c; }
  ul.videos li.selected { background: #1e293b; border-color: #2563eb; }
  ul.videos .meta { font-size: 11px; color: #888; margin-top: 2px; }
  .empty { color: #666; font-size: 13px; padding: 8px; }
  .player-meta { font-size: 12px; color: #aaa; margin-top: 8px;
                 white-space: pre-wrap; word-break: break-word; }
</style>
</head>
<body>
  <h1>Mini-TikTok dev UI</h1>

  <div class="row">
    <div class="col" style="max-width: 380px;">

      <div class="card">
        <h2>Upload</h2>
        <form id="upload-form">
          <label for="author">Author</label>
          <select id="author" required></select>

          <label for="file">Video file</label>
          <input type="file" id="file" accept="video/*" required />

          <label for="caption">Caption</label>
          <input type="text" id="caption" placeholder="optional caption" />

          <label for="hashtags">Hashtags</label>
          <input type="text" id="hashtags" placeholder="e.g. #cooking #shorts" />

          <label for="category">Category</label>
          <input type="text" id="category" placeholder="e.g. cooking" />

          <button type="submit" id="submit-btn">Upload</button>
          <div class="status" id="upload-status"></div>
        </form>
      </div>

      <div class="card">
        <h2>Library</h2>
        <ul class="videos" id="videos"></ul>
      </div>

    </div>

    <div class="col">
      <div class="card">
        <h2>Player</h2>
        <video id="player" controls playsinline preload="metadata"></video>
        <div class="player-meta" id="player-meta">Select a video from the library to play.</div>
      </div>
    </div>
  </div>

<script>
const $ = (id) => document.getElementById(id);

let selectedId = null;
let pollTimer = null;

async function loadUsers() {
  try {
    const r = await fetch('/ui/users');
    const users = await r.json();
    const sel = $('author');
    sel.innerHTML = '';
    if (users.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(no users seeded)';
      sel.appendChild(opt);
      sel.disabled = true;
      $('submit-btn').disabled = true;
      $('upload-status').className = 'status err';
      $('upload-status').textContent = 'No users in DB. Run database/scripts/seed_data.py first.';
      return;
    }
    for (const u of users) {
      const opt = document.createElement('option');
      opt.value = u.user_id;
      opt.textContent = u.username ? `${u.username} (${u.user_id})` : u.user_id;
      sel.appendChild(opt);
    }
  } catch (e) {
    $('upload-status').className = 'status err';
    $('upload-status').textContent = 'Could not load users: ' + e;
  }
}

function badge(status) {
  const cls = (status || 'pending').toLowerCase();
  return `<span class="badge ${cls}">${cls}</span>`;
}

function fmtDuration(s) {
  if (!s) return '';
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, '0')}`;
}

async function loadVideos() {
  try {
    const r = await fetch('/ui/videos');
    const videos = await r.json();
    const list = $('videos');
    list.innerHTML = '';
    if (videos.length === 0) {
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = 'No videos yet. Upload one above.';
      list.appendChild(li);
      return;
    }
    for (const v of videos) {
      const li = document.createElement('li');
      if (v.video_id === selectedId) li.classList.add('selected');
      const cap = v.caption || '(no caption)';
      li.innerHTML = `
        <div><strong>${cap}</strong> ${badge(v.moderation_status)}</div>
        <div class="meta">${v.video_id} - ${fmtDuration(v.duration_seconds)} - ${v.author_id}</div>
      `;
      li.onclick = () => selectVideo(v);
      list.appendChild(li);
    }
  } catch (e) {
    console.error(e);
  }
}

function selectVideo(v) {
  selectedId = v.video_id;
  const player = $('player');
  player.src = `/ui/videos/${v.video_id}/stream`;
  player.load();
  updatePlayerMeta(v);
  loadVideos();  // refresh selection highlight
}

function updatePlayerMeta(v) {
  const parts = [
    `video_id: ${v.video_id}`,
    `author: ${v.author_id}`,
    `duration: ${fmtDuration(v.duration_seconds)}`,
    `caption: ${v.caption || '(none)'}`,
    `hashtags: ${(v.hashtags || []).join(' ') || '(none)'}`,
    `category: ${v.category || '(none)'}`,
    `moderation_status: ${v.moderation_status}`,
    `analyzer_version: ${v.analyzer_version || '(not embedded yet)'}`,
    `content_embedding: ${v.embedding_dim ? v.embedding_dim + '-d' : 'empty (waiting for analyzer)'}`,
  ];
  $('player-meta').textContent = parts.join('\\n');
}

async function pollEmbedding(videoId, attempts) {
  attempts = attempts || 0;
  if (attempts > 60) {  // ~5 min cap
    $('upload-status').className = 'status err';
    $('upload-status').textContent = 'Gave up waiting for the analyzer. Check that the worker is running.';
    return;
  }
  try {
    const r = await fetch(`/ui/videos`);
    const videos = await r.json();
    const v = videos.find(x => x.video_id === videoId);
    if (v && v.embedding_dim > 0) {
      $('upload-status').className = 'status ok';
      $('upload-status').textContent = `[OK] analyzer finished: ${v.embedding_dim}-d, status=${v.moderation_status}`;
      loadVideos();
      if (videoId === selectedId) updatePlayerMeta(v);
      return;
    }
    $('upload-status').className = 'status';
    $('upload-status').textContent = `[INFO] waiting for analyzer... (${attempts * 5}s)`;
  } catch (e) {
    // swallow and retry
  }
  pollTimer = setTimeout(() => pollEmbedding(videoId, attempts + 1), 5000);
}

$('upload-form').onsubmit = async (e) => {
  e.preventDefault();
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }

  const file = $('file').files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('author_id', $('author').value);
  fd.append('caption', $('caption').value);
  fd.append('hashtags', $('hashtags').value);
  if ($('category').value) fd.append('category', $('category').value);

  const btn = $('submit-btn');
  btn.disabled = true;
  $('upload-status').className = 'status';
  $('upload-status').textContent = 'Uploading...';

  try {
    const r = await fetch('/videos', { method: 'POST', body: fd });
    if (!r.ok) {
      const body = await r.text();
      throw new Error(`HTTP ${r.status}: ${body}`);
    }
    const out = await r.json();
    $('upload-status').className = 'status ok';
    $('upload-status').textContent = `[OK] uploaded ${out.video_id}. Polling analyzer...`;
    $('file').value = '';
    await loadVideos();
    selectVideo({
      video_id: out.video_id,
      author_id: $('author').value,
      duration_seconds: out.duration_seconds,
      caption: $('caption').value,
      hashtags: $('hashtags').value.split(/[ ,]+/).filter(Boolean),
      category: $('category').value || null,
      moderation_status: out.moderation_status,
    });
    pollEmbedding(out.video_id);
  } catch (e) {
    $('upload-status').className = 'status err';
    $('upload-status').textContent = '[FAIL] ' + e.message;
  } finally {
    btn.disabled = false;
  }
};

loadUsers();
loadVideos();
setInterval(loadVideos, 15000);  // light periodic refresh
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(_PAGE)


# --- JSON helpers ------------------------------------------------------------

@router.get("/ui/users", include_in_schema=False)
def list_users() -> JSONResponse:
    """Cheap list for the author dropdown. Caps at 50 to keep the page small."""
    db = get_mongo()
    docs = list(
        db.users.find({}, {"_id": 0, "user_id": 1, "username": 1}).limit(50)
    )
    return JSONResponse(docs)


@router.get("/ui/videos", include_in_schema=False)
def list_videos() -> JSONResponse:
    """Recent videos, newest first. Includes analyzer status fields so the UI
    can show whether the embedding has been written."""
    db = get_mongo()
    cursor = db.videos.find(
        {},
        {
            "_id": 0,
            "video_id": 1,
            "author_id": 1,
            "caption": 1,
            "hashtags": 1,
            "category": 1,
            "duration_seconds": 1,
            "moderation_status": 1,
            "analyzer_version": 1,
            "uploaded_at": 1,
            "content_embedding": 1,
        },
    ).sort("uploaded_at", -1).limit(50)

    out = []
    for d in cursor:
        emb = d.get("content_embedding") or []
        out.append({
            "video_id": d["video_id"],
            "author_id": d.get("author_id"),
            "caption": d.get("caption", ""),
            "hashtags": d.get("hashtags", []),
            "category": d.get("category"),
            "duration_seconds": d.get("duration_seconds", 0),
            "moderation_status": str(d.get("moderation_status", "pending")),
            "analyzer_version": d.get("analyzer_version"),
            "embedding_dim": len(emb),
            "uploaded_at": d.get("uploaded_at").isoformat() if d.get("uploaded_at") else None,
        })
    return JSONResponse(out)


# --- Stream proxy ------------------------------------------------------------

# Bytes per chunk read from MinIO and yielded to the browser. 256 KB is a
# good balance: small enough that seek latency is low, large enough that
# Python-side overhead per chunk is negligible.
_CHUNK = 256 * 1024


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse a single 'bytes=start-end' header. Multi-range is not supported
    (HTML5 video never sends it). Returns (start, end) inclusive or None
    if the header is malformed."""
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].split(",")[0].strip()
    if "-" not in spec:
        return None
    s, e = spec.split("-", 1)
    try:
        if s == "":
            # suffix range: bytes=-N -> last N bytes
            length = int(e)
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(s)
            end = int(e) if e else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    end = min(end, size - 1)
    return start, end


def _stream_minio(bucket: str, key: str, start: int, end: int) -> Iterator[bytes]:
    """Yield bytes [start, end] (inclusive) from a MinIO object."""
    client = get_minio()
    length = end - start + 1
    resp = client.get_object(bucket, key, offset=start, length=length)
    try:
        remaining = length
        while remaining > 0:
            chunk = resp.read(min(_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        resp.close()
        resp.release_conn()


@router.get("/ui/videos/{video_id}/stream", include_in_schema=False)
def stream_video(video_id: str, request: Request):
    """Stream the MinIO object for `video_id` with HTTP Range support.

    Looking up the object key via the Mongo doc (rather than guessing
    `{video_id}.mp4`) handles the passthrough-transcode case where the
    extension may not be .mp4.
    """
    db = get_mongo()
    doc = db.videos.find_one({"video_id": video_id}, {"_id": 0, "url": 1})
    if doc is None:
        raise HTTPException(status_code=404, detail="video not found")

    url = doc.get("url") or ""
    bucket = storage.video_bucket()
    # The stored URL is `{public_prefix}/{bucket}/{key}` - we only need the key.
    needle = f"/{bucket}/"
    if needle not in url:
        # Fall back to the conventional .mp4 key. Covers smoke-test fixtures
        # and any doc written without going through storage.upload_file().
        key = f"{video_id}.mp4"
    else:
        key = url.split(needle, 1)[1]

    client = get_minio()
    try:
        stat = client.stat_object(bucket, key)
    except Exception:
        raise HTTPException(status_code=404, detail=f"object {bucket}/{key} not found in MinIO")

    size = stat.size
    content_type = stat.content_type or "video/mp4"

    range_header = request.headers.get("range") or request.headers.get("Range")
    rng = _parse_range(range_header, size) if range_header else None

    if rng is None:
        # No Range -> full body, but still advertise Range support so the
        # browser knows it can seek.
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(size),
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(
            _stream_minio(bucket, key, 0, size - 1),
            media_type=content_type,
            headers=headers,
        )

    start, end = rng
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Content-Length": str(end - start + 1),
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(
        _stream_minio(bucket, key, start, end),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=content_type,
        headers=headers,
    )
