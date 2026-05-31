"""
audio.py — ffmpeg audio extract + Whisper-tiny transcribe
=========================================================

Step 3.6 of the Content Analyzer build.

Two responsibilities, separated so each can be tested in isolation:

1. `extract_audio(video_path, dest)` / `audio(video_path)`
       ffmpeg pulls a mono 16 kHz PCM WAV out of the input file. 16 kHz
       mono is exactly what Whisper expects internally; doing the
       resample ourselves means Whisper does no extra work and the wav
       on disk is small (~32 KB/s).

2. `WhisperTranscriber.transcribe(wav_path) -> str`
       Lazy-loaded `openai-whisper` model (default `tiny`, ~39M params,
       CPU-friendly). Returns the plain transcript text.

Cleanly-skipped audio
---------------------
A video can be legitimately silent (no mic, music-only with the mic
muted, etc.). In those cases Whisper is prone to hallucinating
phrases like "Thanks for watching" or "Subtitles by ...". We pre-empt
that by returning an empty transcript when any of these hold:

- The input has no audio stream at all (ffprobe).
- The extracted wav is shorter than `MIN_AUDIO_SECONDS` (default 0.5).
- The wav's peak volume is below `SILENCE_DBFS` (default -45 dB),
  measured with ffmpeg's `volumedetect` filter.

The downstream text encoder (step 3.7) treats an empty transcript as
"caption only" — there is no error path for silent audio.

ffmpeg/ffprobe binaries are resolved the same way `frames.py` does:
`FFMPEG_BIN` env var, with `ffprobe` derived alongside.

Lifecycle
---------
The whisper-tiny weight file is ~75 MB and the first load takes a few
seconds. As with the CLIP encoder, we pay that once at worker startup
via an explicit `load()`, then call `transcribe()` per video.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env


log = logging.getLogger(__name__)


# --- Errors ------------------------------------------------------------------

class AudioExtractError(RuntimeError):
    """Raised when ffmpeg cannot produce a wav from the input video."""


class TranscriptionError(RuntimeError):
    """Raised when the Whisper backend fails on an otherwise-valid wav."""


# --- Tunables (env-backed) ---------------------------------------------------

def _ffmpeg_bin() -> str:
    return os.getenv("FFMPEG_BIN", "ffmpeg")


def _ffprobe_bin() -> str:
    """ffprobe ships alongside ffmpeg; derive its path from FFMPEG_BIN."""
    ffmpeg = _ffmpeg_bin()
    if ffmpeg.endswith("ffmpeg"):
        return ffmpeg[: -len("ffmpeg")] + "ffprobe"
    if ffmpeg.endswith("ffmpeg.exe"):
        return ffmpeg[: -len("ffmpeg.exe")] + "ffprobe.exe"
    return "ffprobe"


def _min_audio_seconds() -> float:
    try:
        return float(os.getenv("ANALYZER_MIN_AUDIO_SECONDS", "0.5"))
    except ValueError:
        return 0.5


def _silence_dbfs() -> float:
    """Peak-volume threshold (dBFS). Wavs whose `max_volume` is below this
    are treated as silent. -45 dB is conservative: a quiet voice still
    measures around -25 to -15 dB on a phone mic."""
    try:
        return float(os.getenv("ANALYZER_SILENCE_DBFS", "-45.0"))
    except ValueError:
        return -45.0


def _whisper_model_name() -> str:
    return os.getenv("WHISPER_MODEL", "tiny")


# --- ffprobe helpers ---------------------------------------------------------

def _has_audio_stream(video_path: Path) -> bool:
    """Return True if `video_path` contains at least one audio stream."""
    try:
        result = subprocess.run(
            [
                _ffprobe_bin(),
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # ffprobe missing — be permissive; ffmpeg will report the real error
        # if there genuinely is no audio.
        return True
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def _wav_duration_seconds(wav_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                _ffprobe_bin(),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 0.0
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


_MAX_VOL_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def _peak_dbfs(wav_path: Path) -> float:
    """Return the peak (max_volume) in dBFS for the wav, or 0.0 if it could
    not be measured. ffmpeg writes the value to stderr under `volumedetect`.
    A value of 0.0 means "full-scale" (loud); negative values are quieter."""
    try:
        result = subprocess.run(
            [
                _ffmpeg_bin(),
                "-hide_banner",
                "-nostats",
                "-i", str(wav_path),
                "-af", "volumedetect",
                "-vn",
                "-sn",
                "-dn",
                "-f", "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 0.0
    # volumedetect output goes to stderr regardless of exit code.
    match = _MAX_VOL_RE.search(result.stderr or "")
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


# --- Audio extraction --------------------------------------------------------

def extract_audio(video_path: str | Path, dest: str | Path) -> Optional[Path]:
    """Pull a mono 16 kHz PCM WAV out of `video_path` to `dest`.

    Returns the path to the wav on success, or `None` if the input has
    no audio stream (a legitimate "silent video" — the caller treats
    this as an empty transcript). Raises `AudioExtractError` for actual
    ffmpeg failures.
    """
    video_path = Path(video_path)
    dest = Path(dest)

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise AudioExtractError(f"video file missing or empty: {video_path}")

    if not _has_audio_stream(video_path):
        log.info("[INFO] no audio stream in %s — skipping transcription", video_path)
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                _ffmpeg_bin(),
                "-y",
                "-i", str(video_path),
                "-vn",                # drop video
                "-ac", "1",           # mono
                "-ar", "16000",       # 16 kHz (Whisper's native rate)
                "-acodec", "pcm_s16le",
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise AudioExtractError(f"ffmpeg not found: {e}") from e

    if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise AudioExtractError(
            f"ffmpeg failed to extract audio from {video_path} "
            f"(rc={result.returncode}): {(result.stderr or '').strip()[:200]}"
        )
    return dest


@contextmanager
def audio(video_path: str | Path) -> Iterator[Optional[Path]]:
    """Extract audio into a fresh temp dir; clean it up on exit.

    Yields the wav path, or `None` if the video has no audio stream.

    Usage:
        with audio(video_path) as wav:
            transcript = transcribe(wav) if wav else ""
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="content_analyzer_audio_"))
    try:
        wav_path = tmpdir / "audio.wav"
        yield extract_audio(video_path, wav_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Whisper transcription ---------------------------------------------------

class WhisperTranscriber:
    """Thin wrapper around `openai-whisper` for CPU-only transcription.

    Lazy: `whisper` and `torch` are only imported inside `load()` so other
    modules in this package can be unit-tested without paying the import
    cost (or even having the packages installed for non-ML paths).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ):
        self.model_name = model_name or _whisper_model_name()
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load Whisper weights. Idempotent."""
        if self._model is not None:
            return

        import whisper  # type: ignore[import-not-found]

        log.info("[INFO] loading Whisper model %s on %s", self.model_name, self.device)
        self._model = whisper.load_model(self.model_name, device=self.device)
        log.info("[OK] Whisper loaded (%s)", self.model_name)

    def transcribe(self, wav_path: str | Path) -> str:
        """Transcribe a 16 kHz mono wav. Returns the plain text (stripped).

        Empty wavs / unreadable inputs raise `TranscriptionError`. Use the
        `should_transcribe` gate before calling to avoid wasting time on
        silent inputs.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise TranscriptionError(f"wav missing or empty: {wav_path}")

        if self._model is None:
            self.load()

        assert self._model is not None
        try:
            result = self._model.transcribe(
                str(wav_path),
                fp16=False,       # CPU-only path
                language=None,    # autodetect
            )
        except Exception as e:
            raise TranscriptionError(f"whisper.transcribe failed: {e}") from e

        text = (result.get("text") or "").strip()
        return text


# --- Silence / duration gate -------------------------------------------------

def should_transcribe(wav_path: str | Path) -> bool:
    """Return True if the wav is worth sending to Whisper.

    False when the wav is too short (under `ANALYZER_MIN_AUDIO_SECONDS`)
    or too quiet (peak below `ANALYZER_SILENCE_DBFS`). The caller treats
    a False result as an empty transcript — no error, no Whisper call,
    no hallucinated "Thanks for watching".
    """
    wav_path = Path(wav_path)
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return False

    dur = _wav_duration_seconds(wav_path)
    if dur < _min_audio_seconds():
        log.info("[INFO] wav too short (%.2fs < %.2fs) — empty transcript",
                 dur, _min_audio_seconds())
        return False

    peak = _peak_dbfs(wav_path)
    if peak <= _silence_dbfs():
        log.info("[INFO] wav peak %.1f dBFS <= %.1f dBFS — silent, empty transcript",
                 peak, _silence_dbfs())
        return False

    return True


# --- High-level convenience --------------------------------------------------

def transcribe_video(
    video_path: str | Path,
    transcriber: "WhisperTranscriber | None" = None,
) -> str:
    """End-to-end: extract audio, gate on silence/duration, run Whisper.

    Returns the transcript text (possibly empty). Never raises on silent
    or audio-less videos — that's a legitimate empty transcript. Raises
    `AudioExtractError` / `TranscriptionError` for genuine failures.
    """
    with audio(video_path) as wav:
        if wav is None:
            return ""
        if not should_transcribe(wav):
            return ""
        tx = transcriber or get_transcriber()
        return tx.transcribe(wav)


# --- Module-level singleton --------------------------------------------------

_transcriber: Optional[WhisperTranscriber] = None


def get_transcriber() -> WhisperTranscriber:
    """Return a process-wide, lazily-loaded transcriber.

    Production startup should instead instantiate the transcriber
    explicitly in `main.py` (step 3.10) and inject it into the consumer
    handler, so load-time errors fire at startup, not on the first
    event delivery.
    """
    global _transcriber
    if _transcriber is None:
        tx = WhisperTranscriber()
        tx.load()
        _transcriber = tx
    return _transcriber
