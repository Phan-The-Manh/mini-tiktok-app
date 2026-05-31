"""Ad-hoc smoke check for audio extraction + Whisper transcription (step 3.6).

Synthesizes test videos via ffmpeg's lavfi filters (no fixtures, no
downloads required for the extraction half), then exercises:

  1. `extract_audio` on a video with a real audio track produces a
     non-empty 16 kHz mono PCM WAV.
  2. `extract_audio` on a video with no audio stream returns None
     (legitimate silent input).
  3. `should_transcribe` returns False for a too-short wav and False
     for a silent wav, True for a normal one.
  4. `audio()` context manager cleans up its temp directory on exit.
  5. `AudioExtractError` is raised when the input file is missing.
  6. (Whisper-deps available) `WhisperTranscriber` loads `tiny` and
     returns a string from `transcribe()`. The actual text from
     synthesized tones is unpredictable, so we only assert the type and
     that the call did not raise.

ffmpeg-only cases run when ffmpeg is on PATH. Whisper-dependent cases
skip cleanly with a [WARN] when `openai-whisper` / `torch` are missing
(typical state before `pip install -r content_analyzer/requirements.txt`).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import content_analyzer._path  # noqa: F401

from content_analyzer.services.audio import (
    AudioExtractError,
    audio,
    extract_audio,
    should_transcribe,
    _ffmpeg_bin,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


def _ffmpeg_available() -> bool:
    return shutil.which(_ffmpeg_bin()) is not None


def _make_video_with_tone(dst: Path, duration_seconds: float = 3.0) -> bool:
    """Synthesize a video with a 1 kHz sine-wave audio track."""
    result = subprocess.run(
        [
            _ffmpeg_bin(),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration_seconds}:size=160x120:rate=10",
            "-f", "lavfi",
            "-i", f"sine=frequency=1000:duration={duration_seconds}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-shortest",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _make_video_no_audio(dst: Path, duration_seconds: float = 3.0) -> bool:
    """Synthesize a video with no audio stream at all."""
    result = subprocess.run(
        [
            _ffmpeg_bin(),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration_seconds}:size=160x120:rate=10",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _make_silent_wav(dst: Path, duration_seconds: float = 2.0) -> bool:
    """Synthesize a fully-silent 16 kHz mono WAV (anullsrc)."""
    result = subprocess.run(
        [
            _ffmpeg_bin(),
            "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=mono:sample_rate=16000:duration={duration_seconds}",
            "-acodec", "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _is_riff_wav(path: Path) -> bool:
    with path.open("rb") as f:
        header = f.read(12)
    return header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def _whisper_deps_available() -> bool:
    try:
        import whisper  # noqa: F401
        import torch    # noqa: F401
    except Exception as e:
        log.warning("[WARN] whisper deps unavailable: %s", e)
        return False
    return True


def main() -> int:
    if not _ffmpeg_available():
        log.warning("[WARN] ffmpeg not on PATH; skipping audio smoke test")
        return 0

    workdir = Path(tempfile.mkdtemp(prefix="smoke_audio_"))
    try:
        # --- Generate test inputs ---
        tone_video = workdir / "tone.mp4"
        if not _make_video_with_tone(tone_video, duration_seconds=3.0):
            log.error("[FAIL] could not synthesize tone video")
            return 1
        log.info("[OK] synthesized tone video (%d bytes)", tone_video.stat().st_size)

        # === Case 1: extract from a video with an audio track ===
        wav_out = workdir / "tone.wav"
        produced = extract_audio(tone_video, wav_out)
        assert produced == wav_out, f"expected {wav_out}, got {produced}"
        assert wav_out.exists() and wav_out.stat().st_size > 0
        assert _is_riff_wav(wav_out), "extracted file is not a RIFF/WAVE container"
        log.info("[OK] extracted wav %s (%d bytes, RIFF/WAVE)",
                 wav_out.name, wav_out.stat().st_size)

        # === Case 2: video without audio -> None ===
        no_audio_video = workdir / "silent_video.mp4"
        if _make_video_no_audio(no_audio_video, duration_seconds=2.0):
            result = extract_audio(no_audio_video, workdir / "should_not_exist.wav")
            assert result is None, (
                f"video with no audio should return None, got {result}"
            )
            log.info("[OK] video with no audio stream returned None")
        else:
            log.warning("[WARN] could not synthesize no-audio video; skipping case 2")

        # === Case 3a: normal wav -> should_transcribe True ===
        assert should_transcribe(wav_out) is True, (
            "tone wav should pass should_transcribe gate"
        )
        log.info("[OK] tone wav passes should_transcribe gate")

        # === Case 3b: too-short wav -> False ===
        short_wav = workdir / "short.wav"
        if _make_silent_wav(short_wav, duration_seconds=0.1):
            # 0.1s is below MIN_AUDIO_SECONDS default (0.5); also silent.
            assert should_transcribe(short_wav) is False, (
                "0.1s wav should fail should_transcribe gate"
            )
            log.info("[OK] 0.1s wav fails should_transcribe gate")

        # === Case 3c: full-length silent wav -> False (silence) ===
        silent_wav = workdir / "silent.wav"
        if _make_silent_wav(silent_wav, duration_seconds=2.0):
            assert should_transcribe(silent_wav) is False, (
                "fully-silent wav should fail should_transcribe gate"
            )
            log.info("[OK] silent 2s wav fails should_transcribe gate")

        # === Case 4: context manager cleans up ===
        with audio(tone_video) as wav:
            assert wav is not None, "tone video should yield a wav path"
            tmpdir = wav.parent
            assert tmpdir.exists(), "tmpdir should exist inside with-block"
        assert not tmpdir.exists(), (
            f"tmpdir should have been removed on exit: {tmpdir}"
        )
        log.info("[OK] audio() context manager cleaned up %s", tmpdir)

        # === Case 5: missing video raises AudioExtractError ===
        try:
            extract_audio(workdir / "does_not_exist.mp4", workdir / "nope.wav")
        except AudioExtractError as e:
            log.info("[OK] missing video raised AudioExtractError: %s",
                     str(e).splitlines()[0])
        else:
            log.error("[FAIL] expected AudioExtractError for missing video")
            return 1

        # === Case 6: Whisper transcribe round-trip (deps-dependent) ===
        if not _whisper_deps_available():
            log.warning(
                "[WARN] skipping Whisper transcribe case "
                "(pip install -r content_analyzer/requirements.txt)"
            )
        else:
            from content_analyzer.services.audio import WhisperTranscriber

            tx = WhisperTranscriber()
            tx.load()
            text = tx.transcribe(wav_out)
            assert isinstance(text, str), f"transcribe() returned {type(text)}"
            log.info("[OK] Whisper.transcribe returned a string of length %d", len(text))

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    log.info("[OK] all audio cases passed; cleaned up %s", workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
