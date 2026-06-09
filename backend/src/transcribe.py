"""Transcription pipeline — true code-switching via VAD + per-segment transcription.

Code-switching strategy (language == "auto"):
  1. ffmpeg silencedetect → speech segment timestamps (no new deps).
  2. Each segment extracted to a temp WAV and transcribed independently.
  3. Each segment gets its own language detection from faster-whisper.
  4. Merged with correct time offsets.

When language is explicit (e.g. "en"): single-pass, no VAD overhead.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

# ── VAD via ffmpeg silencedetect ──────────────────────────────────────

def _detect_speech_segments(
    wav_path: str,
    noise_db: int = -30,
    min_silence_ms: int = 500,
) -> list[tuple[float, float]]:
    """Return speech segments as list of (start_s, end_s) using ffmpeg silencedetect."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-i", wav_path,
            "-af", f"silencedetect=n={noise_db}dB:d={min_silence_ms / 1000:.1f}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    stderr = result.stderr

    # Parse: silence_start / silence_end
    silences: list[tuple[str, float]] = []  # (type, time_s)
    for m in re.finditer(r"silence_(start|end):\s*([0-9.]+)", stderr):
        silences.append((m.group(1), float(m.group(2))))

    segments: list[tuple[float, float]] = []
    audio_duration = _get_duration(wav_path)

    if not silences:
        return [(0.0, audio_duration)]

    # Build segments = gaps between silence_end → next silence_start
    prev_end = 0.0
    for kind, t in silences:
        if kind == "start" and t > prev_end + 0.1:
            segments.append((prev_end, t))
        if kind == "end":
            prev_end = t

    # Trailing segment
    if prev_end < audio_duration - 0.1:
        segments.append((prev_end, audio_duration))

    # Fallback: if no segments found, treat whole file as one
    if not segments:
        return [(0.0, audio_duration)]

    return segments


def _get_duration(wav_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            wav_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 3600.0  # safe fallback


def _extract_segment(wav_path: str, start_s: float, end_s: float, out_path: str) -> None:
    """Extract audio segment from WAV via ffmpeg (fast, no re-encode)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", wav_path,
            "-ss", str(start_s),
            "-to", str(end_s),
            "-c", "copy",
            out_path,
        ],
        check=True,
    )


# ── ffmpeg transcode ───────────────────────────────────────────────────

def convert_to_wav(source: str, dest: str) -> None:
    """Convert any audio/video to 16 kHz mono WAV via ffmpeg CLI."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", source,
            "-ar", "16000", "-ac", "1",
            dest,
        ],
        check=True,
    )


# ── Core transcription ─────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    model_name: str = "large-v3-turbo",
    language: str = "auto",
    use_gpu: bool = False,
    prompt: str | None = None,
) -> dict:
    """Transcribe audio with optional per-segment code-switching.

    Args:
        audio_path: 16 kHz mono WAV file.
        model_name: faster-whisper model size.
        language: ISO 639-1 or "auto" for code-switching mode.
        use_gpu: CUDA if True, CPU int8 otherwise.
        prompt: optional vocabulary-bias prompt.

    Returns:
        dict with keys: text, language, segments[{startMs, endMs, text, language}]
    """
    from faster_whisper import WhisperModel

    device, compute_type = ("cuda", "float16") if use_gpu else ("cpu", "int8")
    download_root = os.environ.get("HF_HOME")

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )

    # ── Single-language fast path ──────────────────────────────────
    if language != "auto":
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
            initial_prompt=prompt or None,
            word_timestamps=False,
        )
        segments_out: list[dict] = []
        text_parts: list[str] = []
        for seg in segments_iter:
            t = seg.text.strip()
            segments_out.append({
                "startMs": round(seg.start * 1000),
                "endMs": round(seg.end * 1000),
                "text": t,
                "language": info.language,
            })
            text_parts.append(t)
        return {
            "text": " ".join(text_parts).strip(),
            "language": info.language,
            "model": model_name,
            "segments": segments_out,
        }

    # ── Code-switching path: VAD → per-segment transcribe ──────────
    speech_segments = _detect_speech_segments(audio_path)
    all_segments: list[dict] = []
    all_text: list[str] = []
    dominant_lang = None
    tmp_dir = Path(tempfile.mkdtemp(prefix="stt-seg-"))

    try:
        for i, (start_s, end_s) in enumerate(speech_segments):
            seg_wav = tmp_dir / f"seg-{i:04d}.wav"
            _extract_segment(audio_path, start_s, end_s, str(seg_wav))

            seg_iter, seg_info = model.transcribe(
                str(seg_wav),
                language=None,  # auto-detect per segment
                vad_filter=False,  # already VAD'd
                initial_prompt=prompt or None,
                word_timestamps=False,
            )

            dominant_lang = dominant_lang or seg_info.language

            for seg in seg_iter:
                t = seg.text.strip()
                if not t:
                    continue
                all_segments.append({
                    "startMs": round((start_s + seg.start) * 1000),
                    "endMs": round((start_s + seg.end) * 1000),
                    "text": t,
                    "language": seg_info.language,
                })
                all_text.append(t)

    finally:
        # Cleanup temp segment files
        for f in tmp_dir.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return {
        "text": " ".join(all_text).strip(),
        "language": dominant_lang or "unknown",
        "model": model_name,
        "segments": all_segments,
    }
