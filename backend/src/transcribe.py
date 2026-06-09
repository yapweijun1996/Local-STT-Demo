"""Transcription pipeline — VAD + faster-whisper with per-segment language detection.

Code-switching strategy:
  - VAD splits audio into speech segments (via faster-whisper's built-in Silero VAD).
  - When language == "auto": each segment's text is classified with langdetect so the
    response carries per-segment language tags.
  - The underlying Whisper large-v3 model handles multilingual tokens natively; the
    per-segment tag is a lightweight post-hoc classification, not a re-transcribe.
  - When language is explicit (e.g. "en"): single-pass, no per-segment overhead.
"""

from __future__ import annotations

import os
import subprocess

from langdetect import DetectorFactory, detect as langdetect_detect

# Deterministic language detection
DetectorFactory.seed = 0


def resolve_device(use_gpu: bool = False) -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper."""
    if use_gpu:
        return ("cuda", "float16")
    return ("cpu", "int8")


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


def detect_language(text: str, fallback: str | None = None) -> str | None:
    """Classify a text snippet's language. Returns ISO 639-1 code or fallback."""
    if not text or len(text.strip()) < 3:
        return fallback
    try:
        return langdetect_detect(text)
    except Exception:
        return fallback


def transcribe(
    audio_path: str,
    model_name: str = "large-v3-turbo",
    language: str = "auto",
    use_gpu: bool = False,
    prompt: str | None = None,
) -> dict:
    """Run transcription with optional per-segment language tagging.

    Args:
        audio_path: 16 kHz mono WAV file.
        model_name: faster-whisper model size (tiny / base / small / large-v3 / large-v3-turbo).
        language: ISO 639-1 code or "auto".
        use_gpu: if True, attempt CUDA; otherwise CPU int8.
        prompt: optional initial prompt for vocabulary bias.

    Returns:
        dict with keys: text, language, segments[{startMs, endMs, text, language}], info
    """
    # Lazy import so the module is importable even before pip install
    from faster_whisper import WhisperModel

    device, compute_type = resolve_device(use_gpu)
    download_root = os.environ.get("HF_HOME")

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )

    lang_arg = None if language == "auto" else language

    segments_iter, info = model.transcribe(
        audio_path,
        language=lang_arg,
        vad_filter=True,
        initial_prompt=prompt or None,
        word_timestamps=False,
    )

    detected_lang: str = info.language

    segments_out: list[dict] = []
    text_parts: list[str] = []

    for seg in segments_iter:
        text = seg.text.strip()
        seg_lang = (
            detect_language(text, fallback=detected_lang)
            if language == "auto"
            else detected_lang
        )
        segments_out.append(
            {
                "startMs": round(seg.start * 1000),
                "endMs": round(seg.end * 1000),
                "text": text,
                "language": seg_lang,
            }
        )
        text_parts.append(text)

    return {
        "text": " ".join(text_parts).strip(),
        "language": detected_lang,
        "model": model_name,
        "segments": segments_out,
    }
