"""whisper.cpp engine — spawn whisper-cli with GGML_METAL GPU support.

Used when engine="whisper-cpp". Faster for single-language audio on Apple Silicon
because Metal GPU is available (unlike faster-whisper which is CPU-only outside CUDA).
No code-switching — language is set once per file.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def has_gpu() -> bool:
    """Return True when whisper.cpp was built with a GPU backend."""
    cache = whisper_cpp_dir() / "build" / "CMakeCache.txt"
    if not cache.is_file():
        return False
    text = cache.read_text(errors="ignore")
    return "GGML_METAL:BOOL=ON" in text or "GGML_CUDA:BOOL=ON" in text


def whisper_cpp_dir() -> Path:
    return Path(os.environ.get("WHISPER_CPP_DIR", Path(__file__).resolve().parent.parent / "vendor" / "whisper.cpp"))


def whisper_cli_bin() -> Path:
    """Path to whisper-cli binary."""
    custom = os.environ.get("WHISPER_CPP_BIN")
    if custom:
        return Path(custom)
    base = whisper_cpp_dir()
    return base / "build" / "bin" / "whisper-cli"


def model_path(model_name: str) -> Path:
    """Resolve ggml model path from WHISPER_CPP_MODEL_DIR or default."""
    model_dir = Path(os.environ.get("WHISPER_CPP_MODEL_DIR", whisper_cpp_dir() / "models"))
    return model_dir / f"ggml-{model_name}.bin"


def is_available() -> bool:
    """Check if whisper-cli binary and at least one model are present."""
    bin_path = whisper_cli_bin()
    if not bin_path.is_file():
        return False
    model_dir = Path(os.environ.get("WHISPER_CPP_MODEL_DIR", whisper_cpp_dir() / "models"))
    if model_dir.is_dir():
        if any(model_dir.glob("ggml-*.bin")):
            return True
    return False


def installed_models() -> dict[str, bool]:
    model_dir = Path(os.environ.get("WHISPER_CPP_MODEL_DIR", whisper_cpp_dir() / "models"))
    registry = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
    result: dict[str, bool] = {}
    for name in registry:
        result[name] = (model_dir / f"ggml-{name}.bin").is_file()
    return result


def _words_from_cpp_tokens(seg: dict) -> list[dict]:
    """Merge whisper.cpp `-ojf` sub-word tokens into word-level timestamps.

    whisper.cpp emits BPE tokens: a new word starts with a leading space, and
    continuation pieces have none. Special tokens ([_BEG_], [_TT_123], …) carry
    no real text and are skipped. Offsets are already in ms.
    """
    words: list[dict] = []
    cur: dict | None = None
    for tok in seg.get("tokens", []):
        raw = str(tok.get("text", ""))
        # Skip whisper.cpp special / timestamp tokens.
        if raw.startswith("[_") or (raw.startswith("[") and raw.endswith("]")):
            continue
        piece = raw
        starts_word = piece.startswith(" ") or cur is None
        piece = piece.strip()
        if not piece:
            continue
        off = tok.get("offsets", {}) or {}
        f, t = off.get("from"), off.get("to")
        if starts_word:
            if cur and cur["word"]:
                words.append(cur)
            cur = {"startMs": f, "endMs": t, "word": piece}
        else:
            cur["word"] += piece
            if t is not None:
                cur["endMs"] = t
    if cur and cur["word"]:
        words.append(cur)
    return [w for w in words if w.get("startMs") is not None and w.get("endMs") is not None]


def transcribe_cpp(
    audio_path: str,
    model_name: str = "base",
    language: str = "auto",
    use_gpu: bool = True,
    prompt: str | None = None,
    timeout: int | None = None,
) -> dict:
    """Transcribe via whisper-cli subprocess.

    Args:
        audio_path: 16 kHz mono WAV file.
        model_name: ggml model name (tiny/base/small/large-v3/large-v3-turbo).
        language: ISO 639-1 or "auto".
        use_gpu: if True, omit -ng flag (lets Metal run).
        prompt: optional initial prompt.
        timeout: optional subprocess timeout in seconds. None means no timeout.

    Returns:
        dict with text, language, segments, model keys.
    """
    cli = whisper_cli_bin()
    model = model_path(model_name)
    output_base = audio_path.rsplit(".", 1)[0] + "-whisper-out"

    if not cli.is_file():
        raise FileNotFoundError(f"whisper-cli not found: {cli}")
    if not model.is_file():
        raise FileNotFoundError(f"ggml model not found: {model}")

    args = [
        str(cli),
        "-m", str(model),
        "-f", audio_path,
        "-l", language,
        "-oj", "-ojf",
        "-of", output_base,
        "-np",
    ]
    if not use_gpu:
        args.append("-ng")
    if prompt:
        args.extend(["--prompt", prompt])

    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)

    if proc.returncode != 0:
        raise RuntimeError(f"whisper-cli exited {proc.returncode}: {proc.stderr[:500]}")

    output_json = output_base + ".json"
    raw = json.loads(Path(output_json).read_text())

    segments_out = []
    text_parts = []
    for seg in raw.get("transcription", []):
        t = str(seg.get("text", "")).strip()
        segments_out.append({
            "startMs": seg.get("offsets", {}).get("from"),
            "endMs": seg.get("offsets", {}).get("to"),
            "text": t,
            "language": raw.get("result", {}).get("language", language),
            "words": _words_from_cpp_tokens(seg),
        })
        text_parts.append(t)

    # Cleanup output files
    for ext in (".json", ".wav", ".srt", ".vtt", ".txt"):
        try:
            Path(output_base + ext).unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "text": " ".join(text_parts).strip(),
        "language": raw.get("result", {}).get("language", language),
        "model": raw.get("model", {}).get("type", model_name),
        "segments": segments_out,
    }
