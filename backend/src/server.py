"""Local STT backend — FastAPI + dual engine (faster-whisper / whisper.cpp).

Two engines:
  - faster-whisper: code-switching (per-segment language detect), CPU only.
  - whisper-cpp:    Metal GPU (Apple Silicon), single language, faster for fixed lang.

Select via engine param in /api/transcribe.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .transcribe import convert_to_wav, transcribe as faster_transcribe
from .whisper_cpp import is_available as cpp_available, installed_models as cpp_models
from .whisper_cpp import transcribe_cpp

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", ROOT / "uploads"))
TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", ROOT / "transcripts"))
PUBLIC_DIR = ROOT / "public"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", "6601"))
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
DEFAULT_ENGINE = os.environ.get("STT_ENGINE", "faster-whisper")
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(200 * 1024 * 1024)))

MODEL_REGISTRY = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]

# faster-whisper HF repo owner differs by model
_FW_REPO = {k: "Systran" for k in MODEL_REGISTRY}
_FW_REPO["large-v3-turbo"] = "mobiuslabsgmbh"

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Local STT", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    hf_root = os.environ.get("HF_HOME")
    fw_installed: dict[str, bool] = {}
    for key in MODEL_REGISTRY:
        if hf_root:
            org = _FW_REPO[key]
            d = Path(hf_root) / f"models--{org}--faster-whisper-{key}"
            fw_installed[key] = d.is_dir()
        else:
            fw_installed[key] = False

    cpp_ok = cpp_available()
    cpp_installed = cpp_models() if cpp_ok else {}

    engines = {
        "faster-whisper": {
            "available": any(fw_installed.values()),
            "gpu": "CUDA only (CPU on Mac)",
            "codeSwitching": True,
            "installed": fw_installed,
        },
        "whisper-cpp": {
            "available": cpp_ok,
            "gpu": "Metal (Apple Silicon)" if cpp_ok else "unavailable",
            "codeSwitching": False,
            "installed": cpp_installed,
        },
    }

    return {
        "ok": engines["faster-whisper"]["available"] or engines["whisper-cpp"]["available"],
        "defaultModel": DEFAULT_MODEL,
        "defaultEngine": DEFAULT_ENGINE,
        "engines": engines,
    }


# ── Transcribe ─────────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def api_transcribe(
    audio: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form("auto"),
    engine: str = Form(""),
    useGpu: str = Form("0"),
    prompt: str = Form(""),
):
    started_at = time.time()
    req_id = f"{int(started_at * 1000)}-{uuid.uuid4().hex[:8]}"
    original_name = audio.filename or "audio"

    source_path = UPLOAD_DIR / f"{req_id}_{original_name}"
    wav_path = UPLOAD_DIR / f"{req_id}.wav"

    try:
        content = await audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            return JSONResponse(
                {"error": f"File too large (max {MAX_AUDIO_BYTES} bytes)"},
                status_code=413,
            )
        source_path.write_bytes(content)
        convert_to_wav(str(source_path), str(wav_path))

        model_name = model.strip() or DEFAULT_MODEL
        if model_name not in MODEL_REGISTRY:
            return JSONResponse(
                {"error": f"Unknown model '{model_name}'. Use: {', '.join(MODEL_REGISTRY)}"},
                status_code=400,
            )

        engine_name = engine.strip() or DEFAULT_ENGINE

        if engine_name == "whisper-cpp":
            result = transcribe_cpp(
                str(wav_path),
                model_name=model_name,
                language=language.strip() or "auto",
                use_gpu=useGpu == "1",
                prompt=prompt.strip() or None,
            )
        else:
            result = faster_transcribe(
                str(wav_path),
                model_name=model_name,
                language=language.strip() or "auto",
                use_gpu=useGpu == "1",
                prompt=prompt.strip() or None,
            )

        duration_ms = int((time.time() - started_at) * 1000)

        return {
            "id": req_id,
            "text": result["text"],
            "language": result["language"],
            "model": result["model"],
            "engine": engine_name or "faster-whisper",
            "durationMs": duration_ms,
            "segments": result["segments"],
            "files": {"originalName": original_name},
        }

    except Exception as exc:
        return JSONResponse(
            {"error": str(exc), "originalName": original_name},
            status_code=500,
        )

    finally:
        _safe_unlink(source_path)
        _safe_unlink(wav_path)


# Mount static UI AFTER routes so API endpoints take precedence.
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")


# ── Helpers ────────────────────────────────────────────────────────────
def _safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


# ── Entrypoint ─────────────────────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
