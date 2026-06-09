"""Local STT backend — FastAPI + faster-whisper.

Replaces the Node.js + whisper.cpp server. Same API contract, adds code-switching
via per-segment language detection.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .transcribe import convert_to_wav, transcribe

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
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(200 * 1024 * 1024)))

MODEL_REGISTRY = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
}

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Local STT", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)



# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    download_root = os.environ.get("HF_HOME")
    installed: dict[str, bool] = {}
    for key in MODEL_REGISTRY:
        if download_root:
            # faster-whisper caches under models--<name>
            model_dir = Path(download_root) / f"models--Systran--faster-whisper-{key}"
            installed[key] = model_dir.is_dir()
        else:
            installed[key] = False

    ok = any(installed.values())

    return {
        "ok": ok,
        "defaultModel": DEFAULT_MODEL,
        "models": {k: f"faster-whisper {v}" for k, v in MODEL_REGISTRY.items()},
        "installed": installed,
        "useGpu": False,
        "codeSwitching": True,
        "engine": "faster-whisper (CTranslate2)",
    }


# ── Transcribe ─────────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def api_transcribe(
    audio: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form("auto"),
    useGpu: str = Form("0"),
    prompt: str = Form(""),
):
    started_at = time.time()
    req_id = f"{int(started_at * 1000)}-{uuid.uuid4().hex[:8]}"
    original_name = audio.filename or "audio"

    # Save uploaded file
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

        # Convert to 16 kHz mono WAV
        convert_to_wav(str(source_path), str(wav_path))

        # Resolve model
        model_name = model.strip() or DEFAULT_MODEL
        if model_name not in MODEL_REGISTRY:
            return JSONResponse(
                {
                    "error": f"Unsupported model '{model_name}'. "
                    f"Use one of: {', '.join(MODEL_REGISTRY)}"
                },
                status_code=400,
            )

        # Transcribe
        result = transcribe(
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
        # Cleanup temp files (same privacy model as Node version)
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

    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
