"""Local STT backend — FastAPI + Redis queue + dual STT engine.

POST /api/transcribe saves the upload, creates a Redis-backed job, and returns a
jobId immediately. Internal workers consume a FIFO Redis queue so long
transcriptions avoid Cloudflare 524 and cannot exceed the configured concurrency.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import diarize as diarization
from .transcribe import convert_to_wav, transcribe as faster_transcribe
from .whisper_cpp import has_gpu as cpp_has_gpu
from .whisper_cpp import installed_models as cpp_models
from .whisper_cpp import is_available as cpp_available
from .whisper_cpp import transcribe_cpp

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", ROOT / "uploads"))
TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", ROOT / "transcripts"))
PUBLIC_DIR = ROOT / "public"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", "6601"))
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEFAULT_ENGINE = os.environ.get("STT_ENGINE", "whisper-cpp")
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(200 * 1024 * 1024)))
TRANSCRIPTION_TIMEOUT = int(os.environ.get("TRANSCRIPTION_TIMEOUT", "0"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
JOB_TTL = int(os.environ.get("JOB_TTL", str(12 * 60 * 60)))
WORKER_CONCURRENCY = max(1, int(os.environ.get("WORKER_CONCURRENCY", "2")))
LEADER_TTL = int(os.environ.get("WORKER_LEADER_TTL", "30"))
QUEUE_POP_TIMEOUT = int(os.environ.get("QUEUE_POP_TIMEOUT", "5"))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", str(30 * 60)))
ORPHAN_MAX_AGE_SECONDS = int(os.environ.get("ORPHAN_MAX_AGE_SECONDS", str(24 * 60 * 60)))
UPLOADS_MAX_BYTES = int(os.environ.get("UPLOADS_MAX_BYTES", str(5 * 1024 * 1024 * 1024)))
ENABLE_CHUNK_TRANSCRIPTION = os.environ.get("ENABLE_CHUNK_TRANSCRIPTION", "1") != "0"
CHUNK_SECONDS = max(1, int(os.environ.get("CHUNK_SECONDS", "180")))
CHUNK_OVERLAP_SECONDS = max(0, int(os.environ.get("CHUNK_OVERLAP_SECONDS", "5")))
# Speaker diarization (pyannote). Off by default per-request; this env is the
# server-wide kill switch. Requires pyannote.audio + HF_TOKEN (see diarize.py).
ENABLE_DIARIZATION = os.environ.get("ENABLE_DIARIZATION", "1") != "0"

MODEL_REGISTRY = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
_FW_REPO = {k: "Systran" for k in MODEL_REGISTRY}
_FW_REPO["large-v3-turbo"] = "mobiuslabsgmbh"

# ── Redis keys ─────────────────────────────────────────────────────────
JOB_PREFIX = "stt:job:"
PENDING_QUEUE = "stt:queue:pending"
RUNNING_SET = "stt:queue:running"
LEADER_KEY = "stt:worker:leader"

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Local STT", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ── Runtime state ──────────────────────────────────────────────────────
_redis = None
_leader_token = f"{os.getpid()}-{uuid.uuid4().hex}"
_worker_tasks: list[asyncio.Task] = []
_leader_task: asyncio.Task | None = None
_cleanup_task: asyncio.Task | None = None


def _init_redis() -> None:
    global _redis
    try:
        import redis as _redis_lib

        client = _redis_lib.Redis.from_url(
            REDIS_URL,
            socket_connect_timeout=2,
            socket_timeout=10,
            decode_responses=True,
        )
        client.ping()
        _redis = client
        log.info("Job store: Redis at %s", REDIS_URL)
    except Exception as exc:
        log.error("Redis unavailable (%s) — queue disabled", exc)
        _redis = None


def _job_key(job_id: str) -> str:
    return f"{JOB_PREFIX}{job_id}"


def _require_redis_response():
    if _redis is None:
        return JSONResponse(
            {"error": "Redis unavailable. STT queue is disabled."},
            status_code=503,
        )
    return None


def _job_get(job_id: str) -> dict[str, Any] | None:
    if _redis is None:
        return None
    raw = _redis.get(_job_key(job_id))
    if not raw:
        return None
    return json.loads(raw)


def _job_set(job_id: str, data: dict[str, Any]) -> None:
    if _redis is None:
        raise RuntimeError("Redis unavailable")
    _redis.setex(_job_key(job_id), JOB_TTL, json.dumps(data))


def _job_update(job_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    current = _job_get(job_id) or {}
    current.update(patch)
    current["updatedAt"] = time.time()
    _job_set(job_id, current)
    return current


def _queue_counts() -> tuple[int, int]:
    if _redis is None:
        return (0, 0)
    return (int(_redis.llen(PENDING_QUEUE)), int(_redis.scard(RUNNING_SET)))


def _queue_position(job_id: str) -> int | None:
    if _redis is None:
        return None
    pending = _redis.lrange(PENDING_QUEUE, 0, -1)
    try:
        return pending.index(job_id) + 1
    except ValueError:
        return None


def _enqueue_job(job_id: str, *, status: str = "queued") -> None:
    if _redis is None:
        raise RuntimeError("Redis unavailable")
    _redis.lrem(PENDING_QUEUE, 0, job_id)
    _redis.srem(RUNNING_SET, job_id)
    _redis.rpush(PENDING_QUEUE, job_id)
    _job_update(job_id, {"status": status, "queuedAt": time.time()})


def _uploads_bytes() -> int:
    total = 0
    for p in UPLOAD_DIR.iterdir():
        if p.is_file() and p.name != ".gitkeep":
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _active_upload_paths() -> set[Path]:
    active: set[Path] = set()
    if _redis is None:
        return active
    job_ids = set(_redis.lrange(PENDING_QUEUE, 0, -1)) | set(_redis.smembers(RUNNING_SET))
    for job_id in job_ids:
        job = _job_get(job_id)
        if not job:
            continue
        for key in ("sourcePath", "wavPath", "chunkDir"):
            path = job.get(key)
            if path:
                active.add(Path(path).resolve())
    return active


def _cleanup_orphan_uploads(*, force: bool = False) -> dict[str, int]:
    now = time.time()
    active = _active_upload_paths()
    deleted_files = 0
    deleted_bytes = 0

    for path in UPLOAD_DIR.iterdir():
        if path.name == ".gitkeep":
            continue
        try:
            stat = path.stat()
            is_active = path.resolve() in active
            is_old = (now - stat.st_mtime) > ORPHAN_MAX_AGE_SECONDS
            if is_active or (not force and not is_old):
                continue
            size = stat.st_size
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            deleted_files += 1
            deleted_bytes += size
        except OSError:
            continue

    if deleted_files:
        log.info("Cleaned %s orphan upload files (%s bytes)", deleted_files, deleted_bytes)
    return {"deletedFiles": deleted_files, "deletedBytes": deleted_bytes}


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def _safe_rmtree(p: Path) -> None:
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _cleanup_job_files(job: dict[str, Any] | None) -> None:
    if not job:
        return
    for key in ("sourcePath", "wavPath"):
        value = job.get(key)
        if value:
            _safe_unlink(Path(value))
    chunk_dir = job.get("chunkDir")
    if chunk_dir:
        _safe_rmtree(Path(chunk_dir))


def _enforce_upload_capacity() -> JSONResponse | None:
    if _uploads_bytes() <= UPLOADS_MAX_BYTES:
        return None
    _cleanup_orphan_uploads(force=True)
    used = _uploads_bytes()
    if used <= UPLOADS_MAX_BYTES:
        return None
    return JSONResponse(
        {
            "error": "Upload storage is full. Try again after current jobs finish.",
            "uploadsBytes": used,
            "uploadsMaxBytes": UPLOADS_MAX_BYTES,
        },
        status_code=507,
    )


def _acquire_leader() -> bool:
    if _redis is None:
        return False
    return bool(_redis.set(LEADER_KEY, _leader_token, nx=True, ex=LEADER_TTL))


def _is_leader() -> bool:
    return _redis is not None and _redis.get(LEADER_KEY) == _leader_token


async def _refresh_leader_loop() -> None:
    while _is_leader():
        _redis.expire(LEADER_KEY, LEADER_TTL)
        await asyncio.sleep(max(1, LEADER_TTL // 2))


def _recover_jobs() -> None:
    if _redis is None:
        return
    recovered = 0
    failed = 0
    _redis.delete(RUNNING_SET)

    for key in _redis.scan_iter(f"{JOB_PREFIX}*"):
        raw = _redis.get(key)
        if not raw:
            continue
        job = json.loads(raw)
        job_id = str(job.get("id") or key.removeprefix(JOB_PREFIX))
        status = job.get("status")
        if status not in {"queued", "running", "pending"}:
            continue
        source_path = Path(str(job.get("sourcePath", "")))
        if source_path.is_file():
            _enqueue_job(job_id, status="queued")
            recovered += 1
        else:
            _job_update(job_id, {
                "status": "error",
                "error": "Source audio missing after restart.",
            })
            failed += 1

    if recovered or failed:
        log.info("Queue recovery complete: %s recovered, %s failed", recovered, failed)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.to_thread(_cleanup_orphan_uploads)
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


async def _worker_loop(worker_id: int) -> None:
    while True:
        if not _is_leader():
            await asyncio.sleep(1)
            continue

        try:
            item = await asyncio.to_thread(_redis.blpop, PENDING_QUEUE, QUEUE_POP_TIMEOUT)
        except Exception as exc:
            log.warning("Worker %s queue pop failed: %s", worker_id, exc)
            await asyncio.sleep(2)
            continue

        if not item:
            continue

        _, job_id = item
        job = _job_get(job_id)
        if not job:
            _redis.srem(RUNNING_SET, job_id)
            continue

        source_path = Path(str(job.get("sourcePath", "")))
        if not source_path.is_file():
            _job_update(job_id, {
                "status": "error",
                "error": "Source audio missing before transcription.",
            })
            _redis.srem(RUNNING_SET, job_id)
            continue

        _redis.sadd(RUNNING_SET, job_id)
        _job_update(job_id, {
            "status": "running",
            "workerId": worker_id,
            "runStartedAt": time.time(),
        })

        try:
            result = await asyncio.to_thread(_do_transcription, job_id)
            _job_update(job_id, {"status": "done", "result": result})
        except Exception as exc:
            _job_update(job_id, {"status": "error", "error": str(exc)})
        finally:
            latest = _job_get(job_id)
            _redis.srem(RUNNING_SET, job_id)
            _cleanup_job_files(latest)


@app.on_event("startup")
async def _startup() -> None:
    global _leader_task, _cleanup_task

    _init_redis()
    if _redis is None:
        return

    await asyncio.to_thread(_cleanup_orphan_uploads)

    if _acquire_leader():
        log.info("Acquired STT queue leader lock; starting %s workers", WORKER_CONCURRENCY)
        _recover_jobs()
        for i in range(WORKER_CONCURRENCY):
            _worker_tasks.append(asyncio.create_task(_worker_loop(i + 1)))
        _leader_task = asyncio.create_task(_refresh_leader_loop())
        _cleanup_task = asyncio.create_task(_cleanup_loop())
    else:
        log.info("Another STT process owns the worker leader lock")


@app.on_event("shutdown")
async def _shutdown() -> None:
    for task in [*_worker_tasks, _leader_task, _cleanup_task]:
        if task is not None:
            task.cancel()
    if _is_leader():
        _redis.delete(LEADER_KEY)


# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    hf_root = os.environ.get("HF_HOME")
    fw_installed: dict[str, bool] = {}
    for key in MODEL_REGISTRY:
        if hf_root:
            org = _FW_REPO[key]
            d = Path(hf_root) / f"models--{org}--faster-whisper-{key}"
            snapshots = d / "snapshots"
            fw_installed[key] = (
                any((snap / "model.bin").is_file() for snap in snapshots.iterdir())
                if snapshots.is_dir()
                else False
            )
        else:
            fw_installed[key] = False

    cpp_ok = cpp_available()
    cpp_installed = cpp_models() if cpp_ok else {}
    queued_count, running_count = _queue_counts()

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
        "ok": (
            _redis is not None
            and (engines["faster-whisper"]["available"] or engines["whisper-cpp"]["available"])
        ),
        "defaultModel": DEFAULT_MODEL,
        "defaultEngine": DEFAULT_ENGINE,
        "jobStore": "redis" if _redis is not None else "unavailable",
        "jobTtlSeconds": JOB_TTL,
        "transcriptionTimeoutSeconds": TRANSCRIPTION_TIMEOUT,
        "workerConcurrency": WORKER_CONCURRENCY,
        "workerLeader": _is_leader(),
        "queuedCount": queued_count,
        "runningCount": running_count,
        "uploadsBytes": _uploads_bytes(),
        "uploadsMaxBytes": UPLOADS_MAX_BYTES,
        "chunkTranscriptionEnabled": ENABLE_CHUNK_TRANSCRIPTION,
        "chunkSeconds": CHUNK_SECONDS,
        "chunkOverlapSeconds": CHUNK_OVERLAP_SECONDS,
        "engines": engines,
        "diarization": {"enabled": ENABLE_DIARIZATION, **diarization.status()},
    }


# ── Transcribe: accept upload, queue job, return immediately ───────────
@app.post("/api/transcribe")
async def api_transcribe(
    audio: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form("auto"),
    engine: str = Form(""),
    useGpu: str = Form("1"),
    prompt: str = Form(""),
    diarize: str = Form("0"),
):
    redis_error = _require_redis_response()
    if redis_error is not None:
        return redis_error

    storage_error = _enforce_upload_capacity()
    if storage_error is not None:
        return storage_error

    started_at = time.time()
    job_id = f"{int(started_at * 1000)}-{uuid.uuid4().hex[:8]}"
    original_name = audio.filename or "audio"
    source_path = UPLOAD_DIR / f"{job_id}_{original_name}"
    wav_path = UPLOAD_DIR / f"{job_id}.wav"

    try:
        content = await audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            return JSONResponse(
                {"error": f"File too large (max {MAX_AUDIO_BYTES} bytes)"},
                status_code=413,
            )
        source_path.write_bytes(content)
    except Exception as exc:
        return JSONResponse(
            {"error": str(exc), "originalName": original_name},
            status_code=500,
        )

    if _uploads_bytes() > UPLOADS_MAX_BYTES:
        _safe_unlink(source_path)
        return JSONResponse(
            {
                "error": "Upload storage limit would be exceeded. Try again after current jobs finish.",
                "uploadsBytes": _uploads_bytes(),
                "uploadsMaxBytes": UPLOADS_MAX_BYTES,
            },
            status_code=507,
        )

    model_name = model.strip() or DEFAULT_MODEL
    if model_name not in MODEL_REGISTRY:
        _safe_unlink(source_path)
        return JSONResponse(
            {"error": f"Unknown model '{model_name}'. Use: {', '.join(MODEL_REGISTRY)}"},
            status_code=400,
        )

    engine_name = engine.strip() or DEFAULT_ENGINE
    if engine_name == "whisper-cpp":
        if useGpu != "1":
            _safe_unlink(source_path)
            return JSONResponse(
                {
                    "error": "whisper-cpp requires GPU/Metal in this deployment. "
                    "Set useGpu=1 (CPU fallback is disabled).",
                },
                status_code=400,
            )
        if not cpp_has_gpu():
            _safe_unlink(source_path)
            return JSONResponse(
                {
                    "error": "whisper-cpp is not GPU-enabled in this build. "
                    "Rebuild whisper.cpp with Metal/CUDA or switch to faster-whisper.",
                },
                status_code=400,
            )

    job = {
        "id": job_id,
        "status": "queued",
        "createdAt": started_at,
        "queuedAt": started_at,
        "updatedAt": started_at,
        "originalName": original_name,
        "sourcePath": str(source_path),
        "wavPath": str(wav_path),
        "modelName": model_name,
        "language": language.strip() or "auto",
        "engineName": engine_name,
        "useGpu": useGpu == "1",
        "prompt": prompt.strip() or None,
        "diarize": ENABLE_DIARIZATION and diarize == "1",
        "chunkSeconds": CHUNK_SECONDS,
        "chunkOverlapSeconds": CHUNK_OVERLAP_SECONDS,
        "chunkTranscriptionEnabled": ENABLE_CHUNK_TRANSCRIPTION,
        "totalChunks": None,
        "completedChunks": 0,
        "currentChunk": None,
        "partialText": "",
        "partialSegments": [],
    }
    _job_set(job_id, job)
    _enqueue_job(job_id)

    queued_count, running_count = _queue_counts()
    return {
        "jobId": job_id,
        "status": "queued",
        "queuePosition": _queue_position(job_id),
        "queuedCount": queued_count,
        "runningCount": running_count,
    }


# ── Job status poll endpoint ───────────────────────────────────────────
@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    redis_error = _require_redis_response()
    if redis_error is not None:
        return redis_error

    job = _job_get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    queued_count, running_count = _queue_counts()
    status = job.get("status", "queued")
    if status == "done":
        result = job["result"]
        result.update({
            "status": "done",
            "queuedCount": queued_count,
            "runningCount": running_count,
        })
        return result
    if status == "error":
        return JSONResponse(
            {
                "error": job.get("error", "Job failed"),
                "status": "error",
                "queuedCount": queued_count,
                "runningCount": running_count,
            },
            status_code=500,
        )

    return {
        "status": status,
        "queuePosition": _queue_position(job_id) if status == "queued" else None,
        "queuedCount": queued_count,
        "runningCount": running_count,
        "totalChunks": job.get("totalChunks"),
        "completedChunks": job.get("completedChunks", 0),
        "currentChunk": job.get("currentChunk"),
        "chunkSeconds": job.get("chunkSeconds"),
        "partialText": job.get("partialText", ""),
        "partialSegments": job.get("partialSegments", []),
        "startedAt": job.get("runStartedAt") or job.get("createdAt"),
        "updatedAt": job.get("updatedAt"),
        "originalName": job.get("originalName"),
    }


# ── Transcription worker ───────────────────────────────────────────────
def _probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError:
        return 0.0


def _extract_wav_range(source: str, dest: str, start_s: float, end_s: float) -> None:
    duration = max(0.05, end_s - start_s)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-i", source,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            dest,
        ],
        check=True,
    )


def _chunk_windows(duration_s: float) -> list[dict[str, float]]:
    if not ENABLE_CHUNK_TRANSCRIPTION or duration_s <= 0:
        return [{"coreStart": 0.0, "coreEnd": max(duration_s, CHUNK_SECONDS), "extractStart": 0.0, "extractEnd": duration_s}]

    windows: list[dict[str, float]] = []
    start = 0.0
    while start < duration_s:
        end = min(duration_s, start + CHUNK_SECONDS)
        windows.append({
            "coreStart": start,
            "coreEnd": end,
            "extractStart": max(0.0, start - CHUNK_OVERLAP_SECONDS),
            "extractEnd": min(duration_s, end + CHUNK_OVERLAP_SECONDS),
        })
        start += CHUNK_SECONDS
    return windows or [{"coreStart": 0.0, "coreEnd": duration_s, "extractStart": 0.0, "extractEnd": duration_s}]


def _transcribe_audio_file(
    path: str,
    *,
    model_name: str,
    language: str,
    engine_name: str,
    use_gpu: bool,
    prompt: str | None,
) -> dict[str, Any]:
    if engine_name == "whisper-cpp":
        return transcribe_cpp(
            path,
            model_name=model_name,
            language=language,
            use_gpu=True,
            prompt=prompt,
            timeout=TRANSCRIPTION_TIMEOUT if TRANSCRIPTION_TIMEOUT > 0 else None,
        )
    return faster_transcribe(
        path,
        model_name=model_name,
        language=language,
        use_gpu=use_gpu,
        prompt=prompt,
    )


def _normalize_text_for_dedupe(text: str) -> str:
    return "".join(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _is_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < 18:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.94


def _compress_repeated_sentences(text: str) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text) if p.strip()]
    if len(parts) < 2:
        return text.strip()
    out: list[str] = []
    last_norm = ""
    repeat_count = 0
    for part in parts:
        norm = _normalize_text_for_dedupe(part)
        if norm and norm == last_norm:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            last_norm = norm
            repeat_count = 0
        out.append(part)
    return " ".join(out).strip()


def _dedupe_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    recent_norms: list[str] = []
    repeated_norm = ""
    repeat_count = 0

    for seg in sorted(segments, key=lambda s: (s.get("startMs") or 0, s.get("endMs") or 0)):
        text = _compress_repeated_sentences(str(seg.get("text") or "").strip())
        norm = _normalize_text_for_dedupe(text)
        if not norm:
            continue

        if recent_norms and norm == recent_norms[-1]:
            continue

        if norm == repeated_norm:
            repeat_count += 1
        else:
            repeated_norm = norm
            repeat_count = 1
        if repeat_count >= 3:
            continue

        similar_recent = sum(1 for prev in recent_norms[-6:] if _is_similar(norm, prev))
        if similar_recent >= 2:
            continue

        seg = dict(seg)
        seg["text"] = text
        deduped.append(seg)
        recent_norms.append(norm)
        recent_norms = recent_norms[-8:]

    return deduped


def _offset_and_filter_segments(
    raw_segments: list[dict[str, Any]],
    *,
    extract_start_s: float,
    core_start_s: float,
    core_end_s: float,
    is_last: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    core_start_ms = round(core_start_s * 1000)
    core_end_ms = round(core_end_s * 1000)
    offset_ms = round(extract_start_s * 1000)

    for seg in raw_segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("startMs")
        end = seg.get("endMs")
        if start is None or end is None:
            start_ms = core_start_ms
            end_ms = core_start_ms
        else:
            start_ms = int(start) + offset_ms
            end_ms = int(end) + offset_ms
        midpoint = (start_ms + end_ms) / 2
        if midpoint < core_start_ms:
            continue
        if is_last:
            if midpoint > core_end_ms:
                continue
        elif midpoint >= core_end_ms:
            continue
        out.append({
            "startMs": max(core_start_ms, start_ms),
            "endMs": min(core_end_ms, max(end_ms, start_ms)),
            "text": text,
            "language": seg.get("language"),
        })
    return out


def _run_chunked_transcription(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    source_path = str(job["sourcePath"])
    wav_path = str(job["wavPath"])
    model_name = str(job["modelName"])
    language = str(job.get("language") or "auto")
    engine_name = str(job["engineName"])
    use_gpu = bool(job.get("useGpu"))
    prompt = job.get("prompt")
    started_at = float(job.get("createdAt") or time.time())

    convert_to_wav(source_path, wav_path)
    duration_s = _probe_duration(wav_path)
    windows = _chunk_windows(duration_s)
    total_chunks = len(windows)
    chunk_dir = UPLOAD_DIR / f"{job_id}-chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    _job_update(job_id, {
        "chunkDir": str(chunk_dir),
        "audioDurationSeconds": duration_s,
        "totalChunks": total_chunks,
        "completedChunks": 0,
        "currentChunk": 1 if total_chunks else None,
        "partialText": "",
        "partialSegments": [],
    })

    all_segments: list[dict[str, Any]] = []
    result_model = model_name
    result_language = "unknown"

    try:
        for index, win in enumerate(windows, start=1):
            chunk_path = chunk_dir / f"chunk-{index:04d}.wav"
            _job_update(job_id, {"currentChunk": index})
            try:
                _extract_wav_range(wav_path, str(chunk_path), win["extractStart"], win["extractEnd"])
                chunk_result = _transcribe_audio_file(
                    str(chunk_path),
                    model_name=model_name,
                    language=language,
                    engine_name=engine_name,
                    use_gpu=use_gpu,
                    prompt=prompt,
                )
                result_model = str(chunk_result.get("model") or result_model)
                if result_language == "unknown":
                    result_language = str(chunk_result.get("language") or result_language)
                filtered = _offset_and_filter_segments(
                    list(chunk_result.get("segments") or []),
                    extract_start_s=win["extractStart"],
                    core_start_s=win["coreStart"],
                    core_end_s=win["coreEnd"],
                    is_last=index == total_chunks,
                )
                all_segments.extend(filtered)
                all_segments = _dedupe_segments(all_segments)
                partial_text = " ".join(s["text"] for s in all_segments).strip()
                _job_update(job_id, {
                    "completedChunks": index,
                    "currentChunk": index if index < total_chunks else total_chunks,
                    "partialText": partial_text,
                    "partialSegments": all_segments,
                })
            finally:
                _safe_unlink(chunk_path)
    finally:
        _safe_rmtree(chunk_dir)

    # ── Speaker diarization (optional) ─────────────────────────────────
    # Run once on the WHOLE wav (global timestamps), then tag the already-merged
    # segments. Best-effort: failure here must not drop the transcript.
    speaker_count = 0
    diarize_requested = bool(job.get("diarize"))
    if diarize_requested and all_segments:
        if diarization.is_available():
            _job_update(job_id, {"status": "running", "stage": "diarizing"})
            try:
                turns = diarization.diarize_file(wav_path)
                speaker_count = diarization.assign_speakers(all_segments, turns)
                log.info("diarization: %s tagged %d speakers over %d turns",
                         job_id, speaker_count, len(turns))
            except Exception as exc:
                log.error("diarization: unexpected failure on %s (%s)", job_id, exc)
        else:
            log.warning("diarization requested but unavailable (%s)",
                        diarization.status().get("reason"))

    text = " ".join(s["text"] for s in all_segments).strip()
    return {
        "id": job_id,
        "text": text,
        "language": result_language,
        "model": result_model,
        "engine": engine_name,
        "durationMs": int((time.time() - started_at) * 1000),
        "segments": all_segments,
        "totalChunks": total_chunks,
        "completedChunks": total_chunks,
        "chunkSeconds": CHUNK_SECONDS,
        "audioDurationSeconds": duration_s,
        "diarized": diarize_requested and speaker_count > 0,
        "speakerCount": speaker_count,
        "files": {"originalName": job.get("originalName") or "audio"},
    }


def _do_transcription(job_id: str) -> dict[str, Any]:
    job = _job_get(job_id)
    if not job:
        raise RuntimeError("Job not found")
    return _run_chunked_transcription(job_id, job)


# Mount static UI AFTER routes so API endpoints take precedence.
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")


# ── Entrypoint ─────────────────────────────────────────────────────────
def main():
    import uvicorn

    uvicorn.run("src.server:app", host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
