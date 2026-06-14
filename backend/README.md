# Local STT — Python FastAPI backend

Server-side speech-to-text backend for the bundled web UI and API clients. It accepts
audio/video uploads, converts them to 16 kHz mono WAV with ffmpeg, queues work in Redis,
and transcribes locally with either:

- `whisper-cpp` for Apple Silicon / native GPU speed.
- `faster-whisper` for Docker and CPU-friendly deployments.

The API returns a `jobId` immediately, then the browser polls `/api/job/{jobId}` until
the transcript is ready. Uploaded audio and intermediate chunks are deleted after each
job finishes.

## Requirements

- Python 3.12 recommended
- ffmpeg
- Redis
- Optional native mode: whisper.cpp built with Metal/CUDA and local `ggml-*.bin` models

macOS:

```bash
brew install ffmpeg redis cmake
```

## Native Setup

```bash
cd backend
python3.12 -m venv .venv312
. .venv312/bin/activate
pip install -r requirements.txt

# Optional: build whisper.cpp for native Apple Metal mode
./scripts/setup-whisper-cpp.sh
```

Start Redis, then run:

```bash
redis-server
python -m src.server
# http://localhost:6601/
```

For native whisper.cpp mode, configure:

```bash
export STT_ENGINE=whisper-cpp
export WHISPER_MODEL=large-v3
export WHISPER_CPP_DIR="$PWD/vendor/whisper.cpp"
python -m src.server
```

## Docker

Docker uses `faster-whisper` by default because whisper.cpp GPU/Metal is not available
inside the image.

```bash
docker compose up --build
# http://localhost:6601/
```

Models are cached in the `stt-models` volume. The compose default is
`WHISPER_MODEL=large-v3-turbo` and `STT_ENGINE=faster-whisper`.

## API

Health:

```bash
curl -sS http://localhost:6601/health | jq .
```

Submit a job:

```bash
curl -sS -X POST http://localhost:6601/api/transcribe \
  -F audio=@/path/to/clip.mp4 \
  -F model=large-v3-turbo \
  -F engine=faster-whisper \
  -F language=auto \
  | jq .
```

Poll the returned job:

```bash
curl -sS http://localhost:6601/api/job/JOB_ID | jq .
```

## Configuration

- `PORT` — API port, default `6601`.
- `REDIS_URL` — Redis connection string, default `redis://localhost:6379/0`.
- `STT_ENGINE` — `whisper-cpp` or `faster-whisper`, default `whisper-cpp`.
- `WHISPER_MODEL` — default model key, default `large-v3`.
- `HF_HOME` — faster-whisper / pyannote cache directory.
- `WHISPER_CPP_DIR` / `WHISPER_CPP_BIN` / `WHISPER_CPP_MODEL_DIR` — whisper.cpp paths.
- `MAX_AUDIO_BYTES` — per-upload size limit, default 200 MB.
- `UPLOADS_MAX_BYTES` — total upload workspace cap, default 5 GB.
- `WORKER_CONCURRENCY` — concurrent transcriptions for the elected worker leader.
- `JOB_TTL` — Redis job lifetime, default 12 hours.
- `ENABLE_CHUNK_TRANSCRIPTION` — set `0` to disable chunked transcription.
- `CHUNK_SECONDS` / `CHUNK_OVERLAP_SECONDS` — chunk sizing for long audio.
- `ENABLE_DIARIZATION` — server-wide speaker diarization switch.
- `STT_CORS_ORIGINS` — comma-separated allowed browser origins. Defaults to localhost origins.
- `STT_RATE_LIMIT_PER_MINUTE` — per-IP submit limit, default `20`; set `0` to disable.
- `STT_API_KEY` — optional API key. When set, `/api/transcribe` and `/api/job/{id}` require
  `X-STT-API-Key: ...` or `Authorization: Bearer ...`. The bundled browser UI sends
  `localStorage["stt-api-key"]` as `X-STT-API-Key` when that value is present.

## Models

Registry keys: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`.

Only installed models are accepted for the selected engine. Docker pre-warms the configured
faster-whisper model at container startup. Native whisper.cpp models live under
`backend/vendor/whisper.cpp/models/` as `ggml-*.bin`.

## Privacy And Security

Audio files are cleaned after job completion. CORS is no longer open by default; configure
`STT_CORS_ORIGINS` explicitly before exposing this service to browsers on other domains.

For public deployments, set `STT_API_KEY`, keep rate limiting enabled, and put the service
behind your normal reverse-proxy limits.

## License

MIT. This backend wraps whisper.cpp / faster-whisper and uses ffmpeg as a system dependency.
