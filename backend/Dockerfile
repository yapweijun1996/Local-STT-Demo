# Local STT backend — Python + faster-whisper + ffmpeg.
# Dual-engine: faster-whisper (code-switching) always available; whisper-cpp
# (Metal GPU) auto-detected on bare-metal Mac, unavailable in Docker.
# Models are NOT baked in — they download into a mounted volume on first start
# (see docker-entrypoint.sh), so the image stays lean and models persist.
FROM python:3.12-slim-bookworm

# ffmpeg = audio/video decode; curl = model pre-warm in entrypoint.
# No cmake / build-essential needed — faster-whisper ships pre-built wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Python deps first for layer caching (only invalidates when requirements.txt changes).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source (see .dockerignore — __pycache__ / node_modules / vendor excluded).
# Lands AFTER pip install so code changes don't re-run pip.
COPY . .

ENV PORT=6601 \
    WHISPER_MODEL=large-v3-turbo \
    STT_ENGINE=faster-whisper \
    HF_HOME=/app/backend/vendor/models

EXPOSE 6601

# Models persist here; entrypoint pre-downloads the configured model on first start.
VOLUME ["/app/backend/vendor/models"]

ENTRYPOINT ["bash", "docker-entrypoint.sh"]
