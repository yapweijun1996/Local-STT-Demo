#!/usr/bin/env bash
set -e

# Pre-download the configured faster-whisper model so the first request
# doesn't time out waiting for a multi-GB download from HuggingFace.
#
# WHISPER_MODEL: one of tiny / base / small / medium / large-v3 / large-v3-turbo
# (default: large-v3-turbo).
# HF_HOME: model cache dir (mounted volume), default /app/backend/vendor/models.

MODEL="${WHISPER_MODEL:-large-v3-turbo}"
HF_HOME="${HF_HOME:-/app/backend/vendor/models}"
export HF_HOME

echo "[entrypoint] HuggingFace cache: $HF_HOME"
echo "[entrypoint] Ensuring model '$MODEL' is downloaded..."

# Use a tiny Python snippet to trigger faster-whisper's auto-download.
# Downloads the CTranslate2 model (~1.5 GB for large-v3-turbo) if not cached.
python3 -c "
import os, sys
from faster_whisper import WhisperModel

model = WhisperModel('${MODEL}', device='cpu', compute_type='int8',
                     download_root=os.environ.get('HF_HOME'))
# Just initialising the model triggers the download.
# No audio passed — we only want the weights on disk.
print(f'[entrypoint] Model ${MODEL} ready.')
" || echo "[entrypoint] WARN: model pre-download failed; will retry on first request"

echo "[entrypoint] Starting server on port ${PORT:-8789}…"
exec python3 -m src.server
