#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_DIR="${WHISPER_CPP_DIR:-$ROOT_DIR/vendor/whisper.cpp}"
CLI="${WHISPER_BIN:-$WHISPER_DIR/build/bin/whisper-cli}"
SAMPLE="${1:-$WHISPER_DIR/samples/jfk.wav}"
LANGUAGE="${LANGUAGE:-en}"
DEVICE_FLAG="${WHISPER_USE_GPU:-0}"
PROMPT="${WHISPER_PROMPT:-}"
# Models to benchmark; override with: MODELS="base small large-v3-turbo" npm run bench
MODELS="${MODELS:-base small}"

mkdir -p "$ROOT_DIR/transcripts/bench"

if [ ! -x "$CLI" ]; then
  echo "whisper-cli not found at $CLI. Run npm run setup first." >&2
  exit 1
fi

EXTRA_ARGS=()
if [ "$DEVICE_FLAG" != "1" ]; then
  EXTRA_ARGS+=(-ng)
fi

if [ -n "$PROMPT" ]; then
  EXTRA_ARGS+=(--prompt "$PROMPT")
fi

for model in $MODELS; do
  MODEL_PATH="$WHISPER_DIR/models/ggml-$model.bin"
  OUT="$ROOT_DIR/transcripts/bench/$(basename "$SAMPLE" | sed 's/\.[^.]*$//')-$model"
  if [ ! -f "$MODEL_PATH" ]; then
    echo "Missing model: $MODEL_PATH" >&2
    exit 1
  fi
  echo "== $model =="
  /usr/bin/time -p "$CLI" \
    -m "$MODEL_PATH" \
    -f "$SAMPLE" \
    -l "$LANGUAGE" \
    -oj -ojf \
    -of "$OUT" \
    -np \
    "${EXTRA_ARGS[@]}"
done
