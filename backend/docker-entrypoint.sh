#!/usr/bin/env bash
set -e

# Download any ggml models listed in WHISPER_MODELS (space-separated) that aren't
# already present in the (mounted) models dir. Self-contained curl — does not rely
# on the upstream download script, which a volume mount can shadow.
MODELS_DIR="${WHISPER_CPP_DIR:-/app/backend/vendor/whisper.cpp}/models"
mkdir -p "$MODELS_DIR"

for m in ${WHISPER_MODELS:-base}; do
  f="$MODELS_DIR/ggml-$m.bin"
  if [ ! -s "$f" ]; then
    echo "[entrypoint] downloading model: $m"
    if ! curl -fL --retry 3 -o "$f" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$m.bin"; then
      echo "[entrypoint] WARN: failed to download '$m' (check the name); continuing"
      rm -f "$f"
    fi
  fi
done

echo "[entrypoint] models present: $(ls -1 "$MODELS_DIR"/ggml-*.bin 2>/dev/null | wc -l | tr -d ' ')"
exec node src/server.js
