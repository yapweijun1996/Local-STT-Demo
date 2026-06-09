#!/usr/bin/env bash
# Build whisper.cpp with Apple Metal GPU support for the whisper-cpp engine.
# Prereqs: git, cmake, Xcode CLI tools.
#   chmod +x scripts/setup-whisper-cpp.sh
#   ./scripts/setup-whisper-cpp.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_DIR="${WHISPER_CPP_DIR:-$ROOT_DIR/vendor/whisper.cpp}"
MODEL_DIR="$WHISPER_DIR/models"

mkdir -p "$ROOT_DIR/vendor"
mkdir -p "$MODEL_DIR"

if [ ! -d "$WHISPER_DIR/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$WHISPER_DIR"
else
  git -C "$WHISPER_DIR" pull --ff-only
fi

cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" \
  -DWHISPER_BUILD_TESTS=OFF \
  -DGGML_METAL=ON

cmake --build "$WHISPER_DIR/build" --config Release -j"$(sysctl -n hw.ncpu)"

echo "whisper-cli built: $WHISPER_DIR/build/bin/whisper-cli"
echo "Models dir: $MODEL_DIR"
echo
echo "Download models: cd $MODEL_DIR && ./download-ggml-model.sh large-v3"
