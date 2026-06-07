#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_DIR="${WHISPER_CPP_DIR:-$ROOT_DIR/vendor/whisper.cpp}"
# Default models. Add "large-v3-turbo" for best accuracy:
#   WHISPER_MODELS="base small large-v3-turbo" npm run setup
MODEL_LIST="${WHISPER_MODELS:-base small}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required. macOS: brew install cmake. Windows: winget install Kitware.CMake" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required for reliable audio conversion. macOS: brew install ffmpeg. Windows: winget install Gyan.FFmpeg" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/vendor" "$ROOT_DIR/uploads" "$ROOT_DIR/transcripts"

if [ ! -d "$WHISPER_DIR/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$WHISPER_DIR"
else
  git -C "$WHISPER_DIR" pull --ff-only
fi

CMAKE_ARGS=(-DWHISPER_BUILD_TESTS=OFF)
case "$(uname -s)" in
  Darwin)
    CMAKE_ARGS+=(-DGGML_METAL=ON)
    ;;
esac

cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" "${CMAKE_ARGS[@]}"
cmake --build "$WHISPER_DIR/build" --config Release -j

(
  cd "$WHISPER_DIR/models"
  for model in $MODEL_LIST; do
    ./download-ggml-model.sh "$model"
  done
)

echo "Setup complete."
echo "whisper-cli: $WHISPER_DIR/build/bin/whisper-cli"
echo "models: $WHISPER_DIR/models"
