# Local STT backend — Node + whisper.cpp + ffmpeg in one image.
# whisper.cpp is compiled at build time (CPU); models are NOT baked in — they
# download into a mounted volume on first start (see docker-entrypoint.sh), so
# the image stays lean and models persist across container restarts.
FROM node:22-bookworm-slim

# ffmpeg = audio/video decode; git+cmake+build-essential = compile whisper.cpp;
# curl+ca-certificates = download ggml models at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git cmake build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Node deps first for layer caching.
COPY package*.json ./
RUN npm ci --omit=dev

# App source (see .dockerignore — host vendor/node_modules/uploads excluded).
COPY . .

# Clone + build whisper.cpp (CPU build; no Metal/CUDA in this image).
# GGML_NATIVE=OFF + explicit -march: on aarch64 the fp16 NEON intrinsics need a
# +fp16-capable arch or GCC aborts ("vfmaq_f16 target specific option mismatch").
# -Wno-error neutralises a GCC stringop-overflow false-positive; -j2 caps build RAM.
RUN set -e; \
    ARCH="$(uname -m)"; \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
      export CFLAGS="-march=armv8.2-a+fp16+dotprod -Wno-error"; \
    else \
      export CFLAGS="-Wno-error"; \
    fi; \
    export CXXFLAGS="$CFLAGS"; \
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git vendor/whisper.cpp; \
    cmake -S vendor/whisper.cpp -B vendor/whisper.cpp/build -DWHISPER_BUILD_TESTS=OFF -DGGML_NATIVE=OFF; \
    cmake --build vendor/whisper.cpp/build --config Release -j2; \
    rm -rf vendor/whisper.cpp/.git vendor/whisper.cpp/build/CMakeFiles

ENV PORT=8789 \
    WHISPER_MODELS=base
EXPOSE 8789

# Models persist here; entrypoint fetches any missing ones listed in WHISPER_MODELS.
VOLUME ["/app/backend/vendor/whisper.cpp/models"]

ENTRYPOINT ["bash", "docker-entrypoint.sh"]
