# Local STT — Node.js + whisper.cpp backend

A small **Node.js (Express)** server that runs **whisper.cpp** locally for speech-to-text.
Upload **mp4 / mp3 / wav / m4a / ogg** or record from the **mic** in the bundled web UI;
the file is converted to 16 kHz mono by **ffmpeg** and transcribed by `whisper-cli`.
Nothing is sent to any cloud — audio is processed on your machine and deleted after each request.

This is the **server-side** counterpart to the zero-install browser demo in the repo root
([`../index.html`](../index.html)). Use this backend when you want native speed, larger
models, longer files, or to plug STT into an ERP / agent backend.

## Why a backend (vs. the browser demo)

| | Browser (`../index.html`) | This Node backend |
|---|---|---|
| Install | none — double-click | Node + ffmpeg + compile whisper.cpp |
| Engine | Transformers.js (ONNX, WebGPU/WASM) | whisper.cpp (native, Metal/CPU) |
| Speed / long files | good, browser-bound | faster, server-bound |
| Integration | client-only | HTTP API for ERP / agents |

## Requirements

- **Node.js** 18+
- **ffmpeg** (audio/video → 16 kHz mono wav)
- **cmake** + a C/C++ toolchain (to build whisper.cpp)

macOS: `brew install cmake ffmpeg` · Windows: `winget install Kitware.CMake Gyan.FFmpeg`

## Setup

```bash
cd backend
npm install
npm run setup          # clones + builds whisper.cpp, downloads base + small models
```

Want the most accurate model too? Download it during setup:

```bash
WHISPER_MODELS="base small large-v3-turbo" npm run setup
```

Windows: `npm run setup:win`.

## Run

```bash
npm start
# → http://localhost:8789/
```

Web UI: open `http://localhost:8789/`. The health badge shows which models are installed.

Health check:

```bash
curl -sS http://localhost:8789/health | jq .
```

Transcribe via API:

```bash
curl -sS -X POST http://localhost:8789/api/transcribe \
  -F audio=@/path/to/clip.mp4 \
  -F model=small \
  -F language=auto \
  -F 'prompt=ERP vocabulary: sales order, SKU A123, Acme Singapore.' \
  | jq '{text,language,model,durationMs,segments}'
```

## Configuration (env vars)

- `PORT` — API port, default `8789`.
- `WHISPER_MODEL` — default model path (default `vendor/whisper.cpp/models/ggml-base.bin`).
- `WHISPER_CPP_DIR` / `WHISPER_BIN` — custom whisper.cpp dir / `whisper-cli` path.
- `WHISPER_USE_GPU=1` — allow GPU/Metal (default CPU; short clips are often faster on CPU).
- `WHISPER_PROMPT` — default vocabulary/context prompt (empty = neutral). Per-request `prompt` overrides it.
- `ALLOW_CUSTOM_MODEL_PATH=1` — let clients pass an arbitrary model path (off by default).
- `MAX_AUDIO_BYTES` — upload limit, default 200 MB.
- `KEEP_TRANSCRIPTS=1` — keep per-request transcript JSON in `transcripts/` (default: deleted).

## Models

Registry keys: `tiny`, `base`, `small`, `large-v3-turbo`. Only installed models are
selectable in the UI. `base` is a good default; `large-v3-turbo` is the most accurate
(larger download + compute). Pass a domain `prompt` for proper nouns / SKUs / brands.

## Privacy

The uploaded source and the intermediate 16 kHz wav are **always deleted** after each
request (success or failure). Transcript JSON is also deleted unless `KEEP_TRANSCRIPTS=1`.
CORS is `*` so a `file://` browser page can call the API — tighten this before exposing
beyond localhost.

## License

MIT (this server). It wraps **whisper.cpp** (MIT) and **OpenAI Whisper** weights (MIT).
**ffmpeg** is a separate system dependency (LGPL/GPL) — it is *required at runtime* but not
bundled or redistributed here; review its license if you ship ffmpeg binaries yourself.
