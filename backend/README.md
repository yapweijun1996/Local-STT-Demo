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
WHISPER_MODELS="base small large-v3" npm run setup
```

Windows: `npm run setup:win`.

### GitHub push note

Model artifacts are intentionally kept local and excluded from git:

- `backend/vendor/whisper.cpp/models/` (downloaded `.bin` models)
- `backend/uploads/` and `backend/transcripts/` (runtime temp/output)

So `git push` stays lightweight and does not include binary data.  
If you need to bootstrap a fresh environment, run `npm run setup` there instead of checking in binaries.

If you later want to version a small subset of models, do it via Git LFS and explicit patterns only
after reviewing GitHub LFS quotas.

## Run

```bash
npm start
# → http://localhost:8789/
```

Web UI: open `http://localhost:8789/`. The health badge shows which models are installed.

**Interaction model:** an **uploaded file** waits for you to click **Transcribe file** (you may
want to change the model/language first). A **mic recording** transcribes **automatically when you
press Stop** — Stop is treated as "I'm done, go". While a transcription is running the Transcribe
button is disabled and shows "Transcribing…" so you can't double-submit. The mic also shows a live
waveform while recording. It's an installable PWA (works offline for the UI shell over http/localhost).

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

## Docker

Run the whole backend (Node + whisper.cpp + ffmpeg + web UI) in one container — no
local toolchain needed. From the repo root:

```bash
docker compose up --build
# → http://localhost:8789/
```

- whisper.cpp is **compiled into the image** (CPU build); ffmpeg is included.
- Models are **not baked in** — they download into a named volume (`stt-models`) on
  first start, controlled by `WHISPER_MODELS` (default `base`). They persist across restarts.
- Want more accuracy? Edit `WHISPER_MODELS` in `docker-compose.yml`
  (e.g. `"base small large-v3-turbo"` or `"base small large-v3"`) and restart;
  only missing models download.
- GPU/Metal isn't available in the CPU image (`WHISPER_USE_GPU=0`). For Apple Metal /
  NVIDIA acceleration, run natively (`npm run setup && npm start`).

Without compose:

```bash
docker build -t local-stt-backend ./backend
docker run --rm -p 8789:8789 -e WHISPER_MODELS="base" \
  -v stt-models:/app/backend/vendor/whisper.cpp/models local-stt-backend
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

Registry keys: `tiny`, `base`, `small`, `large-v3`, `large-v3-turbo`. Only installed models are
selectable in the UI. `base` is a good default. `large-v3` is the highest-quality option,
while `large-v3-turbo` is a better speed/quality tradeoff.

## Benchmark (base vs. small vs. large-v3-turbo)

Measured on this machine (Apple Silicon Mac) with whisper.cpp's bundled
`samples/jfk.wav` (~11s clean English), via `npm run bench`. `real` is wall time;
**×realtime** = audio seconds ÷ processing seconds (higher is faster).

| Model | Size | CPU (`-ng`) | Metal (`WHISPER_USE_GPU=1`) | ×realtime (Metal) | Accuracy on this clip |
|---|---:|---:|---:|---:|---|
| base | 141 MB | 0.54s | 0.56s | ~20× | correct, missed a comma |
| small | 465 MB | 1.56s | 1.07s | ~10× | correct, good punctuation |
| large-v3 | 2.6 GB | 5.8s | 2.73s | ~4× | best quality |
| large-v3-turbo | 1.5 GB | 4.10s | 2.09s | ~5× | correct, best punctuation |

Takeaways:
- All three are **far faster than real time** even on CPU — the backend's edge is native speed.
- **Metal** helps the larger models most (turbo ~2× faster); on tiny clips `base` sees little
  gain because Metal setup overhead dominates.
- On clean English the gap is smaller, and `large-v3` is typically best for raw quality,
  with `large-v3-turbo` usually winning for practical throughput and accuracy balance.

Reproduce with another file: `npm run bench -- /path/to/clip.wav` (or set `MODELS=...`,
`LANGUAGE=...`, `WHISPER_USE_GPU=1`).

## Privacy

The uploaded source and the intermediate 16 kHz wav are **always deleted** after each
request (success or failure). Transcript JSON is also deleted unless `KEEP_TRANSCRIPTS=1`.
CORS is `*` so a `file://` browser page can call the API — tighten this before exposing
beyond localhost.

## License

MIT (this server). It wraps **whisper.cpp** (MIT) and **OpenAI Whisper** weights (MIT).
**ffmpeg** is a separate system dependency (LGPL/GPL) — it is *required at runtime* but not
bundled or redistributed here; review its license if you ship ffmpeg binaries yourself.
