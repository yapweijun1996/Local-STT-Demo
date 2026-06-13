# STT Model Choice Guide

_Updated 2026-06-13._ This repo has two transcription paths; pick the model per path.

## Browser demo (Transformers.js + ONNX, WebGPU/WASM)

Sizes below are the **actual download on the default WebGPU path** (tiny/base/small = fp32;
turbo = fp16 + q4; large-v3 = fp16 + q8). On WASM/CPU the weights are quantized and smaller.

| Model (dropdown value) | Size (WebGPU) | Use it for | Notes |
|---|---|---|---|
| `Xenova/whisper-tiny` | ~145 MB | smoke-testing | rough output |
| `Xenova/whisper-base` | ~280 MB | fast, decent English | good default |
| `Xenova/whisper-small` | ~920 MB | better accuracy | slower |
| `onnx-community/whisper-large-v3-turbo` | ~1.5 GB | **recommended top-tier** | high quality **and actually runs in-browser**; WebGPU only |
| `Xenova/whisper-large-v3` | ~2.0 GB | absolute max quality | ⚠️ often **runs out of memory at inference** in-browser — see below |

### ⚠️ Why `whisper-large-v3` usually fails in the browser

Its merged decoder only ships `fp32` (several GB — instant OOM) and `quantized`/`q8`
(int8). int8 is **not a WebGPU op**, so the decoder runs on the **CPU/WASM** execution
provider. The browser runs WASM **single-threaded** (no `SharedArrayBuffer`, because
`file://` and GitHub Pages aren't cross-origin-isolated), so it can't get the allocation and
onnxruntime throws an OOM (a bare byte number like `180096440`). This happens even on an
Apple-Silicon + WebGPU Mac, so it will fail on most machines.

**Use `whisper-large-v3-turbo` instead** — it's the distilled large-v3 (fewer decoder layers),
ships a `q4` merged decoder that runs well, and gives near-large-v3 quality. The demo keeps
`whisper-large-v3` as an option (with an upfront warning) for powerful setups, and maps the OOM
to a friendly "too heavy — try turbo" message.

### Download behavior

- The progress bar shows **real per-file bytes + ETA** from the library's `progress_callback`.
  (The Hugging Face CDN sends no `Timing-Allow-Origin`, so the browser's own Resource Timing
  byte counts are 0 — don't rely on them.)
- **Network drops are retried** (`ERR_CONTENT_LENGTH_MISMATCH` / "network error"), up to 3
  attempts. Completed files are cached in the browser, so a retry **resumes** — only the
  interrupted file re-downloads.
- After the first download the model is cached and works fully offline.

## Node backend — two engines (whisper.cpp vs faster-whisper)

The backend exposes **two transcription engines**, selectable per request via the `engine`
form field on `/api/transcribe` and via the **Engine dropdown** in the web UI. `/health`
reports which engines + models are installed.

| Engine | Hardware path | Speed (Apple Silicon) | Pick it for |
|---|---|---|---|
| `whisper-cpp` (default) | **Metal GPU** on macOS | fast | everyday use on this Mac deployment |
| `faster-whisper` | **CPU / int8** on macOS (CUDA float16 on NVIDIA) | slow on Mac | code-switching, or when run on a CUDA server |

**Benchmark (verified 2026-06-13, jfk.wav ~11 s, large-v3, this Mac):** whisper-cpp Metal
transcribes in ~1.1 s vs faster-whisper CPU ~14.5 s — **~13× faster**. Accuracy is identical
(same weights). On Apple Silicon, prefer `whisper-cpp`; `faster-whisper` only wins on NVIDIA.
The UI shows a "CPU-only, ~13× slower, be patient" warning when faster-whisper is selected.

### Models per engine

whisper.cpp uses `ggml-*.bin` files (Metal):

| Model alias | Use it for |
|---|---|
| `base` | fast default |
| `small` | better punctuation / non-English |
| `large-v3-turbo` | best speed/quality balance |
| `large-v3` | absolute best quality (largest, slowest) |

Install extra models: `WHISPER_MODELS="base small large-v3-turbo" npm run setup` (or set
`WHISPER_MODELS` in `docker-compose.yml` for Docker). See [backend/README.md](../backend/README.md).

faster-whisper uses CTranslate2 weights cached in `HF_HOME` (e.g.
`Systran/faster-whisper-large-v3`, `mobiuslabsgmbh/faster-whisper-large-v3-turbo`).
**Pre-download required** — the production server runs `HF_HUB_OFFLINE=1`, so a model that
isn't already cached returns a clear `400` instead of fetching on demand. The UI only enables
faster-whisper models that are actually installed; the API validates the same.

## Decision rule

- Fastest local responsiveness → `small` (either path).
- Best **in-browser** quality → `whisper-large-v3-turbo` (needs WebGPU: Chrome/Edge).
- Best quality overall, latency/memory OK → `large-v3` on the **backend** (not the browser).
- Non-English → set the language explicitly; small models guess poorly on auto-detect.

## Speaker diarization ("who spoke when")

Tick **Detect speakers** (on by default) to label each segment `Speaker 1 / 2 / …`. Runs
pyannote `speaker-diarization-3.1` on the whole file (token-free, offline). Two accuracy levers:

- **Set the speaker count** when you know it (the "How many speakers?" dropdown / `speakers`
  form field, 1–8). This is the **biggest** accuracy boost — auto-estimation frequently
  mis-counts (splits one person into several, or merges two). A 2-person interview → pick **2**.
- **Boundaries** are aligned to the transcript. A segment that genuinely spans a speaker change
  is split at **word granularity** (using word timestamps from either engine) so the label
  flips between the right words, not mid-phrase. faster-whisper gives slightly finer word
  timing; whisper-cpp (Metal) is much faster and good enough for most audio.

Diarization adds a full second pass over the audio, so it's slower than plain transcription —
especially on the faster-whisper CPU engine. Untick it when you don't need speaker labels.

## Notes

- Model binaries are kept out of git (`backend/.gitignore`); the backend downloads them on
  `npm run setup` / first Docker start.
- The browser caches model files per-browser; clearing site data forces a re-download.
