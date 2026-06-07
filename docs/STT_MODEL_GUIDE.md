# STT Model Choice Guide

_Updated 2026-06-07._ This repo has two transcription paths; pick the model per path.

## Browser demo (Transformers.js + ONNX, WebGPU/WASM)

| Model (dropdown value) | Size (download) | Use it for | Notes |
|---|---|---|---|
| `Xenova/whisper-tiny` | ~40 MB | smoke-testing | rough output |
| `Xenova/whisper-base` | ~150 MB | fast, decent English | good default for WASM/no-GPU |
| `Xenova/whisper-small` | ~480 MB | better accuracy | slower |
| `onnx-community/whisper-large-v3-turbo` | ~1.2 GB | **recommended top-tier** | high quality **and actually runs in-browser**; WebGPU only |
| `Xenova/whisper-large-v3` | ~1.5 GB | absolute max quality | ⚠️ often **runs out of memory at inference** in-browser — see below |

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

## Node backend (whisper.cpp + ffmpeg)

The backend runs native whisper.cpp (Metal on macOS, CPU elsewhere), so large models are fine.

| Model alias | Use it for |
|---|---|
| `base` | fast default |
| `small` | better punctuation / non-English |
| `large-v3-turbo` | best speed/quality balance |
| `large-v3` | absolute best quality (largest, slowest) |

Install extra models: `WHISPER_MODELS="base small large-v3-turbo" npm run setup` (or set
`WHISPER_MODELS` in `docker-compose.yml` for Docker). See [backend/README.md](../backend/README.md).

## Decision rule

- Fastest local responsiveness → `small` (either path).
- Best **in-browser** quality → `whisper-large-v3-turbo` (needs WebGPU: Chrome/Edge).
- Best quality overall, latency/memory OK → `large-v3` on the **backend** (not the browser).
- Non-English → set the language explicitly; small models guess poorly on auto-detect.

## Notes

- Model binaries are kept out of git (`backend/.gitignore`); the backend downloads them on
  `npm run setup` / first Docker start.
- The browser caches model files per-browser; clearing site data forces a re-download.
