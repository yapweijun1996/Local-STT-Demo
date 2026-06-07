# Local STT Demo — in-browser Whisper

A single self-contained `index.html` (vanilla JS, opens from `file://`, **zero dependencies**)
that runs **speech-to-text fully in your browser**. Upload **mp4 / mp3 / wav / m4a / ogg**
or record from your **microphone**, and get a transcript with timestamps — **nothing leaves
your machine**.

Powered by [Transformers.js](https://github.com/huggingface/transformers.js) running
[Whisper](https://github.com/openai/whisper) ONNX models on **WebGPU** (with automatic
**WASM** fallback). Audio is decoded with the browser's native Web Audio API, so there is
**no ffmpeg** — which keeps the whole project MIT-clean and install-free.

## Features

- 🎙️ **Two inputs** — drag-and-drop a file, or record live from the mic.
- 🎞️ **Any common format** — mp4 (audio track), mp3, wav, m4a, ogg. Decoded natively in-browser.
- 🧠 **Model choice** — `whisper-tiny` (fastest), `whisper-base`, `whisper-small`,
  `onnx-community/whisper-large-v3-turbo` (**recommended top-tier** — high quality and
  actually runs in-browser), and `Xenova/whisper-large-v3` (highest quality, but ~2.0 GB and
  often runs out of memory in-browser — see the [model guide](docs/STT_MODEL_GUIDE.md)).
- ⚡ **WebGPU first**, automatic fallback to WASM (CPU) for Safari / Firefox.
- 🌐 **Language** — auto-detect or pick (en / zh / ms / ja / es …).
- ⏱️ **Timestamps** — per-segment timings, plus processing time + ×realtime after each run.
- 🌓 **Theme toggle** — light/dark theme with preference saved locally and fallback to system setting.
- ⬇️ **Export** — download `.txt` and `.srt` subtitles.
- 📲 **Installable PWA** — add to home screen / desktop; UI shell works offline.
- 🔒 **100% local** — model is cached in the browser; audio never uploaded.

## Install as an app (PWA)

When served over **https** (e.g. GitHub Pages) or **http://localhost**, this demo is an
installable Progressive Web App: a service worker caches the UI shell for offline use and the
browser offers an **Install** button (also in the header). It follows the standard PWA setup —
`manifest.webmanifest`, `sw.js`, `offline.html`, and `any` + `maskable` icons.

> ⚠️ **`file://` can't run a service worker.** Double-clicking `index.html` still works for
> transcription, but to *install* it as a PWA you must serve it over http(s). Quick local serve:
> `python3 -m http.server` then open `http://localhost:8000/`. Or use the GitHub Pages deploy below.

The service worker never touches the model downloads (cross-origin CDN/Hugging Face) or any API
call — those always go to the network, so transcription is unaffected by caching.

### Deploy to GitHub Pages

A workflow at [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) publishes this
browser demo to GitHub Pages on every push to `main`. One-time setup: **Settings → Pages →
Build and deployment → Source = "GitHub Actions"**. After it runs, the demo is live (and
installable) at `https://yapweijun1996.github.io/Local-STT-Demo/`. The Node backend is not
deployed there — it needs a server; run it locally (see [`backend/`](backend/)).

## Quick start

1. Open `index.html` in a browser (double-click — `file://` works). Chrome/Edge recommended for WebGPU.
2. Pick a **model** (start with `tiny` to test, use `base`/`small` for real accuracy,
   and `whisper-large-v3-turbo` for the best in-browser quality. `whisper-large-v3` is the
   highest quality but is heavy and may run out of memory — if it errors, fall back to turbo).
3. Pick the **language** (set it explicitly for non-English — e.g. `zh` for Chinese — small models guess poorly on `auto`).
4. **Drop a file** or **Record mic**, then click **Transcribe**.
5. First run downloads the model — on WebGPU: tiny ~145MB · base ~280MB · small ~920MB · turbo ~1.5GB · large ~2.0GB (WASM downloads smaller quantized weights); after that it works offline.

## Optional backend STT service

For API-based use (ERP/agent pipelines, larger files, server-side control), use the bundled Node backend in [`backend/`](backend/).

```bash
cd backend
npm install
npm run setup          # builds whisper.cpp + downloads base/small models by default
npm start              # serves http://localhost:8789
```

That backend exposes `/health` and `POST /api/transcribe` and accepts `base`, `small`, `tiny`,
`large-v3-turbo`, or `large-v3` (if those model files are installed) as model aliases.

## Pushing to GitHub (important)

- **Do not commit model binaries.** `backend` uses `.gitignore` to keep `vendor/`, `models/`,
  `uploads/`, and `transcripts/` local-only.  
- Use `npm run setup` on each machine/environment (or CI job) to fetch models locally.  
- Your current setup has already downloaded large models locally; they are ignored by git and will not
  be pushed.
- If you must version model artifacts in this repo (not recommended for this size), use Git LFS and
  track only specific files after removing `vendor/` from ignore for those paths.

## How it works

```
mp4/mp3/wav ──┐
              ├─→ decodeAudioData() → mix to mono → resample to 16kHz ──┐
mic (MediaRecorder) ┘                                                   │
                                                                        ▼
                            Transformers.js Whisper (WebGPU → WASM)
                                          │
                              text + timestamped segments → .txt / .srt
```

The key design rule: **all inputs converge to one format — 16kHz mono Float32 PCM —**
before reaching the model. Whisper only ever sees one kind of data; the format complexity
is handled at the IO boundary.

## Benchmark: base vs. large-v3-turbo

Measured in-browser on this machine (Apple Silicon Mac, Chrome **WebGPU**), same
20.8s Chinese clip, language set to `zh`. The UI shows
processing time and **×realtime** (audio seconds ÷ processing seconds) after each run.

| Model | Source | Processing | ×realtime | Accuracy on this clip |
|---|---|---:|---:|---|
| `whisper-base` | Xenova | 2.3s | 9.0× | rough — `撕上`, `大雲加`, `Facebook AI`, `XR-7车` |
| `whisper-large-v3-turbo` | onnx-community | 3.3s | 6.2× | clean — `厮杀`, `大赢家`, `Physical AI`, `XR 汽车` |

Takeaway: on a WebGPU machine, **turbo costs only ~1s more but is dramatically more
accurate** (especially for non-English and proper nouns), so the app defaults to turbo
when WebGPU is available and falls back to `base` on WASM. Both run several times faster
than real time. (Numbers exclude the one-time model download; turbo is ~1.5GB on first run.)

## Notes & limits

- **Accuracy is model-dependent.** `tiny` is rough; use `base`/`small` for real work.
  `large-v3-turbo` is high quality and a good default for non-English; `whisper-large` is the
  highest quality option when memory and latency allow.
- **Set the language** for non-English. On `auto`, small models sometimes mis-detect and translate.
- **mp4 must have an audio track.** A video-only mp4 (or an exotic codec) will fail to decode — you'll get a clear error; try Chrome or convert to wav/mp3.
- **WebGPU** is best in Chrome/Edge. Other browsers fall back to WASM (slower but works).
- First model download is large; subsequent runs are cached and offline.
- **Big-model downloads & memory.** The progress bar shows real per-file bytes + ETA (from the
  library's `progress_callback`; the Hugging Face CDN sends no `Timing-Allow-Origin`, so the
  browser's own Resource Timing can't be used). Transient network drops
  (`ERR_CONTENT_LENGTH_MISMATCH`) are retried automatically — already-downloaded files are cached,
  so a retry resumes. `whisper-large-v3` (~2.0 GB) can still run out of memory **at inference**
  in-browser (its quantized decoder runs on CPU/WASM); you'll get a clear "too heavy — try turbo"
  message rather than a cryptic error. **`whisper-large-v3-turbo` is the reliable top-tier.**

## Two flavors in this repo

| | Browser demo (this `index.html`) | Node backend ([`backend/`](backend/)) |
|---|---|---|
| Install | none — double-click | Node + ffmpeg + compile whisper.cpp |
| Engine | Transformers.js (ONNX, WebGPU/WASM) | whisper.cpp (native, Metal/CPU) |
| Best for | "double-click and it works" demo, full privacy | native speed, long files, ERP/agent integration |
| Dependency tree | MIT-clean, no ffmpeg | MIT code; ffmpeg required at runtime |

The browser demo trades some speed for **zero install, zero server**. The
[`backend/`](backend/) is the server-side option when you need native speed or an HTTP API.
See [backend/README.md](backend/README.md) for setup.

## License

MIT — see [LICENSE](LICENSE).

Third-party components (preserve their notices when redistributing):

- **Transformers.js** — Apache-2.0 — https://github.com/huggingface/transformers.js
- **OpenAI Whisper** model weights (ONNX conversions by Xenova) — MIT — https://github.com/openai/whisper

No ffmpeg is bundled or required, so this project carries no LGPL/GPL obligations.
