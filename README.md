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
- 🧠 **Model choice** — `whisper-tiny` (fastest) · `whisper-base` · `whisper-small` (most accurate).
- ⚡ **WebGPU first**, automatic fallback to WASM (CPU) for Safari / Firefox.
- 🌐 **Language** — auto-detect or pick (en / zh / ms / ja / es …).
- ⏱️ **Timestamps** — per-segment timings, plus processing time + ×realtime after each run.
- ⬇️ **Export** — download `.txt` and `.srt` subtitles.
- 🔒 **100% local** — model is cached in the browser; audio never uploaded.

## Quick start

1. Open `index.html` in a browser (double-click — `file://` works). Chrome/Edge recommended for WebGPU.
2. Pick a **model** (start with `tiny` to test, use `base`/`small` for real accuracy).
3. Pick the **language** (set it explicitly for non-English — e.g. `zh` for Chinese — small models guess poorly on `auto`).
4. **Drop a file** or **Record mic**, then click **Transcribe**.
5. First run downloads the model (tiny ~40MB · base ~150MB · small ~480MB); after that it works offline.

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
20.8s Chinese clip (`sample-projects/test.mp4`), language set to `zh`. The UI shows
processing time and **×realtime** (audio seconds ÷ processing seconds) after each run.

| Model | Source | Processing | ×realtime | Accuracy on this clip |
|---|---|---:|---:|---|
| `whisper-base` | Xenova | 2.3s | 9.0× | rough — `撕上`, `大雲加`, `Facebook AI`, `XR-7车` |
| `whisper-large-v3-turbo` | onnx-community | 3.3s | 6.2× | clean — `厮杀`, `大赢家`, `Physical AI`, `XR 汽车` |

Takeaway: on a WebGPU machine, **turbo costs only ~1s more but is dramatically more
accurate** (especially for non-English and proper nouns), so the app defaults to turbo
when WebGPU is available and falls back to `base` on WASM. Both run several times faster
than real time. (Numbers exclude the one-time model download; turbo is ~600MB on first run.)

## Notes & limits

- **Accuracy is model-dependent.** `tiny` is rough; use `base`/`small` for real work. Whisper can hallucinate or repeat on noisy/silent audio.
- **Set the language** for non-English. On `auto`, small models sometimes mis-detect and translate.
- **mp4 must have an audio track.** A video-only mp4 (or an exotic codec) will fail to decode — you'll get a clear error; try Chrome or convert to wav/mp3.
- **WebGPU** is best in Chrome/Edge. Other browsers fall back to WASM (slower but works).
- First model download is large; subsequent runs are cached and offline.

## Comparison: this vs. a native whisper.cpp server

This repo's [`sample-projects/`](sample-projects/) contains a Node + whisper.cpp + ffmpeg
STT server (faster and more accurate, but requires installing Node, ffmpeg, and compiling
whisper.cpp). This browser demo trades some speed/accuracy for **zero install, zero server,
and an MIT-clean dependency tree** — ideal as a "double-click and it works" demo.

## License

MIT — see [LICENSE](LICENSE).

Third-party components (preserve their notices when redistributing):

- **Transformers.js** — Apache-2.0 — https://github.com/huggingface/transformers.js
- **OpenAI Whisper** model weights (ONNX conversions by Xenova) — MIT — https://github.com/openai/whisper

No ffmpeg is bundled or required, so this project carries no LGPL/GPL obligations.
