# STT Model Choice Guide

## 2026-06-07 recommended defaults

Your repo has two transcription paths:

- **Browser (Transformers.js + ONNX)**
  - Best practical quality/experience: `whisper-large-v3-turbo`
  - Absolute max quality: `Xenova/whisper-large` (higher VRAM + slower load)
  - Light/faster: `Xenova/whisper-base` or `Xenova/whisper-small`

- **Node backend (whisper.cpp + ffmpeg)**
  - Best quality: `large-v3` (`ggml-large-v3.bin`)
  - Better speed/accuracy balance: `large-v3-turbo`

## Decision rule

- If you need fastest local responsiveness: `small` (both browser/backend)
- If you need non-English quality and can run WebGPU: `whisper-large-v3-turbo` (browser)
- If you need best quality and can afford latency/memory: `large-v3` (backend)

## Notes

- `large-v3` downloads and memory use are much larger than `large-v3-turbo`.
- The browser path still caches files per browser model cache.
- Backend and Docker keep all model binaries out of git by default in `.gitignore`.
