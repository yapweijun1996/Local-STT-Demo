# Task Log — Local STT Demo

Newest first. Scope: `backend/public/` live UI (stt.yapweijun1996.com) unless noted.

## 2026-06-13 — Diarization offline mode (token-free runtime)

- `diarize.py`: HF token is now only needed ONCE to download the gated models.
  Set `HF_HUB_OFFLINE=1` (or `DIARIZATION_OFFLINE=1`) and the pipeline loads from the
  local HF cache with no token / no network — recommended prod setup (pre-warm cache
  once with a token, then run offline). `_offline()` forces `local_files_only=True` on
  every `hf_hub_download` call + sets the offline env/constant. `is_available()` and
  `/health.diarization` now report `offline` and treat an offline flag as a valid entry
  (no token required). Verified: ran the 2-speaker test with `env -u HF_TOKEN
  HF_HUB_OFFLINE=1` → Speaker 1/2 labels in 3.0s, zero token used.
- **Enabled in production (pm2 `local-stt`, port 6601):** pre-warmed the pyannote models
  into pm2's `HF_HOME=/Users/yapweijun/.cache/stt-models` once (with token), then added
  `ENABLE_DIARIZATION=1` + `HF_HUB_OFFLINE=1` to ecosystem.config.cjs and `pm2 restart
  --update-env`. `/health` on 6601 now reports diarization available + offline + tokenPresent:false.
  Verified a real whisper-cpp(Metal) job with diarize=1 → diarized 2 speakers, token-free.
  Note: whisper-cpp segments are coarser than faster-whisper, so speaker boundaries are
  less granular on the whisper-cpp engine.

## 2026-06-13 — Speaker diarization (pyannote)

- New `backend/src/diarize.py`: pyannote.audio `speaker-diarization-3.1` pipeline (lazy
  singleton, MPS/CUDA/CPU auto). `diarize_file()` runs on the FULL wav → global-timestamp
  speaker turns; `assign_speakers()` tags merged transcript segments by max temporal overlap.
  Degrades gracefully (returns []/False) when pyannote or `HF_TOKEN` is missing — never raises.
- `server.py`: `/api/transcribe` accepts `diarize=1`; diarization runs once after chunk merge
  (sidesteps cross-chunk speaker-id consistency). Segments gain a `speaker` field; result gains
  `diarized` + `speakerCount`. `/health` reports diarization availability. `ENABLE_DIARIZATION`
  env kill switch.
- `requirements.txt`: added `pyannote.audio` (heavy — pulls torch). **Setup required**: pip
  install, accept model terms on huggingface.co (speaker-diarization-3.1 + segmentation-3.0),
  set `HF_TOKEN`.
- UI (`backend/public/index.html`): Settings "Detect speakers" toggle; segment view shows a
  `Speaker N` badge; .txt export groups by speaker, .srt prefixes `Speaker N:`.
- Compat shims in diarize.py for the installed stack (torch 2.8 / hub 1.8 / speechbrain 1.1):
  rewrite `use_auth_token`→`token`, force `torch.load(weights_only=False)`, soft-fail
  speechbrain's optional lazy imports (k2_fsa, etc.).
- **Verified end-to-end 2026-06-13**: 2-speaker test audio → faster-whisper(base)+pyannote 3.1
  correctly labelled Speaker 1/2/1/2/1, via both the API and the web UI (screenshot taken).
  Tested on an isolated server (port 6602, Redis db 15) so it never touched the pm2 `local-stt`
  production instance on 6601. Production (pm2) still runs with diarization OFF (no HF_TOKEN in
  its env) — enabling it there needs HF_TOKEN added to ecosystem.config.cjs + pm2 restart.

## 2026-06-10 — Admin-panel UI redesign + PWA force reload

- Rewrote `backend/public/index.html` as an admin-panel layout per [DESIGN.md](DESIGN.md):
  sidebar (always dark) + topbar + 3 client-routed views (Transcribe / History / API Reference),
  KPI strip fed from `/health`, unified component system (card / btn / field / tag / kpi / empty-state).
- Responsive: fixed sidebar ≥1024px, slide-in drawer + scrim below; 1-col <760px;
  iOS safe-area insets, 16px controls under 760px (no focus zoom), 100dvh shell.
- PWA force reload: client listens for `controllerchange` and reloads once on SW update
  (skipped while recording/transcribing); `registration.update()` on visibilitychange + hourly.
  `sw.js` VERSION bumped (now `2026-06-10-3`).
- All existing logic preserved: upload + compression, mic flow, progress chips, transcript,
  .txt/.srt export, IndexedDB history (+ sidebar count badge + empty state), API docs tabs, theme.
- Verified via Chrome DevTools: desktop 1440 / phone 390 (drawer), light+dark themes,
  end-to-end transcription HTTP 200, SW update auto-reload (navigation type `reload`).

## 2026-06-10 — Audio upload compression

- Mic: `MediaRecorder` now records 32 kbps Opus mono (`audioBitsPerSecond` + mime probe,
  Safari falls back to audio/mp4). Measured 3.94× smaller than browser default (~124 kbps).
- File uploads >5 MB: decoded in browser → 16 kHz mono → 32 kbps Opus via WebCodecs
  `AudioEncoder` + vendored `webm-muxer@5.1.4` (`backend/public/vendor/webm-muxer.min.js`, MIT).
  Measured: 5.0 MB WAV → 140.9 KB in 174 ms (36×). Fallback to original on any failure.
- Server unchanged (ffmpeg decodes webm/ogg/mp4 natively).
