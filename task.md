# Task Log — Local STT Demo

Newest first. Scope: `backend/public/` live UI (stt.yapweijun1996.com) unless noted.

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
