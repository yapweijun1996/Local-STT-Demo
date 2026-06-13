# Task Log — Local STT Demo

Newest first. Scope: `backend/public/` live UI (stt.yapweijun1996.com) unless noted.

## 2026-06-14 — UI: Simple/Advanced modes + transcript states + PWA install/update/viewport

All in `backend/public/index.html` (+ `sw.js` cache bumps). Verified in Chrome desktop + 390px.

- **Simple / Advanced mode toggle (progressive disclosure)**: new users land in **Simple** —
  a Record/Upload segmented toggle (Record default) → transcript; the Settings card, KPI strip
  and Developer nav are hidden, and Simple submits fast zero-config defaults (whisper-cpp/Metal +
  large-v3, language auto, no diarization). A topbar SVG toggle (immediate CSS hover tooltip)
  switches to **Advanced** = the full existing layout. Mode persisted in `localStorage.stt-mode`.
  Lightweight **i18n** scaffold (en/zh, auto-detected from `navigator.language`) for new strings.
- **Install app → icon button**: replaced the wide text button with a download SVG icon
  (declutters the topbar, esp. mobile) + i18n hover tooltip.
- **Transcript states**: card is hidden when idle, shows a **shimmer skeleton** while loading
  (caption follows the real phase Uploading → Queued → Transcribing, with pulsing dots and a
  cascading shimmer), and the real transcript once any text (partial or final) lands. Live
  partial-text streaming preserved (skeleton only before first words). Empty stat badge
  auto-hides (`#stat:empty`). Respects `prefers-reduced-motion`.
- **PWA fixes** (end-user feedback):
  - Install button is platform-aware: **iOS** (no `beforeinstallprompt`) opens an "Add to Home
    Screen" instruction sheet (Share icon, i18n); **Chromium** uses the native prompt; **installed**
    hides it. Persisted via `localStorage.stt-installed` (set once standalone is ever seen, since
    iOS never fires `appinstalled`) **plus** a bulletproof CSS `@media (display-mode: standalone)
    { #installBtn { display:none } }` so a stale-cached HTML can't show it.
  - **Update prompt instead of silent reload**: `sw.js` install no longer calls `skipWaiting()`;
    the page detects the waiting worker and shows an "Update available" modal (Update / Later,
    i18n). Update → `postMessage({type:"SKIP_WAITING"})` → activate → `controllerchange` → reload.
    Deferred while recording/transcribing.
  - **Viewport locked**: `maximum-scale=1, user-scalable=no` (no pinch-zoom; a11y trade-off).
- **Gotcha logged**: an end user saw the install button inside the installed iOS app — root cause
  was a **stale cached HTML** (the app was installed before these fixes), not a logic bug
  (simulating standalone confirmed the current JS hides it). Fix path: deploy + remove/re-add to
  home screen, and the CSS media-query hide as the cache-proof backstop. Also a real bug fixed:
  the iOS sheet markup had to live **before** `<script>` or the `ui` lookups returned null.

## 2026-06-13 — Upgrade diarization to pyannote community-1 (pyannote.audio 4.0, Python 3.12)

Swapped the diarization model from `speaker-diarization-3.1` to `speaker-diarization-community-1`
(pyannote.audio 4.0) — markedly better speaker counting/assignment (the root cause of inaccurate
labels), still free/local/offline. Official DER drops on most benchmarks (AMI-SDM 22.7→19.9,
AliMeeting 24.5→20.3, MSDWild 25.4→22.8).

- **Python rebuild**: 4.0 needs Python ≥3.10; the prod venv was 3.9. Built a new `.venv312`
  alongside (old `.venv` kept for rollback) and pointed pm2 at it.
  - ⚠️ **Homebrew `python@3.12` was unusable**: broken `libexpat` linkage (`pyexpat` →
    `Symbol not found: _XML_SetAllocTrackerActivationThreshold`, hard-linked to system
    `/usr/lib/libexpat.1.dylib`), which breaks `pip`. Fix: installed **`uv`** and used a
    `uv`-managed **standalone CPython 3.12** (self-contained libexpat) — sidesteps the whole
    Homebrew/Xcode-CLT mess.
- **`diarize.py` (4.0 API)**: default model → community-1; `Pipeline.from_pretrained(token=…)`
  (legacy `use_auth_token` removed); handle new `DiarizeOutput.speaker_diarization`; deleted the
  3.x `torch.load` / speechbrain / `hf_hub_download` monkeypatches — 4.0 loads cleanly without them.
- **`requirements.txt`**: `pyannote.audio>=4.0,<5.0`.
- **`ecosystem.config.cjs`**: `script` → `.venv312/bin/python`; `DIARIZATION_MODEL=community-1`.
- ⚠️ **HF cache gotcha**: community-1 first downloaded to the *default* `~/.cache/huggingface/hub`,
  but prod sets `HF_HOME=/Users/.../stt-models` → offline load FAILED (model not in that cache).
  Fix: re-downloaded with `HF_HOME=stt-models` so it lives in `stt-models/hub`. (3.1 didn't hit
  this because old pyannote used `~/.cache/torch/pyannote`; 4.0 uses the HF cache.)
- **Verified**: community-1 loads+infers under 4.0 + MPS; loads OFFLINE under prod HF_HOME with no
  token; full server boots under `.venv312` (isolated :6699 smoke test, separate Redis db); prod
  E2E transcribe+diarize on :6601 clean. NOTE: multi-speaker accuracy A/B still needs a real
  interview clip — jfk.wav is single-speaker. The HF token used for download should be rotated.

## 2026-06-13 — Diarization accuracy: default-on + speaker-count hint + word-level splitting

Three rounds addressing "speaker detection is inaccurate / I don't see speakers".

- **Default ON** (`backend/public/index.html`): the "Detect speakers (diarization)" toggle now
  ships `checked`. Users kept missing labels because it was off and the toggle is read only at
  submit time. (`sw.js` VERSION bumped.)
- **Speaker-count hint** (`server.py` + `index.html`): `/api/transcribe` gains a `speakers`
  field (1–8; ''/auto → estimate), parsed by `_parse_speakers()` and passed to pyannote's
  `num_speakers`. UI adds a "How many speakers?" dropdown under the toggle (Auto default,
  disabled when diarization is off). **Passing the known count is the single biggest accuracy
  lever** — auto-estimation often mis-counts (splits one person into several or merges two).
- **Boundary splitting** (`diarize.py`): `assign_speakers()` now returns `(segments, count)` and
  SPLITS a transcript segment that genuinely spans a speaker change (dominant speaker < 85% of
  overlapped time, each piece ≥ 800 ms) instead of mislabeling the whole thing as one person.
  Single-speaker / tiny-spillover segments stay whole.
- **Word-level splitting** (`transcribe.py`, `whisper_cpp.py`, `server.py`, `diarize.py`): when a
  split segment has word timestamps, it breaks at WORD granularity (precise) rather than
  time-proportional text. faster-whisper: `word_timestamps=True` on both paths. whisper.cpp:
  parse the existing `-ojf` sub-word tokens into words (merge BPE pieces on leading space, skip
  special tokens). `words` are threaded through chunk-merge with correct offsets, then stripped
  from the final segments + per-chunk `partialSegments` so they never bloat the API/Redis payload.
  `_split_by_words()` labels each word by overlapping turn, groups consecutive same-speaker words,
  and folds sub-800 ms groups to kill one-word flicker.
- **Verified**: both engines emit real word timestamps on jfk.wav (whisper-cpp 5/seg,
  faster-whisper 22/seg); unit-tested splits (clean word-boundary break; single-speaker and
  tiny-spillover stay whole; stray short word folds back); E2E both engines + diarize run clean,
  leak no `words`. NOTE: jfk.wav is single-speaker so the split path only shows on multi-speaker
  audio — test an interview with the speaker count set for the real effect.

## 2026-06-13 — User-selectable STT engine (whisper-cpp vs faster-whisper)

- **UI (`backend/public/index.html`)**: new **Engine** selector in Settings — Auto (server
  default) / `whisper-cpp · Metal GPU (fast)` / `faster-whisper · CPU (slower, be patient)`.
  The Model dropdown now filters to models installed **for the chosen engine** (driven by
  `/health` per-engine `installed` maps), auto-correcting the pick when the current model isn't
  available on that engine. Selecting faster-whisper shows an amber warning ("CPU-only, ~13×
  slower than Metal, long files take minutes") and forces `useGpu=0` on submit (whisper-cpp
  requires GPU=1). Engine sent as the `engine` form field only when not Auto.
- **Backend (`backend/src/server.py`)**: `/api/transcribe` validates `engine` — unknown engine
  → `400`; `faster-whisper` + a model not pre-downloaded → `400` with a clear message (instead
  of failing cryptically under `HF_HUB_OFFLINE=1`). Factored `_fw_model_installed()` helper,
  reused by `/health`. The worker already dispatched to `faster_transcribe` for non-cpp engines.
- **`backend/public/sw.js`**: VERSION bumped `2026-06-10-4` → `2026-06-13-1` so the new UI ships
  past the PWA precache.
- **Model**: pre-downloaded `Systran/faster-whisper-large-v3` (~3 GB) into pm2's
  `HF_HOME=/Users/yapweijun/.cache/stt-models` so faster-whisper has at least large-v3 offline.
- **Benchmark (this Mac, jfk.wav ~11 s, large-v3)**: whisper-cpp Metal ~1.1 s vs faster-whisper
  CPU ~14.5 s = **~13× faster**, identical text. faster-whisper only wins on NVIDIA CUDA. See
  [docs/STT_MODEL_GUIDE.md](docs/STT_MODEL_GUIDE.md).
- **Verified end-to-end 2026-06-13**: API rejects `engine=bogus` + faster-whisper/`tiny`
  (undownloaded); faster-whisper + large-v3 transcribes jfk.wav → correct text; UI engine/model
  filtering + warning confirmed in Chrome (screenshot). pm2 `local-stt` (6601) restarted clean,
  default engine still whisper-cpp.

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
