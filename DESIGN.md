# Local STT — Admin Panel UI Redesign

> Scope: `backend/public/` (the live UI at stt.yapweijun1996.com).
> The root `index.html` (GitHub Pages, in-browser transformers.js demo) is a separate app and is NOT covered here.

## 1. Goals

1. **Professional admin-panel layout** — persistent sidebar navigation + topbar, replacing the single scrolling page.
2. **Unified component system** — one set of tokens and components (card, button, field, badge, table, progress) used everywhere; no ad-hoc inline styles.
3. **Responsive PWA** — works as an installed app on iOS, Android, iPad, and desktop. Safe-area aware, touch-target compliant.
4. **Force-fresh PWA updates** — clients always run the latest deployed source without manual cache clearing.
5. **Zero regression** — all existing behavior preserved: upload + client-side Opus compression, mic record + auto-transcribe, progress chips, transcript + .txt/.srt export, IndexedDB history, API reference, theme toggle, install prompt.

## 2. Information Architecture

Three views routed client-side (show/hide, no page reload — transcription continues across view switches):

| View | Nav label | Contents |
|---|---|---|
| `transcribe` (default) | 🎙 Transcribe | KPI strip → Settings card → Upload + Mic cards (2-col) → Progress card → Transcript card |
| `history` | 🗂 History | Recording history list (IndexedDB) with per-item playback/delete, Clear All; empty state when none |
| `api` | 🔌 API Reference | Existing endpoint docs (GET /health, POST /api/transcribe), curl/JS/Python tabs, Swagger links |

Sidebar footer: live server health summary (engine + GPU) and project tagline.
Topbar: hamburger (mobile), current view title, API health chip, Install button (when installable), theme toggle.

## 3. Layout & Breakpoints

```
Desktop ≥ 1024px          Tablet/Mobile < 1024px
┌──────┬───────────────┐   ┌───────────────────┐
│ Side │ Topbar        │   │ ☰ Topbar          │
│ bar  ├───────────────┤   ├───────────────────┤
│ 248px│ Content       │   │ Content (1 col    │
│ fixed│ max 1100px    │   │ below 760px)      │
│      │               │   │                   │
└──────┴───────────────┘   └───────────────────┘
                            Sidebar = slide-in drawer
                            (scrim + ESC/scrim-tap to close)
```

- **≥ 1024px**: fixed sidebar 248px, content `max-width: 1100px`, KPI strip 4-up, Upload/Mic 2-up.
- **760–1023px**: sidebar becomes overlay drawer; KPI strip 2-up; Upload/Mic still 2-up ≥ 860px.
- **< 760px**: everything 1 column; KPI strip 2-up; full-width buttons.
- **iOS specifics**: `viewport-fit=cover` + `env(safe-area-inset-*)` padding on topbar/sidebar/content; form controls `font-size: 16px` under 760px to prevent focus zoom; touch targets ≥ 44px; `100dvh` for drawer height (not `100vh`).

## 4. Design Tokens

| Token | Dark (default) | Light |
|---|---|---|
| `--bg` | `#0e1015` | `#f3f5f9` |
| `--surface` | `#161922` | `#ffffff` |
| `--surface-2` | `#1d212c` | `#eef1f6` |
| `--border` | `#262c38` | `#dbe1ea` |
| `--text` | `#e7eaf0` | `#101725` |
| `--muted` | `#98a2b3` | `#5b6472` |
| `--primary` | `#5b9cff` | `#2563eb` |
| `--success` | `#34d399` | `#16a34a` |
| `--danger` | `#f87171` | `#dc2626` |
| `--warning` | `#fbbf24` | `#d97706` |

Sidebar is **always dark** (`--side-bg: #11131c`, own text tokens) in both themes — standard admin-panel convention, anchors the layout.

- Radius: `--r-sm: 8px`, `--r-md: 12px`, `--r-lg: 16px`
- Spacing scale: 4 / 8 / 12 / 16 / 20 / 24
- Type scale: 11 (labels) / 12.5 (meta) / 13.5 (body-sm) / 14.5 (body) / 16 (card title) / 18 (view title)
- Soft tints via `color-mix(in srgb, var(--primary) 12%, transparent)` for badges/active states.

## 5. Component Inventory (unified)

| Component | Classes | Notes |
|---|---|---|
| Card | `.card` > `.card-head` (title + actions) + `.card-body` | Replaces `.panel`; single source of padding/border/radius |
| Button | `.btn` + `.btn-primary` / `.btn-danger` / `.btn-ghost` + `.btn-lg` | `.rec` state class kept for the mic button (JS toggles it) |
| Field | `.field` > `.field-label` + control | Selects/inputs/textarea share one control style |
| Badge/chip | `.tag` (+ `.ok` / `.bad`), `.pill`, `.nav-badge` | Tinted backgrounds, not just borders |
| KPI stat | `.kpi` > `.kpi-label` + `.kpi-value` + `.kpi-sub` | New; fed from `/health` |
| Progress | `.bar > i` (+ `.indet`) | Unchanged behavior, restyled |
| Table | `.param-table` | API docs |
| Tabs | `.tabs` > `.tab-btn` / `.tab-pane` | API docs code samples |
| Empty state | `.empty-state` | History view when no recordings |
| Nav item | `.nav-item` (+ `.active`) | Sidebar; badge slot for history count |

All interactive elements: visible `:focus-visible` outline, `min-height: 40px` (48px for primary actions).

## 6. PWA Update Strategy — force latest source

Problem: installed PWAs can keep serving stale cached shells.

Mechanism (3 layers):

1. **Server worker (`sw.js`)** — versioned cache name; `install → skipWaiting()`, `activate → purge old caches + clients.claim()`. Already in place; VERSION bumped each deploy.
2. **Client auto-reload** — new: listen for `controllerchange`; when a new SW takes control **and** the page already had a controller (i.e. this is an update, not first install), `location.reload()` once. Guard: skip the reload while a transcription or recording is in flight — it retries on next update check instead of destroying work.
3. **Aggressive update checks** — new: `registration.update()` on every `visibilitychange → visible` (catches iOS standalone resume) plus hourly interval. Navigations remain network-first in the SW, so HTML is always fresh when online; the reload only matters for the precached shell on installed standalone apps.

Result: deploy → bump `VERSION` in `sw.js` → clients pick it up on next focus/visit and self-reload to the new build.

## 7. Accessibility

- Drawer: `aria-hidden` sync, scrim click + `Escape` to close, focus not trapped (small app).
- `aria-current="page"` on the active nav item; `aria-label` on icon-only buttons.
- `prefers-reduced-motion: reduce` kills transitions/animations.
- Color-independent status: health chip uses text + color; progress shows numeric %.

## 8. Test Matrix (verified via Chrome DevTools emulation + real API)

| Surface | Check |
|---|---|
| Desktop 1440px | Sidebar fixed, 2-col cards, KPI 4-up, theme toggle both ways |
| iPad 834px | Drawer opens/closes, content readable, 2-col cards |
| Phone 390px (iPhone) / 360px (Android) | 1-col, drawer + scrim, inputs ≥16px (no iOS zoom), touch targets |
| Function | File pick → compress → transcribe 200 OK; mic record path; history render/delete; tabs; .txt/.srt links |
| PWA | manifest valid, SW registers, version bump triggers client reload |

## 9. Changelog

- **2026-06-10**: Initial redesign implemented (this document). Mic 32 kbps Opus recording + client-side file compression (earlier same day) carried over unchanged.
