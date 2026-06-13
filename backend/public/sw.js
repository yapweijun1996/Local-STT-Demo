/* Local STT Demo — service worker (zero-build, per KB PWA blueprint).
   - Versioned cache; activate purges old caches.
   - Navigation: network-first → cache → offline.html.
   - Same-origin GET assets: stale-while-revalidate.
   - NEVER touches: non-GET (transcribe POST), cross-origin (CDN / HF model
     downloads), or API endpoints (/api, /health) — those must hit the network. */
const VERSION = "2026-06-13-6";
const CACHE = `local-stt-${VERSION}`;
const SHELL = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./offline.html",
  "./vendor/webm-muxer.min.js",
  "./icons/icon.svg",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("message", (e) => {
  if (e.data && e.data.type === "SKIP_WAITING") self.skipWaiting();
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;                       // never cache POST (transcribe)
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;        // CDN / HF model fetches → network
  if (url.pathname.startsWith("/api") || url.pathname.endsWith("/health")) return; // backend API

  // Navigations: network-first so users always get fresh HTML when online.
  if (req.mode === "navigate") {
    e.respondWith(
      fetch(req)
        .then((res) => { cachePut(req, res.clone()); return res; })
        .catch(() => caches.match(req).then((c) => c || caches.match("./offline.html")))
    );
    return;
  }

  // Other same-origin assets: stale-while-revalidate.
  e.respondWith(
    caches.match(req).then((cached) => {
      const network = fetch(req)
        .then((res) => { if (res && res.ok && res.type === "basic") cachePut(req, res.clone()); return res; })
        .catch(() => cached);
      return cached || network;
    })
  );
});

function cachePut(req, res) {
  caches.open(CACHE).then((c) => c.put(req, res)).catch(() => {});
}
