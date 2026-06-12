module.exports = {
  apps: [
    {
      name: "local-stt",
      script: ".venv/bin/python",
      args: "-m src.server",
      cwd: "/Users/yapweijun/Documents/GitHub/Local-STT-Demo/backend",
      interpreter: "none",
      env: {
        PORT: "6601",
        WHISPER_MODEL: "large-v3",
        STT_ENGINE: "whisper-cpp",
        HF_HOME: "/Users/yapweijun/.cache/stt-models",
        TRANSCRIPTION_TIMEOUT: "0",
        JOB_TTL: String(12 * 60 * 60),
        WORKER_CONCURRENCY: "2",
        UPLOADS_MAX_BYTES: String(5 * 1024 * 1024 * 1024),
        ORPHAN_MAX_AGE_SECONDS: String(24 * 60 * 60),
        CLEANUP_INTERVAL_SECONDS: String(30 * 60),
        ENABLE_CHUNK_TRANSCRIPTION: "1",
        CHUNK_SECONDS: "180",
        CHUNK_OVERLAP_SECONDS: "5",
        // Speaker diarization (pyannote). Models pre-warmed into HF_HOME above;
        // HF_HUB_OFFLINE=1 runs them from cache with no token / no network.
        ENABLE_DIARIZATION: "1",
        HF_HUB_OFFLINE: "1",
        REDIS_URL: "redis://127.0.0.1:6379/0",
        // whisper.cpp engine (Metal GPU) — auto-detected if binary exists
        WHISPER_CPP_DIR: "/Users/yapweijun/Documents/GitHub/Local-STT-Demo/backend/vendor/whisper.cpp",
      },
      autorestart: true,
      max_restarts: 5,
      max_memory_restart: "8G",
    },
  ],
};
