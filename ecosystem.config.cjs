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
        // whisper.cpp engine (Metal GPU) — auto-detected if binary exists
        WHISPER_CPP_DIR: "/Users/yapweijun/Documents/GitHub/Local-STT-Demo/backend/vendor/whisper.cpp",
      },
      autorestart: true,
      max_restarts: 5,
      max_memory_restart: "8G",
    },
  ],
};
