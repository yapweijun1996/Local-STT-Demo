import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import express from "express";
import multer from "multer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const whisperDir = process.env.WHISPER_CPP_DIR ?? path.join(rootDir, "vendor", "whisper.cpp");
const whisperBin = process.env.WHISPER_BIN ?? (
  process.platform === "win32"
    ? path.join(whisperDir, "build", "bin", "Release", "whisper-cli.exe")
    : path.join(whisperDir, "build", "bin", "whisper-cli")
);
const modelsDir = path.join(whisperDir, "models");
const defaultModel = process.env.WHISPER_MODEL ?? path.join(modelsDir, "ggml-base.bin");
// Keys map to ggml model files. Run `npm run setup` (or set WHISPER_MODELS) to fetch them.
const modelRegistry = {
  tiny: path.join(modelsDir, "ggml-tiny.bin"),
  base: path.join(modelsDir, "ggml-base.bin"),
  small: path.join(modelsDir, "ggml-small.bin"),
  "large-v3-turbo": path.join(modelsDir, "ggml-large-v3-turbo.bin")
};
const uploadDir = process.env.UPLOAD_DIR ?? path.join(rootDir, "uploads");
const transcriptDir = process.env.TRANSCRIPT_DIR ?? path.join(rootDir, "transcripts");
const ffmpegBin = process.env.FFMPEG_BIN ?? "ffmpeg";
const port = Number(process.env.PORT ?? 8789);

await fs.mkdir(uploadDir, { recursive: true });
await fs.mkdir(transcriptDir, { recursive: true });

const upload = multer({
  dest: uploadDir,
  limits: {
    fileSize: Number(process.env.MAX_AUDIO_BYTES ?? 200 * 1024 * 1024)
  }
});

const app = express();

// CORS: allow the sibling single-file browser client (served from file://, Origin
// "null") to call this STT service. Multipart POST is a simple request (no preflight),
// but OPTIONS is handled defensively.
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }
  next();
});

app.get("/favicon.ico", (_req, res) => {
  res.status(204).end();
});
app.use(express.static(path.join(rootDir, "public")));

app.get("/health", async (_req, res) => {
  const installed = {};
  for (const [key, p] of Object.entries(modelRegistry)) {
    installed[key] = await exists(p);
  }
  res.json({
    ok: (await exists(whisperBin)) && Object.values(installed).some(Boolean),
    whisperBin,
    defaultModel,
    models: modelRegistry,
    installed,
    uploadDir,
    transcriptDir,
    useGpu: process.env.WHISPER_USE_GPU === "1"
  });
});

app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "Upload an audio/video file in the multipart field named 'audio'." });
    return;
  }

  const startedAt = Date.now();
  const id = `${Date.now()}-${crypto.randomUUID()}`;
  const originalName = req.file.originalname || "audio";
  const sourcePath = req.file.path;
  const wavPath = path.join(uploadDir, `${id}.wav`);
  const outputBase = path.join(transcriptDir, id);
  const outputJson = `${outputBase}.json`;
  const language = String(req.body.language || process.env.LANGUAGE || "auto");
  const modelInput = req.body.model;
  const prompt = req.body.prompt ? String(req.body.prompt) : process.env.WHISPER_PROMPT;
  const useGpu = String(req.body.useGpu ?? process.env.WHISPER_USE_GPU ?? "0") === "1";

  try {
    const model = resolveModel(modelInput);
    await assertFile(whisperBin, "whisper-cli binary");
    await assertFile(model, "Whisper ggml model");
    // ffmpeg decodes ANY container (mp4/mp3/wav/m4a/ogg/webm-from-mic) → 16kHz mono wav.
    await run(ffmpegBin, [
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      sourcePath,
      "-ar",
      "16000",
      "-ac",
      "1",
      wavPath
    ]);

    const args = [
      "-m",
      model,
      "-f",
      wavPath,
      "-l",
      language,
      "-oj",
      "-ojf",
      "-of",
      outputBase,
      "-np"
    ];

    if (!useGpu) args.push("-ng");
    if (prompt) args.push("--prompt", prompt);

    const whisperRun = await run(whisperBin, args);
    const raw = JSON.parse(await fs.readFile(outputJson, "utf8"));
    const segments = (raw.transcription ?? []).map((segment) => ({
      startMs: segment.offsets?.from ?? null,
      endMs: segment.offsets?.to ?? null,
      text: String(segment.text ?? "").trim()
    }));
    const text = segments.map((segment) => segment.text).join(" ").replace(/\s+/g, " ").trim();

    res.json({
      id,
      text,
      language: raw.result?.language ?? language,
      model: raw.model?.type ?? path.basename(model),
      durationMs: Date.now() - startedAt,
      segments,
      files: {
        originalName,
        transcriptJson: outputJson
      },
      raw,
      diagnostics: {
        whisperStdout: whisperRun.stdout.trim(),
        whisperStderr: whisperRun.stderr.trim()
      }
    });
  } catch (error) {
    res.status(500).json({
      error: error.message,
      originalName
    });
  } finally {
    // Privacy/disk: never retain raw audio. Always remove the uploaded source and
    // the intermediate wav. The transcript JSON is redundant with the HTTP
    // response, so drop it too unless KEEP_TRANSCRIPTS=1.
    await safeUnlink(sourcePath);
    await safeUnlink(wavPath);
    if (process.env.KEEP_TRANSCRIPTS !== "1") {
      await safeUnlink(outputJson);
    }
  }
});

app.listen(port, () => {
  console.log(`Local STT backend (whisper.cpp) listening on http://localhost:${port}`);
});

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function safeUnlink(filePath) {
  try {
    await fs.unlink(filePath);
  } catch {
    /* already gone or never created */
  }
}

async function assertFile(filePath, label) {
  if (!(await exists(filePath))) {
    throw new Error(`${label} not found: ${filePath}. Run 'npm run setup' to build whisper.cpp and download models.`);
  }
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: rootDir });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`${command} exited with code ${code}\n${stderr || stdout}`));
      }
    });
  });
}

function resolveModel(input) {
  if (!input) return defaultModel;

  const requested = String(input);
  if (modelRegistry[requested]) return modelRegistry[requested];

  if (process.env.ALLOW_CUSTOM_MODEL_PATH === "1") {
    return requested;
  }

  throw new Error(
    `Unsupported model '${requested}'. Use one of: ${Object.keys(modelRegistry).join(", ")}, or set ALLOW_CUSTOM_MODEL_PATH=1.`
  );
}
