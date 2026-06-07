$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$WhisperDir = if ($env:WHISPER_CPP_DIR) { $env:WHISPER_CPP_DIR } else { Join-Path $RootDir "vendor\whisper.cpp" }
# Default models. Add "large-v3-turbo" for best accuracy via $env:WHISPER_MODELS.
$ModelList = if ($env:WHISPER_MODELS) { $env:WHISPER_MODELS -split "\s+" } else { @("base", "small") }

function Require-Command($Name, $InstallHint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    Write-Error "$Name is required. $InstallHint"
  }
}

Require-Command "git" "winget install Git.Git"
Require-Command "cmake" "winget install Kitware.CMake"
Require-Command "ffmpeg" "winget install Gyan.FFmpeg"

New-Item -ItemType Directory -Force -Path (Join-Path $RootDir "vendor") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RootDir "uploads") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RootDir "transcripts") | Out-Null

if (-not (Test-Path (Join-Path $WhisperDir ".git"))) {
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git $WhisperDir
} else {
  git -C $WhisperDir pull --ff-only
}

cmake -S $WhisperDir -B (Join-Path $WhisperDir "build") -DWHISPER_BUILD_TESTS=OFF
cmake --build (Join-Path $WhisperDir "build") --config Release

Push-Location (Join-Path $WhisperDir "models")
try {
  foreach ($Model in $ModelList) {
    .\download-ggml-model.cmd $Model
  }
} finally {
  Pop-Location
}

Write-Host "Setup complete."
Write-Host "whisper-cli: $(Join-Path $WhisperDir 'build\bin\Release\whisper-cli.exe')"
Write-Host "models: $(Join-Path $WhisperDir 'models')"
