"""Speaker diarization — "who spoke when" — via pyannote.audio.

Runs OUTSIDE the STT engines: pyannote analyses the full 16 kHz mono WAV and
returns global-timestamped speaker turns. The transcript segments (produced by
whisper.cpp / faster-whisper, possibly across chunks) are then tagged by
maximum temporal overlap against those turns. Diarizing the whole file once —
rather than per chunk — is what keeps "Speaker 1" the same person across chunk
boundaries.

Setup (heavy, optional):
  1. pip install -r requirements.txt           # pulls pyannote.audio + torch
  2. Accept the model terms on huggingface.co:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
  3. export HF_TOKEN=hf_xxx                     # your read token

If any of that is missing, is_available() returns False and transcription
proceeds normally without speaker labels.

Token-free operation after first download:
  The HF token is only needed ONCE to download the gated models. Once they are
  in the local HF cache, set HF_HUB_OFFLINE=1 (or DIARIZATION_OFFLINE=1) and the
  pipeline loads from cache with no token and no network. This is the recommended
  production setup: pre-warm the cache once with a token, then run offline.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

log = logging.getLogger(__name__)

DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _offline() -> bool:
    """Run from the local HF cache only — no token, no network.

    True when explicitly requested (HF_HUB_OFFLINE=1 / DIARIZATION_OFFLINE=1) OR
    when no token is present (the only way it can work then is from a prior cache,
    so force local-only rather than hit the Hub expecting auth). When a token IS
    present and no offline flag is set, we stay online so first-run downloads work.
    """
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("DIARIZATION_OFFLINE") == "1":
        return True
    return _hf_token() is None


# Lazy, process-wide singleton — loading the pipeline costs seconds + RAM.
_pipeline: Any = None
_pipeline_failed = False
_lock = threading.Lock()


def _pyannote_installed() -> bool:
    try:
        import pyannote.audio  # noqa: F401

        return True
    except Exception:
        return False


def _explicit_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("DIARIZATION_OFFLINE") == "1"


def is_available() -> bool:
    """True if pyannote is importable and we have a way in: either an HF token
    (can download) or an explicit offline flag (use the already-cached model)."""
    return _pyannote_installed() and (_hf_token() is not None or _explicit_offline())


def status() -> dict[str, Any]:
    """Health-endpoint summary (no model load)."""
    installed = _pyannote_installed()
    has_token = _hf_token() is not None
    offline = _explicit_offline()
    if not installed:
        reason = "pyannote.audio not installed"
    elif not has_token and not offline:
        reason = "no HF_TOKEN (set HF_HUB_OFFLINE=1 to run from cache without a token)"
    elif _pipeline_failed:
        reason = (
            "pipeline failed to load — model not in local cache"
            if offline and not has_token
            else "pipeline failed to load (check model terms accepted)"
        )
    else:
        reason = None
    return {
        "available": installed and (has_token or offline) and not _pipeline_failed,
        "installed": installed,
        "tokenPresent": has_token,
        "offline": offline,
        "model": DIARIZATION_MODEL,
        "reason": reason,
    }


def _patch_hf_token_kwarg() -> None:
    """pyannote 3.x calls hf_hub_download(use_auth_token=...), a kwarg removed in
    huggingface_hub >= 1.0. Rewrite it to token= so we don't have to downgrade the
    hub (faster-whisper / speechbrain / tokenizers share it). Idempotent."""
    import sys

    import huggingface_hub

    orig = huggingface_hub.hf_hub_download
    if getattr(orig, "_token_kwarg_patched", False):
        return

    def patched(*args, **kwargs):
        if "use_auth_token" in kwargs:
            tok = kwargs.pop("use_auth_token")
            if tok is not None and "token" not in kwargs:
                kwargs["token"] = tok
        if _offline():
            # Serve from the local cache only: no network, no token required.
            kwargs.setdefault("local_files_only", True)
        return orig(*args, **kwargs)

    patched._token_kwarg_patched = True
    huggingface_hub.hf_hub_download = patched
    # Rebind modules that already did `from huggingface_hub import hf_hub_download`.
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "hf_hub_download", None) is orig:
                mod.hf_hub_download = patched
        except Exception:
            pass


def _soften_speechbrain_lazy_imports() -> None:
    """speechbrain 1.1's LazyModule eagerly resolves EVERY optional integration
    submodule (k2_fsa, huggingface.wordemb, …) when pyannote enumerates the
    speaker-verification module, and the ones whose heavy deps (k2, fairseq, …) are
    absent raise ImportError. Diarization uses none of them. speechbrain already
    swallows this for `inspect.py` (importutils.py:89) but not for pyannote's path —
    so make any failed lazy import degrade to an empty stub instead of raising."""
    import types

    from speechbrain.utils import importutils as _iu

    LazyModule = _iu.LazyModule
    if getattr(LazyModule, "_lazy_softfail_patched", False):
        return

    orig_ensure = LazyModule.ensure_module

    def safe_ensure(self, stacklevel=1):
        try:
            return orig_ensure(self, stacklevel)
        except Exception:
            if self.lazy_module is None:
                self.lazy_module = types.ModuleType(getattr(self, "target", "sb_stub"))
            return self.lazy_module

    LazyModule.ensure_module = safe_ensure
    LazyModule._lazy_softfail_patched = True


def _patch_torch_load() -> None:
    """torch >= 2.6 flipped torch.load(weights_only) default to True, which rejects
    the custom globals (TorchVersion, omegaconf, …) baked into pyannote/speechbrain
    checkpoints. These are the official, license-gated pyannote weights — trusted —
    so force weights_only=False for them. Idempotent; only touched on the diarize path."""
    import torch

    orig = torch.load
    if getattr(orig, "_weights_only_patched", False):
        return

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    patched._weights_only_patched = True
    torch.load = patched


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_pipeline():
    global _pipeline, _pipeline_failed
    if _pipeline is not None:
        return _pipeline
    if _pipeline_failed:
        return None
    with _lock:
        if _pipeline is not None:
            return _pipeline
        if _pipeline_failed:
            return None
        try:
            if _offline():
                # Belt-and-suspenders for any download path we don't wrap.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                try:
                    import huggingface_hub.constants as _hc

                    _hc.HF_HUB_OFFLINE = True
                except Exception:
                    pass

            from pyannote.audio import Pipeline

            _patch_hf_token_kwarg()
            _patch_torch_load()
            _soften_speechbrain_lazy_imports()
            pipe = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                use_auth_token=_hf_token(),
            )
            if pipe is None:
                # from_pretrained returns None when the token can't access the
                # gated model (terms not accepted / wrong token).
                raise RuntimeError(
                    f"Pipeline.from_pretrained returned None for {DIARIZATION_MODEL!r} — "
                    "accept the model terms on huggingface.co and check HF_TOKEN."
                )
            try:
                pipe.to(_device())
            except Exception as exc:  # device move is best-effort
                log.warning("diarization: could not move pipeline to accelerator (%s)", exc)
            _pipeline = pipe
            log.info("diarization: loaded %s", DIARIZATION_MODEL)
            return _pipeline
        except Exception as exc:
            _pipeline_failed = True
            log.error("diarization: failed to load pipeline (%s)", exc)
            return None


def diarize_file(
    wav_path: str,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """Return speaker turns with GLOBAL timestamps:
    [{"startMs": int, "endMs": int, "speaker": "Speaker 1"}, ...] sorted by start.

    Returns [] if diarization is unavailable or fails (never raises).
    """
    pipe = _get_pipeline()
    if pipe is None:
        return []

    kwargs: dict[str, Any] = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)

    try:
        diarization = pipe(wav_path, **kwargs)
    except Exception as exc:
        log.error("diarization: inference failed on %s (%s)", wav_path, exc)
        return []

    # Map pyannote labels (SPEAKER_00 …) to friendly numbers by first appearance.
    label_to_name: dict[str, str] = {}
    turns: list[dict[str, Any]] = []
    for segment, _track, label in diarization.itertracks(yield_label=True):
        name = label_to_name.get(label)
        if name is None:
            name = f"Speaker {len(label_to_name) + 1}"
            label_to_name[label] = name
        turns.append({
            "startMs": int(round(segment.start * 1000)),
            "endMs": int(round(segment.end * 1000)),
            "speaker": name,
        })
    turns.sort(key=lambda t: (t["startMs"], t["endMs"]))
    return turns


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> int:
    """Tag each transcript segment in-place with a "speaker" by max overlap with
    the diarization turns. Falls back to the nearest turn by midpoint when a
    segment overlaps none. Returns the number of distinct speakers assigned.
    """
    if not turns:
        return 0

    seen: set[str] = set()
    for seg in segments:
        s_start = int(seg.get("startMs") or 0)
        s_end = int(seg.get("endMs") or s_start)

        best_speaker = None
        best_overlap = 0
        for turn in turns:
            ov = _overlap_ms(s_start, s_end, turn["startMs"], turn["endMs"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn["speaker"]

        if best_speaker is None:
            # No overlap (e.g. segment in a gap) — pick nearest turn by midpoint.
            mid = (s_start + s_end) / 2
            best_speaker = min(
                turns,
                key=lambda t: abs(((t["startMs"] + t["endMs"]) / 2) - mid),
            )["speaker"]

        seg["speaker"] = best_speaker
        seen.add(best_speaker)

    return len(seen)
