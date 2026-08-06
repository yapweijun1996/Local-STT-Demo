#!/usr/bin/env python3
"""Cloudflare-safe command-line client for the Local STT resumable upload API.

The client keeps only one server-sized chunk in memory at a time:
  POST /api/upload/init
  PUT  /api/upload/{uploadId}/chunk/{index}
  POST /api/upload/{uploadId}/complete
  GET  /api/job/{jobId} (unless --no-poll is used)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://stt.yapweijun1996.com"
DEFAULT_CHUNK_SIZE = 80 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 10 * 60
DEFAULT_RETRIES = 2
DEFAULT_POLL_INTERVAL_SECONDS = 1.5
DEFAULT_POLL_TIMEOUT_SECONDS = 12 * 60 * 60


class ApiError(RuntimeError):
    """An HTTP/API failure with enough context for a CLI user."""

    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _json_body(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _body_message(raw: bytes) -> str:
    body = _json_body(raw)
    message = body.get("error")
    if isinstance(message, str) and message.strip():
        return message.strip()
    text = raw.decode("utf-8", errors="replace").strip()
    if "<html" in text.lower() or "<!doctype" in text.lower():
        return "Cloudflare returned an HTML error page; the request was rejected before reaching the STT server."
    return text[:500] or "request failed"


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _headers(api_key: str | None, content_type: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "local-stt-upload/1.0"}
    if content_type:
        headers["Content-Type"] = content_type
    if api_key:
        headers["X-STT-API-Key"] = api_key
    return headers


def _request(
    *,
    base_url: str,
    path: str,
    method: str,
    api_key: str | None,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        _url(base_url, path),
        data=body,
        headers=_headers(api_key, content_type),
        method=method,
    )
    if body is not None:
        request.add_header("Content-Length", str(len(body)))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ConnectionError(f"{method} {path}: {reason}") from exc
    except TimeoutError as exc:
        raise ConnectionError(f"{method} {path}: request timed out") from exc


def _json_request(
    *,
    base_url: str,
    path: str,
    method: str,
    api_key: str | None,
    payload: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    status, raw = _request(
        base_url=base_url,
        path=path,
        method=method,
        api_key=api_key,
        body=body,
        content_type="application/json" if body is not None else None,
        timeout=timeout,
    )
    if not 200 <= status < 300:
        raise ApiError(status, f"HTTP {status}: {_body_message(raw)}", raw.decode("utf-8", errors="replace"))
    value = _json_body(raw)
    if not value:
        raise ApiError(status, f"HTTP {status}: server returned invalid JSON")
    return value


def _backoff_seconds(attempt: int) -> float:
    return min(8.0, 0.5 * (2**attempt))


def _chunk_count(size: int, chunk_size: int) -> int:
    if size <= 0:
        raise ValueError("file must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    return math.ceil(size / chunk_size)


class SttUploadClient:
    """Small API client that keeps the large-file protocol in one place."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(0, retries)

    def init_upload(self, *, filename: str, size: int, content_type: str) -> dict[str, Any]:
        return _json_request(
            base_url=self.base_url,
            path="/api/upload/init",
            method="POST",
            api_key=self.api_key,
            payload={"filename": filename, "size": size, "contentType": content_type},
            timeout=self.timeout,
        )

    def _put_chunk(self, *, upload_id: str, index: int, data: bytes, content_type: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                status, raw = _request(
                    base_url=self.base_url,
                    path=f"/api/upload/{upload_id}/chunk/{index}",
                    method="PUT",
                    api_key=self.api_key,
                    body=data,
                    content_type=content_type or "application/octet-stream",
                    timeout=self.timeout,
                )
                if 200 <= status < 300:
                    value = _json_body(raw)
                    if not value:
                        raise ApiError(status, f"HTTP {status}: server returned invalid JSON")
                    return value
                error = ApiError(status, f"HTTP {status}: {_body_message(raw)}", raw.decode("utf-8", errors="replace"))
                # A 4xx response is deterministic (including Cloudflare 413); retrying
                # the same bytes cannot fix it. Only transient 5xx responses are retried.
                if status < 500 or attempt >= self.retries:
                    raise error
                last_error = error
            except (ConnectionError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise
            if last_error is not None:
                time.sleep(_backoff_seconds(attempt))
        raise last_error or RuntimeError("chunk upload failed")

    def complete_upload(self, *, upload_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        return _json_request(
            base_url=self.base_url,
            path=f"/api/upload/{upload_id}/complete",
            method="POST",
            api_key=self.api_key,
            payload=settings,
            timeout=self.timeout,
        )

    def cancel_upload(self, upload_id: str) -> None:
        try:
            status, _ = _request(
                base_url=self.base_url,
                path=f"/api/upload/{upload_id}",
                method="DELETE",
                api_key=self.api_key,
                timeout=self.timeout,
            )
            if status not in (200, 404):
                print(f"warning: cleanup returned HTTP {status}", file=sys.stderr)
        except (ConnectionError, TimeoutError) as exc:
            print(f"warning: cleanup failed: {exc}", file=sys.stderr)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return _json_request(
            base_url=self.base_url,
            path=f"/api/job/{job_id}",
            method="GET",
            api_key=self.api_key,
            payload=None,
            timeout=self.timeout,
        )

    def upload_file(
        self,
        *,
        file_path: Path,
        content_type: str,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        size = file_path.stat().st_size
        init = self.init_upload(filename=file_path.name, size=size, content_type=content_type)
        upload_id = str(init.get("uploadId", ""))
        chunk_size = int(init.get("chunkSize", 0))
        total_chunks = int(init.get("totalChunks", 0))
        expected_chunks = _chunk_count(size, chunk_size)
        if not upload_id or chunk_size <= 0 or total_chunks != expected_chunks:
            raise ApiError(200, "server returned invalid upload metadata")

        completed = False
        try:
            with file_path.open("rb") as source:
                for index in range(total_chunks):
                    data = source.read(min(chunk_size, size - index * chunk_size))
                    expected_size = min(chunk_size, size - index * chunk_size)
                    if len(data) != expected_size:
                        raise OSError(
                            f"file changed while reading chunk {index}: expected {expected_size} bytes, got {len(data)}"
                        )
                    self._put_chunk(
                        upload_id=upload_id,
                        index=index,
                        data=data,
                        content_type=content_type,
                    )
                    print(
                        f"uploaded chunk {index + 1}/{total_chunks} "
                        f"({min((index + 1) * chunk_size, size)}/{size} bytes)",
                        file=sys.stderr,
                    )
            result = self.complete_upload(upload_id=upload_id, settings=settings)
            completed = True
            return result
        finally:
            if not completed:
                self.cancel_upload(upload_id)

    def poll_job(
        self,
        *,
        job_id: str,
        interval: float,
        timeout: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            result = self.get_job(job_id)
            status = result.get("status")
            if result.get("text") is not None or status in {"done", "error"}:
                if status == "error":
                    raise ApiError(500, str(result.get("error", "transcription failed")))
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} did not finish within {timeout:g} seconds")
            print(
                f"job {job_id}: {status or 'waiting'} "
                f"(queued={result.get('queuedCount', 0)}, running={result.get('runningCount', 0)})",
                file=sys.stderr,
            )
            time.sleep(max(0.1, interval))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload one audio file to Local STT through the Cloudflare-safe resumable API."
    )
    parser.add_argument("--file", required=True, type=Path, help="local audio/video file (mp3, wav, m4a, mp4, ogg, …)")
    parser.add_argument("--base-url", default=os.environ.get("STT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("STT_API_KEY"), help="optional STT_API_KEY / X-STT-API-Key")
    parser.add_argument("--content-type", help="override MIME type (default: inferred from filename)")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--engine", default="whisper-cpp")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--use-gpu", choices=("0", "1"), default="1")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--diarize", choices=("0", "1"), default="0")
    parser.add_argument("--speakers", default="")
    parser.add_argument("--no-poll", action="store_true", help="print the queued job response without waiting for transcription")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--poll-timeout", type=float, default=DEFAULT_POLL_TIMEOUT_SECONDS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="per HTTP request timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="retries for network errors and HTTP 5xx chunk responses")
    parser.add_argument("--dry-run", action="store_true", help="show the 80 MiB chunk plan without contacting the server")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    file_path = args.file.expanduser()
    if not file_path.is_file():
        print(f"error: file not found: {file_path}", file=sys.stderr)
        return 2
    size = file_path.stat().st_size
    try:
        planned_chunks = _chunk_count(size, DEFAULT_CHUNK_SIZE)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    content_type = args.content_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "file": str(file_path),
                    "size": size,
                    "plannedChunkSize": DEFAULT_CHUNK_SIZE,
                    "plannedTotalChunks": planned_chunks,
                    "contentType": content_type,
                },
                ensure_ascii=False,
            )
        )
        return 0

    client = SttUploadClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=max(1.0, args.timeout),
        retries=max(0, args.retries),
    )
    settings = {
        "model": args.model,
        "engine": args.engine,
        "language": args.language,
        "useGpu": args.use_gpu,
        "prompt": args.prompt,
        "diarize": args.diarize,
        "speakers": args.speakers,
    }
    try:
        job = client.upload_file(file_path=file_path, content_type=content_type, settings=settings)
        if args.no_poll:
            print(json.dumps(job, ensure_ascii=False))
            return 0
        job_id = job.get("jobId")
        if not job_id:
            print(json.dumps(job, ensure_ascii=False))
            return 0
        result = client.poll_job(
            job_id=str(job_id),
            interval=max(0.1, args.poll_interval),
            timeout=max(1.0, args.poll_timeout),
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ApiError, ConnectionError, OSError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
