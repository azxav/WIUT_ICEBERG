"""Temporary-upload API for the short-video website demo."""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile

from solution import RiskEstimator, detect_events


MAX_BYTES = 200 * 1024 * 1024
MAX_SECONDS = 120.0
# A short upload can spend a large share of its runtime loading two model
# instances. Keep the submission's 3x video budget, with bounded demo startup
# slack so short clips do not fail solely on cold initialization.
MAX_RUNTIME_MULTIPLIER = 3.0
RUNTIME_STARTUP_SLACK_SECONDS = 45.0
MAX_JOBS = 4
JOB_TTL_SECONDS = 15 * 60
ROOT = Path(__file__).resolve().parents[1]
jobs: dict[str, dict] = {}
job_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="traffic-demo")

app = FastAPI(title="ICEBERG Traffic Event Demo", version="1.0.0")


def _video_info(path: Path) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not open this MP4 video.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frames / fps if fps > 0 else 0.0
        if not (duration > 0 and width > 0 and height > 0):
            raise ValueError("Video metadata is incomplete.")
        return duration, fps, width, height
    finally:
        cap.release()


def _resize_for_demo(source: Path, target: Path) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not decode the uploaded video.")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 1280 / max(width, height))
    out_w, out_h = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not prepare the resized demo video.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def _set(job_id: str, **values) -> None:
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(values)
            jobs[job_id]["updated"] = time.time()


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with job_lock:
        expired = [job_id for job_id, job in jobs.items() if job.get("updated", 0) < cutoff]
        for job_id in expired:
            jobs.pop(job_id, None)


def _run_job(job_id: str, original: Path, workdir: Path) -> None:
    resized = workdir / "resized.mp4"
    try:
        _set(job_id, status="processing", message="Resizing video to fit the demo pipeline.")
        _resize_for_demo(original, resized)
        _set(job_id, status="processing", message="Running event detection and causal risk analysis.")
        start = time.perf_counter()
        events = detect_events(str(resized))
        cap = cv2.VideoCapture(str(resized))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("The resized video could not be reopened.")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta = {
            "video_id": original.name, "fps": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "n_frames": n_frames,
        }
        duration = n_frames / fps
        estimator = RiskEstimator()
        estimator.reset(meta)
        risk: list[list[float]] = []
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = index / fps
            risk.append([round(t_sec, 4), round(float(estimator.step(frame, t_sec)), 4)])
            index += 1
        cap.release()
        elapsed = time.perf_counter() - start
        if elapsed > MAX_RUNTIME_MULTIPLIER * duration + RUNTIME_STARTUP_SLACK_SECONDS:
            raise TimeoutError("Analysis exceeded the demo runtime limit (3× duration plus 45 s startup slack).")
        result = {
            "events": events or [],
            "risk": risk,
            "duration": duration,
            "runtime_sec": round(elapsed, 2),
        }
        _set(job_id, status="completed", message="Results are ready.", result=result, error=None)
    except Exception as exc:  # surface actionable errors in the user's session
        _set(job_id, status="failed", message="Analysis failed.", error=str(exc))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile = File(...)) -> dict:
    _cleanup_old_jobs()
    if Path(file.filename or "").suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Upload an MP4 file.")
    workdir = Path(tempfile.mkdtemp(prefix="iceberg-demo-"))
    original = workdir / "upload.mp4"
    total = 0
    try:
        with original.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Maximum upload size is 200 MB.")
                target.write(chunk)
        duration, _, _, _ = _video_info(original)
        if duration > MAX_SECONDS:
            raise HTTPException(status_code=400, detail="Maximum video length is 2 minutes.")
        with job_lock:
            active = sum(job.get("status") in {"queued", "processing"} for job in jobs.values())
            if active >= MAX_JOBS:
                raise HTTPException(status_code=429, detail="Demo queue is full. Try again shortly.")
            job_id = uuid.uuid4().hex
            jobs[job_id] = {
                "job_id": job_id, "status": "queued", "message": "Waiting for analysis.",
                "created": time.time(), "updated": time.time(), "result": None, "error": None,
            }
        executor.submit(_run_job, job_id, original, workdir)
        return {"job_id": job_id, "status": "queued"}
    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    _cleanup_old_jobs()
    with job_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found or session expired.")
        return {key: job.get(key) for key in ("job_id", "status", "message", "result", "error")}
