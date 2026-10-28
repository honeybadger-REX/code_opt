"""In-memory job store used by the FastAPI backend."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def create(self, repo_path: str, options: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "repo_path": repo_path,
            "status": "queued",
            "stage": "QUEUED",
            "created_at": _now(),
            "updated_at": _now(),
            "started_at": None,
            "finished_at": None,
            "options": options,
            "report": None,
            "error": None,
            "logs": [],
            "progress": {"iterations_completed": 0, "max_iterations": options.get("max_iterations", 0)},
        }
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
        return dict(job)

    def start(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = "running"
                job["stage"] = "OBSERVE"
                job["started_at"] = _now()
                job["updated_at"] = _now()

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(fields)
            job["updated_at"] = _now()

    def log(self, job_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["logs"].append(entry)
            if len(job["logs"]) > 300:
                del job["logs"][: len(job["logs"]) - 300]
            job["stage"] = entry.get("stage", job["stage"])
            job["updated_at"] = _now()

    def finish(self, job_id: str, report: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["report"] = report
            job["status"] = report.get("status", "completed")
            job["stage"] = report.get("stage", "DONE")
            job["error"] = report.get("error")
            job["finished_at"] = report.get("finished_at") or _now()
            job["updated_at"] = _now()
            job["progress"] = {
                "iterations_completed": len(report.get("iterations", [])),
                "max_iterations": job["options"].get("max_iterations", 0),
                "kept": report.get("kept", 0),
                "reverted": report.get("reverted", 0),
                "rejected": report.get("rejected", 0),
            }
            job["logs"] = report.get("logs", job["logs"])

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._order:
                return None
            return dict(self._jobs[self._order[-1]])

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "job_id": job["job_id"],
                    "repo_path": job["repo_path"],
                    "status": job["status"],
                    "stage": job["stage"],
                    "created_at": job["created_at"],
                }
                for job in (self._jobs[job_id] for job_id in reversed(self._order))
            ]
