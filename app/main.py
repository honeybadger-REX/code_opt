"""FastAPI backend for CodeOptimizer AI."""

from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import Agent
from .analyzer import analyze_repository
from .jobs import JobStore
from .tools import ToolRegistry

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class AnalyzeRequest(BaseModel):
    repo_path: str = Field(..., description="Path to the Python repository to analyze")
    max_files: int = Field(1000, ge=1, le=10000)


class OptimizeRequest(BaseModel):
    repo_path: str = Field(..., description="Path to the Python repository to optimize")
    max_iterations: int = Field(2, ge=1, le=10)
    apply: bool = Field(False, description="Copy verified changes back to the original repository")
    min_improvement_pct: float = Field(5.0, ge=0.0, le=100.0)
    benchmark_size: int = Field(3000, ge=10, le=100000)
    benchmark_repeats: int = Field(3, ge=1, le=20)


def _validate_repo(repo_path: str) -> Path:
    path = Path(repo_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Repository path is not a directory: {path}")
    return path


def create_app(llm: Any | None = None) -> FastAPI:
    app = FastAPI(
        title="CodeOptimizer AI",
        version="0.1.0",
        description="Agentic Python optimization: propose, modify, test, benchmark, verify.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = JobStore()
    app.state.store = store

    def run_job(job_id: str, repo_path: str, options: dict[str, Any]) -> None:
        store.start(job_id)

        def on_update(stage: str, message: str, data: dict[str, Any]) -> None:
            store.log(
                job_id,
                {
                    "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "stage": stage,
                    "message": message,
                    "data": data,
                },
            )

        try:
            agent = Agent(
                llm=llm,
                on_update=on_update,
                min_improvement_pct=options["min_improvement_pct"],
                benchmark_size=options["benchmark_size"],
                benchmark_repeats=options["benchmark_repeats"],
            )
            report = agent.run(
                repo_path,
                max_iterations=options["max_iterations"],
                apply=options["apply"],
                job_id=job_id,
            )
            store.finish(job_id, report)
        except Exception as exc:  # pragma: no cover - defensive
            message = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            current = store.get(job_id) or {}
            stage = current.get("stage") or "ERROR"
            try:
                on_update("ERROR", message, {"traceback": tb})
            except Exception:  # pragma: no cover - logging must never hide the failure
                pass
            # The job runner itself crashed (infrastructure failure outside the
            # agent's own guarded report path).  The agent reports its internal
            # failures as "aborted"; a crash that escapes the runner is a
            # "failed" job, which the API and UI already distinguish.
            store.update(job_id, status="failed", stage=stage, error=message)
            current = store.get(job_id) or {}
            store.finish(
                job_id,
                {
                    "job_id": job_id,
                    "repo_path": repo_path,
                    "status": "failed",
                    "stage": stage,
                    "outcome": "failed",
                    "started_at": current.get("started_at"),
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "duration_seconds": None,
                    "analysis": None,
                    "repository": None,
                    "validation": None,
                    "candidate_assessment": None,
                    "candidates_found": 0,
                    "candidates_attempted": 0,
                    "baseline_tests": None,
                    "baseline_diagnostic": None,
                    "iterations": [],
                    "kept": 0,
                    "reverted": 0,
                    "skipped": 0,
                    "rejected": 0,
                    "outcome_reason": None,
                    "final_diff": "",
                    "patch_path": None,
                    "applied_to_original": False,
                    "apply_result": None,
                    "options": options,
                    "error": message,
                    "failure": {
                        "stage": stage,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "reason": str(exc),
                        "traceback": tb,
                        "candidate": None,
                    },
                    "traceback": tb,
                    "limitations": [
                        "The agent crashed before completing; no code was applied to the original repository."
                    ],
                    "logs": current.get("logs", []),
                },
            )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="frontend/index.html not found")
        return FileResponse(index_path)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "codeoptimizer-ai"}

    @app.get("/tools")
    def tools() -> dict[str, Any]:
        return {"tools": ToolRegistry.specs()}

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        path = _validate_repo(request.repo_path)
        analysis = analyze_repository(path, max_files=request.max_files)
        return {"status": "ok", "analysis": analysis.to_dict()}

    @app.post("/optimize")
    def optimize(request: OptimizeRequest) -> dict[str, Any]:
        path = _validate_repo(request.repo_path)
        options = {
            "max_iterations": request.max_iterations,
            "apply": request.apply,
            "min_improvement_pct": request.min_improvement_pct,
            "benchmark_size": request.benchmark_size,
            "benchmark_repeats": request.benchmark_repeats,
        }
        job = store.create(str(path), options)
        thread = threading.Thread(
            target=run_job,
            args=(job["job_id"], str(path), options),
            daemon=True,
            name=f"codeoptimizer-{job['job_id']}",
        )
        thread.start()
        return {
            "job_id": job["job_id"],
            "status": "queued",
            "status_url": f"/status/{job['job_id']}",
            "report_url": f"/report/{job['job_id']}",
        }

    @app.get("/status")
    def status_latest() -> dict[str, Any]:
        job = store.latest()
        if job is None:
            raise HTTPException(status_code=404, detail="No jobs have been started")
        return _status_payload(job)

    @app.get("/status/{job_id}")
    def status(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        return _status_payload(job)

    @app.get("/report")
    def report_latest() -> Any:
        job = store.latest()
        if job is None:
            raise HTTPException(status_code=404, detail="No jobs have been started")
        return _report_payload(job)

    @app.get("/report/{job_id}")
    def report(job_id: str) -> Any:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        return _report_payload(job)

    @app.get("/jobs")
    def jobs() -> dict[str, Any]:
        return {"jobs": store.list()}

    return app


def _status_payload(job: dict[str, Any]) -> dict[str, Any]:
    report = job.get("report") or {}
    return {
        "job_id": job["job_id"],
        "repo_path": job["repo_path"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job.get("progress", {}),
        "error": job.get("error"),
        "failure": report.get("failure"),
        "repository": report.get("repository"),
        "validation": report.get("validation"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
        "logs": job.get("logs", [])[-25:],
    }


def _report_payload(job: dict[str, Any]) -> Any:
    if job.get("report") is None:
        if job.get("status") in {"failed", "aborted"}:
            return {
                "job_id": job["job_id"],
                "repo_path": job.get("repo_path"),
                "status": job["status"],
                "stage": job.get("stage"),
                "outcome": job["status"],
                "error": job.get("error") or "the job ended without producing a report",
                "failure": job.get("failure"),
                "traceback": job.get("traceback"),
                "repository": None,
                "validation": None,
                "iterations": [],
                "kept": 0,
                "reverted": 0,
                "skipped": 0,
                "rejected": 0,
                "candidates_found": 0,
                "candidates_attempted": 0,
                "final_diff": "",
                "logs": job.get("logs", []),
            }
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job["job_id"],
                "status": job["status"],
                "stage": job["stage"],
                "detail": "report is not ready yet",
            },
        )
    return job["report"]


app = create_app()
