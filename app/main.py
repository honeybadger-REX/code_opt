"""FastAPI backend for CodeOptimizer AI."""

from __future__ import annotations

import difflib
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
from .git_manager import GitError, import_verified_file
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

    @app.get("/jobs/{job_id}/files")
    def job_files(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        if job.get("status") not in {"completed", "failed", "aborted"}:
            raise HTTPException(status_code=400, detail="Job is not completed yet")
        report = job.get("report") or {}
        ws_path_str = report.get("workspace")
        if not ws_path_str:
            return {"job_id": job_id, "files": [], "workspace": None}
        ws_root = Path(ws_path_str)
        if not ws_root.is_dir():
            raise HTTPException(status_code=404, detail="Workspace directory not found")

        accepted = _get_accepted_files_for_job(job)
        file_list = []
        for file_rel, info in accepted.items():
            ws_file = ws_root / file_rel
            if ws_file.is_file():
                file_list.append(info)
        return {
            "job_id": job_id,
            "workspace": str(ws_root),
            "repo_path": job.get("repo_path"),
            "files": file_list,
        }

    @app.get("/jobs/{job_id}/files/{file_path:path}/diff")
    def job_file_diff(job_id: str, file_path: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        report = job.get("report") or {}
        ws_path_str = report.get("workspace")
        if not ws_path_str:
            raise HTTPException(status_code=400, detail="Job has no workspace")
        ws_root = Path(ws_path_str)
        if not ws_root.is_dir():
            raise HTTPException(status_code=404, detail="Workspace directory not found")

        repo_root = Path(job.get("repo_path") or "").resolve()
        norm_path = Path(file_path).as_posix().lstrip("/")
        accepted = _get_accepted_files_for_job(job)
        if norm_path not in accepted:
            raise HTTPException(
                status_code=403,
                detail=f"File '{norm_path}' was not modified by an accepted optimization candidate",
            )

        ws_file = _safe_resolve_relative(ws_root, norm_path)
        if not ws_file.is_file():
            raise HTTPException(status_code=404, detail=f"File not found in workspace: {norm_path}")

        orig_file = _safe_resolve_relative(repo_root, norm_path)
        orig_text = orig_file.read_text(encoding="utf-8", errors="replace") if orig_file.is_file() else ""
        opt_text = ws_file.read_text(encoding="utf-8", errors="replace")

        orig_lines = orig_text.splitlines(keepends=True)
        opt_lines = opt_text.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                opt_lines,
                fromfile=f"a/{norm_path}",
                tofile=f"b/{norm_path}",
            )
        )
        diff_text = "".join(diff_lines)
        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

        return {
            "job_id": job_id,
            "file": norm_path,
            "status": "modified",
            "diff": diff_text,
            "original_content": orig_text,
            "optimized_content": opt_text,
            "additions": additions,
            "deletions": deletions,
            "functions": accepted[norm_path].get("functions", []),
        }

    @app.post("/jobs/{job_id}/files/{file_path:path}/import")
    def job_file_import(job_id: str, file_path: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        if job.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Cannot import from an incomplete or failed job")
        report = job.get("report") or {}
        ws_path_str = report.get("workspace")
        if not ws_path_str:
            raise HTTPException(status_code=400, detail="Job has no workspace")
        ws_root = Path(ws_path_str)
        if not ws_root.is_dir():
            raise HTTPException(status_code=404, detail="Workspace directory no longer exists on disk")

        repo_root = Path(job.get("repo_path") or "").resolve()
        if not repo_root.is_dir():
            raise HTTPException(status_code=400, detail="Original repository directory not found")

        norm_path = Path(file_path).as_posix().lstrip("/")
        accepted = _get_accepted_files_for_job(job)
        if norm_path not in accepted:
            raise HTTPException(
                status_code=403,
                detail=f"Only accepted and verified candidates may be imported; '{norm_path}' is not eligible",
            )

        try:
            result = import_verified_file(repo_root, ws_root, norm_path)
            return {
                "status": "ok",
                "message": f"Successfully imported '{norm_path}' into original repository",
                **result,
            }
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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


def _get_accepted_files_for_job(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    report = job.get("report") or {}
    accepted: dict[str, dict[str, Any]] = {}
    for iteration in report.get("iterations", []):
        verification = iteration.get("verification") or {}
        if verification.get("status") == "kept" or verification.get("action") == "KEEP":
            finding = iteration.get("finding") or {}
            file_rel = finding.get("file")
            if not file_rel:
                continue
            norm_rel = Path(file_rel).as_posix().lstrip("/")
            if norm_rel not in accepted:
                accepted[norm_rel] = {
                    "file": norm_rel,
                    "status": "modified",
                    "functions": [],
                    "iterations": [],
                }
            fn = finding.get("function")
            if fn and fn not in accepted[norm_rel]["functions"]:
                accepted[norm_rel]["functions"].append(fn)
            iter_idx = iteration.get("index")
            if iter_idx is not None and iter_idx not in accepted[norm_rel]["iterations"]:
                accepted[norm_rel]["iterations"].append(iter_idx)
    return accepted


def _safe_resolve_relative(base: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise HTTPException(status_code=400, detail="Absolute paths are not allowed")
    resolved = (base / path).resolve()
    base_resolved = base.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise HTTPException(status_code=400, detail="Path traversal detected")
    return resolved


app = create_app()
