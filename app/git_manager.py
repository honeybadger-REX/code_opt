"""Isolated workspaces, diffs, commits, and reverts.

The user's repository is never modified in place by the agent.  A full copy is
created in a temporary directory, initialised as a Git repository (when Git is
available), and all edits happen there.  Applying changes back to the original
repository is an explicit opt-in and always creates a backup first.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .analyzer import EXCLUDED_DIRS


class GitError(Exception):
    """Raised when a Git operation fails."""


def git_available() -> bool:
    try:
        completed = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_AUTHOR_NAME"] = "CodeOptimizer AI"
    env["GIT_AUTHOR_EMAIL"] = "codeoptimizer@local"
    env["GIT_COMMITTER_NAME"] = "CodeOptimizer AI"
    env["GIT_COMMITTER_EMAIL"] = "codeoptimizer@local"
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if check and completed.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    return completed


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in EXCLUDED_DIRS or name.endswith(".pyc") or name.endswith(".pyo"):
            ignored.add(name)
    return ignored


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_tree(path: Path) -> None:
    """Remove a directory tree, handling read-only files (Windows Git objects)."""

    def _on_error(func, target, _exc_info):  # type: ignore[no-untyped-def]
        try:
            os.chmod(target, 0o700)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_on_error)


@dataclass
class Workspace:
    source: Path
    root: Path
    use_git: bool
    baseline_sha: str | None = None
    baseline_dir: Path | None = None

    def _git(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        return _git(args, self.root, check=check)

    # -- diffing ---------------------------------------------------------
    def diff_file(self, relative_path: str) -> str:
        relative_path = relative_path.replace("\\", "/")
        if self.use_git:
            completed = self._git(["diff", "--no-color", "--", relative_path], check=False)
            return completed.stdout
        if self.baseline_dir is None:
            return ""
        return self._snapshot_diff(only=relative_path)

    def final_diff(self) -> str:
        if self.use_git and self.baseline_sha:
            completed = self._git(["diff", "--no-color", self.baseline_sha, "--", "."], check=False)
            return completed.stdout
        return self._snapshot_diff()

    def _snapshot_diff(self, only: str | None = None) -> str:
        if self.baseline_dir is None:
            return ""
        chunks: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(self.root)).replace("\\", "/")
            if only and relative != only:
                continue
            original = self.baseline_dir / relative
            if not original.exists():
                continue
            old = original.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            new = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            if old != new:
                chunks.extend(
                    difflib.unified_diff(
                        old,
                        new,
                        fromfile=f"a/{relative}",
                        tofile=f"b/{relative}",
                    )
                )
        return "".join(chunks)

    # -- state -----------------------------------------------------------
    def changed_files(self) -> list[str]:
        if self.use_git and self.baseline_sha:
            completed = self._git(
                ["diff", "--name-only", self.baseline_sha, "--", "."], check=False
            )
            files = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            status = self._git(["status", "--porcelain"], check=False)
            for line in status.stdout.splitlines():
                entry = line[3:].strip()
                if entry and entry not in files:
                    files.append(entry)
            return sorted(set(files))
        if self.baseline_dir is None:
            return []
        changed = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = str(path.relative_to(self.root)).replace("\\", "/")
            original = self.baseline_dir / relative
            if not original.exists() or _hash_file(original) != _hash_file(path):
                changed.append(relative)
        return sorted(changed)

    def revert_file(self, relative_path: str) -> None:
        relative_path = relative_path.replace("\\", "/")
        if self.use_git:
            self._git(["checkout", "--", relative_path])
            return
        if self.baseline_dir is None:
            raise GitError("no baseline available for revert")
        original = self.baseline_dir / relative_path
        target = self.root / relative_path
        if original.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)

    def revert_all(self) -> None:
        if self.use_git:
            self._git(["checkout", "--", "."], check=False)
            self._git(["clean", "-fd"], check=False)
            return
        if self.baseline_dir is None:
            raise GitError("no baseline available for revert")
        for path in self.root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(self.root)
                original = self.baseline_dir / relative
                if not original.exists():
                    path.unlink()
                elif _hash_file(original) != _hash_file(path):
                    shutil.copy2(original, path)

    def commit(self, message: str) -> str | None:
        if not self.use_git:
            return None
        self._git(["add", "-A"])
        status = self._git(["status", "--porcelain"], check=False)
        if not status.stdout.strip():
            head = self._git(["rev-parse", "HEAD"], check=False)
            return head.stdout.strip() or None
        self._git(["commit", "-m", message, "--no-gpg-sign"])
        head = self._git(["rev-parse", "HEAD"], check=False)
        return head.stdout.strip() or None

    def save_patch(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.final_diff(), encoding="utf-8")
        return path

    def cleanup(self) -> None:
        _remove_tree(self.root)
        if self.baseline_dir is not None:
            _remove_tree(self.baseline_dir)


def create_workspace(source: Path, base_dir: Path | None = None) -> Workspace:
    """Copy ``source`` into a temporary isolated workspace."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise GitError(f"not a directory: {source}")

    if base_dir is None:
        base_dir = Path(
            os.environ.get(
                "CODEOPTIMIZER_WORKSPACE_DIR",
                str(Path(tempfile.gettempdir()) / "codeoptimizer" / "workspaces"),
            )
        )
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in source.name) or "repo"
    root = base_dir / f"{safe_name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    shutil.copytree(source, root, ignore=_ignore)

    if not git_available():
        baseline_dir = root.parent / f"{root.name}-baseline"
        shutil.copytree(root, baseline_dir, ignore=_ignore)
        return Workspace(source=source, root=root, use_git=False, baseline_dir=baseline_dir)

    try:
        _git(["init", "-q"], root)
        _git(["config", "user.name", "CodeOptimizer AI"], root)
        _git(["config", "user.email", "codeoptimizer@local"], root)
        # Keep temporary benchmark artifacts and caches out of commits and diffs.
        exclude_path = root / ".git" / "info" / "exclude"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        exclude_path.write_text(
            "\n".join(
                [
                    "_codeopt_baseline_*",
                    "_codeopt_optimized_*",
                    "__pycache__/",
                    ".pytest_cache/",
                    "*.pyc",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        _git(["add", "-A"], root)
        _git(["commit", "-m", "codeoptimizer: baseline", "--no-gpg-sign"], root)
        sha = _git(["rev-parse", "HEAD"], root).stdout.strip()
        return Workspace(source=source, root=root, use_git=True, baseline_sha=sha)
    except GitError:
        # Fall back to a plain directory snapshot rather than failing the run.
        baseline_dir = root.parent / f"{root.name}-baseline"
        if baseline_dir.exists():
            _remove_tree(baseline_dir)
        shutil.copytree(root, baseline_dir, ignore=_ignore)
        return Workspace(source=source, root=root, use_git=False, baseline_dir=baseline_dir)


def apply_to_original(workspace: Workspace, files: list[str]) -> dict[str, object]:
    """Copy verified changes back to the source repository, with backups."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    state_dir = workspace.source / ".codeoptimizer"
    backup_dir = state_dir / "backups" / stamp
    patch_path = state_dir / "patches" / f"{stamp}.patch"

    copied: list[str] = []
    for relative in files:
        relative = relative.replace("\\", "/")
        source_file = workspace.root / relative
        target_file = workspace.source / relative
        if not source_file.exists():
            continue
        if target_file.exists():
            backup = backup_dir / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_file, backup)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        copied.append(relative)

    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(workspace.final_diff(), encoding="utf-8")

    return {
        "files": copied,
        "backup_dir": str(backup_dir) if copied else None,
        "patch_path": str(patch_path),
    }


def import_verified_file(
    source_root: Path,
    workspace_root: Path,
    relative_path: str,
) -> dict[str, Any]:
    """Safely copy a single verified file from workspace to source repo with backup."""
    source_root = Path(source_root).resolve()
    workspace_root = Path(workspace_root).resolve()

    norm_rel = Path(relative_path).as_posix().lstrip("/")
    source_file = (source_root / norm_rel).resolve()
    workspace_file = (workspace_root / norm_rel).resolve()

    if source_root not in source_file.parents and source_file != source_root:
        raise GitError(f"Target file escapes source repository: {relative_path}")
    if workspace_root not in workspace_file.parents and workspace_file != workspace_root:
        raise GitError(f"Workspace file escapes workspace directory: {relative_path}")

    if not workspace_file.is_file():
        raise GitError(f"File not found in isolated workspace: {relative_path}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    state_dir = source_root / ".codeoptimizer"
    backup_dir = state_dir / "backups" / stamp
    patch_dir = state_dir / "patches"

    backup_file = None
    old_lines: list[str] = []
    if source_file.exists():
        backup_file = backup_dir / norm_rel
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, backup_file)
        old_lines = backup_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    source_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace_file, source_file)
    new_lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    diff = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{norm_rel}",
            tofile=f"b/{norm_rel}",
        )
    )

    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in norm_rel)
    patch_path = patch_dir / f"{stamp}_{safe_name}.patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(diff, encoding="utf-8")

    return {
        "file": norm_rel,
        "backup_dir": str(backup_dir) if backup_file else None,
        "backup_path": str(backup_file) if backup_file else None,
        "patch_path": str(patch_path),
    }
