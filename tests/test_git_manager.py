from __future__ import annotations

from pathlib import Path

import pytest

from app.git_manager import (
    GitError,
    Workspace,
    apply_to_original,
    create_workspace,
    git_available,
    import_verified_file,
)


def test_workspace_isolated_from_source(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    (source / "mod.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    workspace = create_workspace(source, base_dir=tmp_path / "workspaces")
    try:
        assert workspace.root != source
        assert workspace.root.is_dir()
        assert (workspace.root / "mod.py").is_file()
        assert workspace.changed_files() == []

        (workspace.root / "mod.py").write_text(
            "def add(a, b):\n    return b + a\n", encoding="utf-8"
        )
        assert workspace.changed_files() == ["mod.py"]
        assert "return b + a" in workspace.diff_file("mod.py")

        workspace.revert_file("mod.py")
        assert workspace.changed_files() == []
        assert "return a + b" in (workspace.root / "mod.py").read_text(encoding="utf-8")
        # The user's repository is never touched by the workspace.
        assert "return a + b" in (source / "mod.py").read_text(encoding="utf-8")
    finally:
        workspace.cleanup()
    assert not workspace.root.exists()


def test_git_excludes_benchmark_artifacts(tmp_path):
    if not git_available():
        pytest.skip("git is not available")
    source = tmp_path / "repo"
    source.mkdir()
    (source / "mod.py").write_text("x = 1\n", encoding="utf-8")

    workspace = create_workspace(source, base_dir=tmp_path / "workspaces")
    try:
        assert workspace.use_git is True
        assert workspace.baseline_sha
        (workspace.root / "_codeopt_baseline_abc.py").write_text("x = 1\n", encoding="utf-8")
        (workspace.root / "__pycache__").mkdir()
        (workspace.root / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"junk")
        assert workspace.changed_files() == []
        status = workspace._git(["status", "--porcelain"], check=False)
        assert status.stdout.strip() == ""
    finally:
        workspace.cleanup()


def test_snapshot_fallback_without_git(tmp_path, monkeypatch):
    monkeypatch.setattr("app.git_manager.git_available", lambda: False)
    source = tmp_path / "repo"
    source.mkdir()
    (source / "mod.py").write_text("x = 1\n", encoding="utf-8")

    workspace = create_workspace(source, base_dir=tmp_path / "workspaces")
    try:
        assert workspace.use_git is False
        assert workspace.baseline_dir is not None
        (workspace.root / "mod.py").write_text("x = 2\n", encoding="utf-8")
        assert workspace.changed_files() == ["mod.py"]
        assert "x = 2" in workspace.final_diff()
        workspace.revert_all()
        assert workspace.changed_files() == []
        assert (workspace.root / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    finally:
        workspace.cleanup()


def test_apply_to_original_creates_backup_and_patch(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    (source / "mod.py").write_text("x = 1\n", encoding="utf-8")

    workspace = create_workspace(source, base_dir=tmp_path / "workspaces")
    try:
        (workspace.root / "mod.py").write_text("x = 2\n", encoding="utf-8")
        result = apply_to_original(workspace, ["mod.py"])
        assert result["files"] == ["mod.py"]
        assert (source / "mod.py").read_text(encoding="utf-8") == "x = 2\n"
        backup_dir = Path(str(result["backup_dir"]))
        assert (backup_dir / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
        patch = Path(str(result["patch_path"]))
        assert "x = 2" in patch.read_text(encoding="utf-8")
    finally:
        workspace.cleanup()


def test_create_workspace_rejects_missing_directory(tmp_path):
    with pytest.raises(GitError):
        create_workspace(tmp_path / "missing")


def test_import_verified_file_success_and_backup(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    (source / "calc.py").write_text("x = 1\n", encoding="utf-8")

    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "calc.py").write_text("x = 2\n", encoding="utf-8")

    result = import_verified_file(source, ws, "calc.py")
    assert result["file"] == "calc.py"
    assert (source / "calc.py").read_text(encoding="utf-8") == "x = 2\n"
    assert Path(result["backup_path"]).read_text(encoding="utf-8") == "x = 1\n"
    assert "x = 2" in Path(result["patch_path"]).read_text(encoding="utf-8")


def test_import_verified_file_blocks_path_traversal(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    ws = tmp_path / "workspace"
    ws.mkdir()

    with pytest.raises(GitError, match="escapes"):
        import_verified_file(source, ws, "../escape.py")
