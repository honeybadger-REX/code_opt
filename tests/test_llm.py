from __future__ import annotations

import json
import subprocess

import pytest

from app.llm import (
    LLMError,
    OpenCodeLLM,
    build_optimization_prompt,
    parse_json_object,
)


def completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_parse_json_object_with_fences():
    text = 'Here is the result:\n```json\n{"strategy": "set", "risk": "low"}\n```\nDone.'
    assert parse_json_object(text) == {"strategy": "set", "risk": "low"}


def test_parse_json_object_with_prose():
    text = 'Sure. {"finding_id": "abc", "optimized_function": "def f():\\n    return 1"}'
    assert parse_json_object(text)["finding_id"] == "abc"


def test_parse_json_object_rejects_non_json():
    with pytest.raises(LLMError):
        parse_json_object("I cannot help with that.")


def test_complete_parses_json_lines():
    captured: dict = {}

    def runner(cmd, cwd, timeout, env):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        payload = {"type": "text", "part": {"type": "text", "text": '{"ok": true}'}}
        return completed(json.dumps(payload) + "\n")

    llm = OpenCodeLLM(binary="fake-opencode", runner=runner)
    assert llm.available() is True
    assert llm.complete("hello") == '{"ok": true}'
    assert "--format" in captured["cmd"]
    assert "json" in captured["cmd"]
    assert captured["cmd"][-1] == "hello"


def test_complete_passes_model_when_configured():
    captured: dict = {}

    def runner(cmd, cwd, timeout, env):
        captured["cmd"] = list(cmd)
        return completed(json.dumps({"type": "text", "part": {"type": "text", "text": "ok"}}) + "\n")

    llm = OpenCodeLLM(binary="fake-opencode", model="provider/model", runner=runner)
    llm.complete("hi")
    assert "--model" in captured["cmd"]
    assert "provider/model" in captured["cmd"]


def test_complete_raises_on_nonzero_exit():
    def runner(cmd, cwd, timeout, env):
        return completed("", returncode=2, stderr="boom")

    llm = OpenCodeLLM(binary="fake-opencode", runner=runner)
    with pytest.raises(LLMError):
        llm.complete("hi")


def test_complete_raises_on_timeout():
    def runner(cmd, cwd, timeout, env):
        raise subprocess.TimeoutExpired(cmd, timeout)

    llm = OpenCodeLLM(binary="fake-opencode", runner=runner)
    with pytest.raises(LLMError):
        llm.complete("hi")


def test_unavailable_without_binary():
    llm = OpenCodeLLM(binary="fake-opencode")
    llm.binary = None
    assert llm.available() is False
    with pytest.raises(LLMError):
        llm.complete("hi")


def test_prompt_contains_constraints():
    prompt = build_optimization_prompt(
        {
            "id": "nested-1",
            "kind": "nested_loops",
            "severity": "high",
            "file": "mod.py",
            "function": "find_common",
            "line": 3,
            "message": "nested loops",
            "suggestion": "use a set",
            "details": {"pattern": "duplicate_match"},
        },
        "def find_common(a, b):\n    return []\n",
        ["from collections import Counter"],
    )
    assert "find_common" in prompt
    assert "Preserve observable behaviour EXACTLY" in prompt
    assert "Return ONLY a JSON object" in prompt
    assert "collections" in prompt
