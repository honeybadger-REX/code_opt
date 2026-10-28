"""Structured data models used across CodeOptimizer AI.

The models are intentionally small dataclasses so that they can be serialized
with ``dataclasses.asdict`` and returned directly as JSON by the API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Finding:
    """A single detected optimization opportunity or code smell."""

    id: str
    kind: str
    file: str
    line: int
    function: str | None
    severity: str
    message: str
    suggestion: str
    confidence: float
    auto_optimizable: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionInfo:
    name: str
    qualname: str
    file: str
    lineno: int
    end_lineno: int
    args: list[str]
    returns: str | None
    decorators: list[str]
    is_async: bool
    loops: int
    max_loop_depth: int
    branches: int
    calls: int
    lines: int
    complexity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassInfo:
    name: str
    file: str
    lineno: int
    end_lineno: int
    bases: list[str]
    methods: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImportInfo:
    file: str
    lineno: int
    module: str
    names: list[str]
    level: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileAnalysis:
    file: str
    lines: int
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    imports: list[ImportInfo]
    findings: list[Finding]
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Analysis:
    repo_path: str
    files_analyzed: int
    total_lines: int
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    imports: list[ImportInfo]
    findings: list[Finding]
    metrics: dict[str, Any]
    parse_errors: list[dict[str, Any]]
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    # Prevent pytest from trying to collect this dataclass when it is imported
    # into a test module.
    __test__ = False

    ran: bool
    passed: bool
    passed_count: int
    failed_count: int
    error_count: int
    skipped_count: int
    returncode: int | None
    duration_seconds: float
    output: str
    timed_out: bool = False
    reason: str | None = None
    runner: str = "pytest"
    command: list[str] = field(default_factory=list)
    cwd: str | None = None
    python_executable: str | None = None
    sanitized_env: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    # Prevent pytest from trying to collect this dataclass when imported.
    __test__ = False

    strategy: str
    confidence: str
    ran: bool
    passed: bool
    cases_total: int = 0
    cases_matched: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    duration_seconds: float = 0.0
    input_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    available: bool
    function: str
    before_seconds: float | None = None
    after_seconds: float | None = None
    improvement_pct: float | None = None
    before_times: list[float] = field(default_factory=list)
    after_times: list[float] = field(default_factory=list)
    repeats: int = 0
    outputs_match: bool | None = None
    input_description: str = ""
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Proposal:
    finding_id: str
    file: str
    function: str
    strategy: str
    rationale: str
    risk: str
    source: str
    planner: str
    expected_gain: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
