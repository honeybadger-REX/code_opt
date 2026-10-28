# CodeOptimizer AI

An agentic Python optimizer: it finds a performance problem, asks the configured
OpenCode model for a fix, applies the change in an isolated workspace, runs the
repository's tests, benchmarks the original against the new code, and keeps the
change **only when tests pass, outputs match, and the code is measurably faster**.
Otherwise it reverts and reports why.

```
OBSERVE -> ANALYZE -> PLAN -> MODIFY -> TEST -> BENCHMARK -> VERIFY -> KEEP / REVERT
```

This is a working MVP, not a mock: every tool executes, every test run is real,
and every benchmark number in the report comes from a measured run. When a
reliable benchmark cannot be produced, the agent says so and reverts the change.

## Quick start

```powershell
# 1. Install dependencies (Python 3.10+; developed on 3.12)
python -m pip install -r requirements.txt

# 2. Analyze the bundled example (read-only, no model call)
python run.py analyze examples

# 3. Run the full agent on the example with the configured OpenCode model
python run.py demo --max-iterations 2

# 4. Start the web UI + API at http://127.0.0.1:8000
python run.py serve
```

Open `http://127.0.0.1:8000`, enter a repository path, then click **Analyze** or
**Optimize**. The UI polls the job and shows findings, proposals, test results,
before/after benchmarks, the final diff, and the limitations.

### Model configuration

The agent uses the OpenCode CLI that is already configured on the machine, so no
API key is stored or hard-coded:

- If `CODEOPTIMIZER_MODEL` is unset, OpenCode's configured default model is used.
- Set it to pin a model, e.g. `CODEOPTIMIZER_MODEL=opencode/deepseek-v4.1-flash-free`.
- Set `CODEOPTIMIZER_OPENCODE_BIN` if `opencode` is not on `PATH`.
- Set `CODEOPTIMIZER_LLM_TIMEOUT` (default 300 seconds) for slower models.

`.env.example` documents the available variables; the app reads them from the
process environment, so set them in your shell (it does not auto-load `.env`).

If the model is unavailable, times out, or returns invalid code, the agent falls
back to conservative deterministic rewrites for the patterns it knows
(Counter-based duplicate matching, constant membership sets, list-and-join
string building) and records that in the report.

### Repository configuration

An optional `.codeoptimizer.json` at the repository root controls benchmark
settings:

```json
{
  "benchmark": {
    "size": 2500,
    "repeats": 3,
    "min_improvement_pct": 5
  }
}
```

## How verification works

A proposed change is kept only if **all** of these are true:

1. It is a valid replacement of exactly one existing function.
2. The signature, async/sync nature, decorators, and docstring are unchanged.
3. It contains no banned calls (`eval`, `exec`, `open`, ...) or imports
   (`os`, `sys`, `subprocess`, network, filesystem, ...).
4. The repository's own `pytest` suite passes in the isolated workspace.
5. A differential benchmark runs both versions on the same generated inputs and
   the outputs are equal.
6. The median speedup is at least `min_improvement_pct` (default 5%).

If tests fail, outputs differ, or the change is too small, the file is reverted
(`git checkout` when Git is available, otherwise a snapshot restore) and the
reason is recorded. A candidate that cannot be validated at all (invalid
replacement, benchmark unavailable, or an unexpected error while applying it)
is marked `rejected`; the job continues and the exact exception/reason is kept in
the report.

### Decision states

Validation produces evidence; the `DecisionEngine` (`app/decision.py`) is the
single place that turns evidence into an action, so the acceptance rules are
auditable and unit-tested:

| State | Meaning |
| --- | --- |
| `KEEP` | Validation passed **and** the benchmark measured both versions on the same inputs, outputs matched, and the improvement meets the threshold. |
| `REVERT` | The change was applied but the evidence does not support it (failing tests, behavioural mismatch, outputs differ, improvement below threshold). |
| `REJECT` | The candidate itself is unsafe, invalid, or unverifiable (failed safety checks, invalid replacement, missing validation or benchmark evidence). Nothing executable was proven. |
| `SKIP` | No candidate existed to evaluate; nothing was applied. |
| `ABORT` | Infrastructure/system failure prevented the workflow (job-level, never a candidate verdict). |

Every attempt records `action`, `reasons`, `confidence` (`HIGH` for repository
tests, `MEDIUM` for generated behavioural cases, `LOW`/`UNVERIFIED` otherwise),
and an `evidence` list showing which validation-hierarchy levels (repository
tests, framework tests, generated behavioural cases, differential outputs,
edge-case inputs, benchmark measurement, static safety) were actually
evaluated. Levels that were not evaluated are never claimed. The rules are
fail-closed: no combination of missing evidence can produce `KEEP`.
 The pytest runner ignores inherited `PYTEST_ADDOPTS`,
`PYTEST_PLUGINS`, `PYTEST_DEBUG`, and `PYTEST_CURRENT_TEST` so the caller's
session cannot accidentally deselect the verification tests.
A failed model proposal is retried once, then the deterministic fallback is
tried in the same iteration. The original repository is never edited unless
`apply=true` is explicitly requested; then changes are copied back with a
backup and a `.patch` file under `.codeoptimizer/`.

## Project structure

```
code_optimizer/
├── app/
│   ├── __init__.py
│   ├── main.py          FastAPI app, endpoints, background jobs
│   ├── agent.py         OBSERVE..VERIFY loop, candidate attempts, reporting
│   ├── analyzer.py      AST analysis: findings, complexity, duplicates, dead code
│   ├── optimizer.py     safe function replacement, deterministic rewrites
│   ├── validation.py    repo discovery, safety assessment, generated cases,
│   │                    differential behavioural validation
│   ├── decision.py      DecisionEngine: KEEP/REVERT/REJECT/SKIP rules,
│   │                    validation-hierarchy evidence, fail-closed invariants
│   ├── tester.py        pytest/unittest runners with structured results
│   ├── benchmark.py     isolated differential benchmark + input generation
│   ├── git_manager.py   isolated workspace, diffs, commits, revert, apply
│   ├── llm.py           OpenCode CLI adapter and prompt
│   ├── jobs.py          in-memory job store
│   ├── models.py        dataclasses shared by the API and tools
│   └── tools.py         read/write/search/analyze/test/benchmark/git tools
├── frontend/
│   └── index.html       single-page UI (no build step)
├── examples/
│   ├── inefficient.py   deliberately slow demo functions
│   ├── test_inefficient.py  10 behavioral tests
│   └── .codeoptimizer.json
├── tests/               analyzer, optimizer, validation, decision, tester,
│                        benchmark, LLM, tools, agent, git manager, API
├── requirements.txt
├── .env.example
├── run.py               CLI: serve | analyze | optimize | demo
└── README.md
```

## CLI

```powershell
python run.py analyze [repo] [--json]
python run.py optimize [repo] --max-iterations 2 [--apply] [--no-llm] [--json]
python run.py demo [--max-iterations 2]
python run.py serve [--host 127.0.0.1] [--port 8000] [--reload]
```

`--no-llm` runs the deterministic planner only, which is useful for offline demos
and CI.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check |
| `GET` | `/tools` | Tool catalog |
| `POST` | `/analyze` | Analyze a repository (`repo_path`, `max_files`) |
| `POST` | `/optimize` | Start a background optimization job |
| `GET` | `/status` | Latest job status |
| `GET` | `/status/{job_id}` | Status for one job |
| `GET` | `/report` | Latest full report |
| `GET` | `/report/{job_id}` | Full report for one job |
| `GET` | `/jobs` | Recent jobs |

Example:

```powershell
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" `
  -d '{"repo_path": "examples"}'
```

## Example result (actual run)

Command: `python run.py demo --max-iterations 2` on 2026-10-27, with the
configured OpenCode model and the example's 2500-element benchmark inputs.

```
Status     : completed (optimized)
Duration   : 52.32 s
Kept       : 2   Reverted: 0   Skipped: 0

Iteration 1: nested_loops in inefficient.py:16 (find_common)
  Strategy    : counter_multiplicity [llm]
  Tests       : PASSED (10 passed, 0 failed, 0 errors)
  Before      : Runtime: 244.4 ms
  After       : Runtime: 1.5 ms
  Improvement : 99.39%
  Outputs match: True

Iteration 2: string_concat_in_loop in inefficient.py:42 (build_message)
  Strategy    : join_list_parts [llm]
  Tests       : PASSED (10 passed, 0 failed, 0 errors)
  Before      : Runtime: 942 us
  After       : Runtime: 404 us
  Improvement : 57.1%
  Outputs match: True
```

The kept diff:

```diff
@@ def find_common(a: list[int], b: list[int]) -> list[int]:
+    from collections import Counter
+
+    counts: Counter[int] = Counter(b)
+
+    result: list[int] = []
 
     for x in a:
-        for y in b:
-            if x == y:
-                result.append(x)
+        occurrences = counts.get(x, 0)
+        if occurrences:
+            result.extend([x] * occurrences)
 
     return result
@@ def build_message(items: list[str]) -> str:
-    message = ""
-    for item in items:
-        message += item + ","
-    return message
+    return "".join([item + "," for item in items])
```

The `find_common` rewrite preserves the original multiplicity semantics: an
outer element that matches `k` inner elements is emitted `k` times, in the
original order. A naive `set(b)` rewrite would fail the example tests and would
be reverted.

## Test results

Command: `python -m pytest -q --color=no`

```
138 passed in 68.58 s
```

The suite covers:

- AST detection: nested loops, membership in loops, string concatenation,
  repeated computation, dead code, duplicate functions, parse errors, exclusions.
- Optimizer safety: signature/name/decorator/docstring checks, banned calls and
  imports, method indentation, deterministic rewrites and their semantics.
- Validation engine: repository discovery (pytest/unittest/custom/none), safety
  assessment, generated edge-case inputs, differential behavioural validation
  (match and mismatch detection, float tolerance).
- Decision engine: every decision state, fail-closed rules (no KEEP without
  validation + matching benchmark ≥ threshold), confidence levels, and the
  validation-hierarchy evidence records.
- Test runner: passing, failing, no-tests-collected, and unittest discovery
  outcomes (including Python 3.12's exit code 5).
- Benchmark: improvement detection, wrong-output detection, input mutation
  rejection, annotation inference.
- Git safety: isolated workspaces, excludes, snapshot fallback, diff/revert,
  apply-with-backup, cleanup.
- Agent: keep, revert on test failure, revert when slower, LLM retry, fallback,
  abort on failing/missing baseline tests, apply to original, decision metadata.
- API: health, UI, tools, analyze, optimize job lifecycle, report, 404/400 paths,
  background job crash reporting.
- LLM adapter: JSON parsing, CLI invocation, model flag, timeouts, errors.

## Benchmark methodology

- Both versions are loaded into one fresh subprocess from the same inputs.
- Inputs are generated from type annotations (list/set/tuple/dict of int/str/
  float/bool, optional and `X | None`), with a fixed seed and configurable size.
- The order of A/B measurements alternates; the reported number is the median of
  `repeats` runs after a warmup.
- The return values must be equal (same type and value).
- A function that mutates its arguments is rejected because the comparison would
  no longer be meaningful.
- Functions without annotations, variadic signatures, relative imports, or
  external state make the benchmark unavailable; the change is then rejected
  rather than assumed to be an improvement.

## Known limitations

- Python only. No JavaScript/TypeScript/Go analysis yet.
- The automatic benchmark needs type annotations and non-mutating functions;
  otherwise no speedup can be proven and the change is rejected.
- Test files (`test_*.py`, `*_test.py`, `tests/`, `conftest.py`) are never
  modified, and a repository with no tests is refused because correctness cannot
  be verified.
- The agent runs pytest with the current Python environment, not the target
  repository's own virtualenv.
- Docstrings are preserved verbatim, so an optimization can make the docstring's
  complexity claim stale.
- Relative imports and modules that are not importable on their own may be
  impossible to benchmark.
- Exact float output equality is required, which is stricter than approximate
  numerical comparison.
- The API has no authentication and stores jobs in memory (lost on restart).
- The whole repository is copied to a temporary workspace; very large
  repositories will be slow to copy.
- The model can be slow (tens of seconds to a few minutes per call) or offline;
  deterministic fallbacks cover only a few known patterns.
- `apply=true` copies whole files back, not a Git commit or merge request.

## What to build next for production

1. **More verified rewrites**: dict/set lookups for non-literal membership,
   `deque` for `pop(0)`, hoisting loop-invariant work, generator-to-list
   trade-offs, and batch I/O.
2. **Better benchmarks**: per-repository input fixtures, pytest-benchmark or
   `timeit` profiles, statistical significance, memory measurement, and
   automatic scaling checks (O(n) vs O(n²) across sizes).
3. **Stronger isolation**: Docker/container execution, per-repo virtualenv,
   resource limits, and network isolation.
4. **Repository-aware edits**: multi-file refactors, call-site updates, and
   import management instead of single-function replacement.
5. **Review workflow**: Git branch + pull request, signed commits, approval
   gates, and a web diff review before applying.
6. **Persistence and auth**: a database-backed job store, API keys/OAuth, rate
   limits, and per-user workspaces.
7. **Broader language support**: tree-sitter analyzers and language-specific
   test/benchmark adapters.
8. **LLM robustness**: structured output/schema validation, cheaper local
   models, caching, and cost/telemetry reporting.
