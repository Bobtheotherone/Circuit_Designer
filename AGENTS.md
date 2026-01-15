# AGENTS.md — CODEX Operating Manual for FIDP (Fractal-Order Impedance Discovery Platform)

This repository is being built step-by-step from a design document. **Steps 5–14** are the “main steps” that require actual code. Each of those steps will be given to you as an individual prompt with detailed requirements.

Your job as CODEX is to implement the requested step **correctly, reproducibly, and with targeted tests**. You must behave in a way that is compatible with *any* of the individual step prompts (5–14) we send.

---

## 0) Prime Directive

**Ship incremental, working code with targeted pytests for every step.** At the end of each step you must:

1. Add/modify **targeted pytest tests** under `tests/` that directly validate the new or changed behavior.
2. Run **all tests that are plausibly affected by the latest modifications**, show they pass, and include the command(s) + output summary.
3. Provide a short recap and explicitly state whether we are **greenlit** to proceed to the next prompt.

You do **not** need to run the *entire* test suite on every step, but you **must** run everything that is likely impacted by what you changed (see §6 Test Selection Rubric).

---

## 1) Repository North Star

FIDP is a “discovery factory” for fractal/self-similar impedance networks. We are building:

- circuit generation (DSL/grammar → circuit graphs/netlists),
- multi-fidelity evaluators (fast/analytic → sparse MNA → MOR → SPICE),
- model extraction (fractional fit + rational/vector fitting + passivity checks),
- search/optimization (multi-objective BO, evolution, active learning),
- reproducibility plumbing (pipelines + experiment tracking),
- validation gates (baseline replication + passivity + cross-tool agreement).

When implementing a step, keep all code **modular** and **testable**.

---

## 2) Operating Rules (non-negotiable)

### 2.1 Work style
- Prefer **small, focused diffs** over sweeping refactors.
- If you must refactor to implement a step cleanly, keep it minimal and include tests proving behavior didn’t regress.
- Implement with “correctness first” and add guardrails (input validation, clear errors).

### 2.2 Determinism
- All randomized operations must accept a seed and be deterministic when seeded.
- Tests must be deterministic and stable.

### 2.3 No surprises
- Do not introduce heavyweight dependencies without being asked.
- If a new dependency is required, add it explicitly (e.g., `pyproject.toml` / `requirements.txt`) and justify it in the recap.

### 2.4 Security / safety
- Do not fetch from the internet.
- Do not execute untrusted code.
- Do not write anything that attempts to exploit the system.

---

## 3) Project Conventions

### 3.1 Language & structure
- Python-only unless explicitly instructed otherwise.
- Keep library code under `src/fidp/` (or the existing package root).
- Tests under `tests/`.

### 3.2 Style expectations
- Use clear names, docstrings for public functions/classes, type hints where practical.
- Favor `dataclasses` for structured records.
- Keep functions small and composable.
- Avoid overengineering internal abstractions unless the step explicitly needs it.

### 3.3 Public API stability
- When a module’s interface becomes “public” (used by multiple other modules/tests), avoid breaking changes unless required by the step prompt.
- If you must break an interface, update callers + tests in the same step.

---

## 4) Step Execution Protocol (do this every step)

For each step prompt (5–14), **execute all phases (A–D) sequentially in a single continuous output. Do NOT stop after Phase A to wait for approval.**

### Phase A — Understand & plan
1. Restate the step’s acceptance criteria in your own words.
2. Identify which modules/files are likely impacted.
3. Propose an implementation plan (2–6 bullets).
4. Identify the **test plan** (which tests you’ll add/update; which existing tests you’ll run).

### Phase B — Implement
1. Implement the requested functionality with minimal necessary changes.
2. Add/modify **targeted** tests that verify:
&nbsp;  - correctness on representative cases,
&nbsp;  - error handling for invalid inputs,
&nbsp;  - any invariants (e.g., passivity checks, monotonic metrics, shape constraints).
3. Update docs only if the step requires it or if changes would otherwise be unclear.

### Phase C — Verify with targeted tests
1. Select tests to run using §6.
2. Run the selected tests with pytest.
3. Ensure full pass. If failures occur:
&nbsp;  - fix the code or tests,
&nbsp;  - rerun the targeted test set,
&nbsp;  - do not proceed until green.

### Phase D — Report
Provide a final response that includes:

- **Recap**: what you changed and why (short, concrete).
- **Files changed**: a concise list (paths).
- **Tests added/updated**: list test files + what they cover.
- **Commands run**: exact `pytest ...` commands.
- **Results**: “PASS” summary (and any key logs if relevant).
- **Greenlight**: Explicitly state `GREENLIT: YES` or `GREENLIT: NO`, and if NO, what must be done before the next prompt.

---

## 5) Testing Requirements

### 5.1 Where tests go
- All tests must live under `tests/`.
- Prefer one test module per feature area:
&nbsp; - `tests/test_mna_*.py`
&nbsp; - `tests/test_mor_*.py`
&nbsp; - `tests/test_vector_fit_*.py`
&nbsp; - `tests/test_fractional_fit_*.py`
&nbsp; - `tests/test_dsl_*.py`
&nbsp; - `tests/test_search_*.py`
&nbsp; - etc.

### 5.2 What tests must assert
Tests should verify:
- core numerical correctness within tolerances,
- shape + type expectations,
- edge cases (empty graphs, singular matrices, invalid component values),
- determinism (seeded runs),
- regression for previously fixed issues.

### 5.3 Keep tests fast
- Unit tests should generally run in seconds.
- If a test needs heavier computation, mark it (e.g., `@pytest.mark.slow`) and only run it when the step requires it.

---

## 6) Test Selection Rubric (targeted, but not forgetful)

You must run **all tests likely affected** by the latest modifications.

### 6.1 Always run
- Tests in files you modified.
- Tests you added.

### 6.2 Run by dependency (practical heuristic)
For each modified source module `src/fidp/<area>/X.py`:
1. Run any tests in `tests/` whose filename matches the area or feature:
&nbsp;  - e.g., changing `mna/assemble.py` → run `tests/test_mna_*`
2. Search tests for imports/references to the module and run those test files too.
&nbsp;  - Example technique (use whatever search tool is available):
&nbsp;    - `rg "fidp\.mna\.assemble|from fidp\.mna\.assemble" tests -n`
3. If you changed a shared utility used broadly (e.g., `fidp/utils.py`, core schemas, DesignRecord):
&nbsp;  - run the broader set in that domain (often all `tests/test_*.py` in core areas).

### 6.3 Maintain a lightweight coverage memory
To avoid forgetting accumulated tests over time, keep a small file:

- `tests/TEST_INDEX.md`

Update it whenever you add a new test module. Each entry should say:
- what area it covers,
- which modules it targets.

This index is used to decide what’s “related” when selecting tests later.

**Example entry format:**
- `tests/test_mna_port_impedance.py` — covers `fidp/evaluators/mna/port_impedance.py` + edge cases for singular matrices.

### 6.4 Do not run the entire suite unless needed
Avoid indiscriminate full-suite runs every step. Only run full suite when:
- the step prompt explicitly asks,
- or you changed a foundational core that touches nearly everything (schemas, shared math utilities, package initialization).

---

## 7) Numerical Tolerance Standards (defaults)

Unless a step prompt specifies otherwise:
- Use relative/absolute tolerances appropriate to the calculation:
&nbsp; - `pytest.approx` with `rel=1e-6` to `1e-3` depending on conditioning.
- For fits (fractional/rational), validate:
&nbsp; - error curves are within the step’s defined thresholds,
&nbsp; - monotonic or bounded properties when expected.

If you introduce new metrics, define their tolerances clearly in tests.

---

## 8) Documentation & Artifacts (when relevant)

Only add documentation if:
- the step prompt requires it,
- or the code introduces a new public-facing interface that would be confusing without minimal docs.

When adding artifacts generation, keep outputs under:
- `artifacts/` (gitignored unless instructed)
- or a step-defined path.

---

## 9) Performance Guidance (workstation constraints)

We are on a single workstation (GPU + CPU). Keep the pipeline:
- batch-friendly (vectorized ops where possible),
- cache-aware (avoid recomputing expensive sweeps),
- testable without long runtime.

Avoid any design that requires “billions of explicit components.” Use hierarchical/recursive representations when needed.

---

## 10) Definition of Done (for each step)

A step is **DONE** only when:
- Implementation meets the step prompt’s acceptance criteria.
- Targeted tests are added/updated in `tests/`.
- Relevant tests (per §6) have been run and pass.
- You provide a recap + `GREENLIT: YES`.

If not done, explicitly say `GREENLIT: NO` and list the blocking issues.

---

## 11) Response Template (use at end of each step)

Use this exact structure:

1. **Recap**
2. **Files changed**
3. **Tests added/updated**
4. **Commands run**
5. **Results**
6. **GREENLIT: YES/NO** (with reason)

---

## 12) Notes for Steps 5–14

The step prompts will reference sections from the design doc (evaluator stack, model extraction, AI search, novelty scoring, reproducibility, parallelism, validation gates, etc.). This AGENTS.md applies uniformly:

- Implement what the step asks.
- Add targeted tests for what you changed.
- Run all relevant tests (not necessarily all tests).
- Report and greenlight decision.

That’s it.