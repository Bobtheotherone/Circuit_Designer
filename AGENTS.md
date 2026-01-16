# AGENTS.md — CODEX Operating Manual for FIDP (Fractal-Order Impedance Discovery Platform)

This repository is being built step-by-step from a design document. **Steps 5–14** are the “main steps” that require actual code. Each of those steps will be given to you (CODEX) as an individual prompt with detailed requirements.

Your job is to implement the requested step **correctly, reproducibly, GitHub-ready, and with targeted tests**. You must behave in a way that is compatible with *any* of the individual step prompts (5–14) we send.

**IMPORTANT:** This document operates under a strict **"No Scaffolding"** policy (see §13). Every step must result in a fully hardened, production-ready subsystem. The definitions of "DONE" and "GREENLIT" in the early sections are superseded by the stricter requirements in sections 13–22. **If any requirement conflicts, §13–22 wins.**

---

## 0) Prime Directive

**Ship incremental, working code with targeted pytests for every step.**

At the end of each step you must:

1. Add/modify **targeted pytest tests** under `tests/` that directly validate the new or changed behavior.
2. Run **all tests that are plausibly affected by the latest modifications**, show they pass, and include the command(s) + output summary.
3. Provide a short recap and explicitly state whether we are **greenlit** to proceed to the next prompt (subject to the strict criteria in §13.3).
4. Ensure the repo is **GitHub-ready** (no accidental tracked caches/venv/secrets), and provide a **suggested commit message** plus the exact push command:
   - `./ship.sh "message"`

You do **not** need to run the *entire* test suite on every step, but you **must** run everything that is likely impacted by what you changed (see §6 Test Selection Rubric).

---

## 0.1 GitHub-Ready Workflow (mandatory)

### 0.1.1 `ship.sh` must exist and work

This repo uses a single “always push” script: **`./ship.sh "commit message"`**.

**Rules**

- Do **not** change `ship.sh` unless it is missing, broken, or a step explicitly requires updating it.
- Do **not** run `./ship.sh` automatically unless the step prompt explicitly instructs you to commit/push. Your job is to prepare the repo so the user can run it.

**If `ship.sh` is missing or incorrect**, create/overwrite it with exactly this content and make it executable:

```bash
#!/usr/bin/env bash
set -euo pipefail

msg="${*:-}"
if [[ -z "$msg" ]]; then
  echo "Usage: ./ship.sh \"commit message\""
  exit 2
fi

git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

git commit -m "$msg"
branch="$(git rev-parse --abbrev-ref HEAD)"
git push origin "$branch"
echo "Pushed: $branch"
````

Then ensure: `chmod +x ship.sh`.

### 0.1.2 Repository hygiene (never commit junk)

You must prevent committing:

* Python caches: `__pycache__/`, `*.pyc`, `*.pyo`
* Test caches: `.pytest_cache/`
* Virtual envs: `.venv/`, `venv/`, `env/`, `ENV/`
* Local experiment outputs unless explicitly requested: `mlruns/`, `artifacts/` (default: ignored)

**If any of these are accidentally tracked**, remove them from git tracking (without deleting local files) using:

* `git rm -r --cached <path>`

### 0.1.3 `.gitignore` baseline (ensure present)

If `.gitignore` is missing or incomplete, update it (append-only is fine) to include at minimum:

```gitignore
# Python caches
__pycache__/
*.py[cod]
*$py.class

# Test/cache artifacts
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Virtual envs
.venv/
venv/
ENV/
env/

# Experiment/artifact outputs (default)
mlruns/
artifacts/
```

**Do not** ignore project-critical folders like `src/`, `tests/`, or any DVC-tracked directories unless a step prompt explicitly instructs it.

### 0.1.4 Secrets policy (non-negotiable)

* Never add tokens, API keys, or secrets to any tracked file.
* Never embed auth tokens in git remotes.
* If you detect secrets in tracked files, stop and instruct removal before proceeding.

---

## 0.2 Environment & Dependency Management (mandatory)

### 0.2.1 Single source of truth for setup

* `./scripts/bootstrap.sh` is the canonical, required setup path.
* If a step introduces or changes ANY dependency (Python package or external executable), you must update bootstrap so a clean machine can reach a working state by running:

  * `./scripts/bootstrap.sh`

### 0.2.2 Dependency source-of-truth and locking

* The dependency source-of-truth is:

  * `requirements/requirements.in` (direct deps)
  * `requirements/requirements-cpu.lock` + `requirements/requirements-cu128.lock` (resolved locks)
* `pip freeze` is NOT an acceptable lock strategy.
* When deps change, you must:

  1. update `requirements/requirements.in`
  2. update `pyproject.toml` dependency ranges to remain consistent with requirements.in intent
  3. regenerate BOTH lock files using pip-tools:

     * `pip-compile requirements/requirements.in --output-file requirements/requirements-cpu.lock --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple`
     * `pip-compile requirements/requirements.in --output-file requirements/requirements-cu128.lock --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple`

### 0.2.3 External executables are required, and must be handled

* Required external tools must be checked by `python -m fidp.env_check`.
* If an executable can be installed automatically (e.g., ngspice on Ubuntu), bootstrap MUST install it.
* If an executable cannot be installed automatically (e.g., Xyce), env_check must fail with clear, actionable manual install instructions.
* No fake/stub executables are allowed to “make checks pass”.

### 0.2.4 Version consistency (never drift)

Whenever you change the minimum Python version OR introduce syntax requiring a newer Python:

* Update **both**:

  * `pyproject.toml` `requires-python`
  * `scripts/bootstrap.sh` `REQUIRED=...`
* If these drift, the step is NOT DONE.

### 0.2.5 Tests must not depend on host environment

* Unit tests must be deterministic.
* Any tests involving executable presence must monkeypatch `shutil.which` (or equivalent) rather than depending on what’s installed on the machine.

---

## 1) Repository North Star

FIDP is a “discovery factory” for fractal/self-similar impedance networks. We are building:

* circuit generation (DSL/grammar → circuit graphs/netlists),
* multi-fidelity evaluators (fast/analytic → sparse MNA → MOR → SPICE),
* model extraction (fractional fit + rational/vector fitting + passivity checks),
* search/optimization (multi-objective BO, evolution, active learning),
* reproducibility plumbing (pipelines + experiment tracking),
* validation gates (baseline replication + passivity + cross-tool agreement).

When implementing a step, keep all code **modular** and **testable**.

---

## 2) Operating Rules (non-negotiable)

### 2.1 Work style

* **Strict No-Scaffolding Policy:** See §13.1. Zero placeholders, zero `pass`, zero `NotImplementedError` (unless explicitly requested).
* Prefer **small, focused diffs** over sweeping refactors.
* If you must refactor to implement a step cleanly, keep it minimal and include tests proving behavior didn’t regress.
* Implement with “correctness first” and add guardrails (input validation, clear errors).

### 2.2 Determinism

* All randomized operations must accept a seed and be deterministic when seeded.
* Tests must be deterministic and stable.
* Artifacts (listings, candidates) must have deterministic ordering (§14.4).

### 2.3 No surprises

* Do not introduce heavyweight dependencies without being asked.
* If a new dependency is required, you MUST:

  * add it to `requirements/requirements.in` and `pyproject.toml` (consistent intent)
  * regenerate both lock files
  * ensure `./scripts/bootstrap.sh` installs it
  * ensure `python -m fidp.env_check` validates it
* Do not implement optional-import fallbacks for required functionality.

### 2.4 Security / safety

* Do not fetch arbitrary data from the internet.
* Allowed network actions ONLY for repository setup:

  * `pip` installs from official indexes (PyPI and the official PyTorch index URLs used in bootstrap)
  * OS package installs strictly for required executables (e.g., `apt-get install ngspice` in bootstrap)
* Do not curl random tarballs, scrape websites, or clone repos.
* Do not execute untrusted code.
* Do not write anything that attempts to exploit the system.
* Do not touch user SSH config or credentials; only repo-local files.
* Never add tokens, API keys, or secrets to any tracked file.

---

## 3) Project Conventions

### 3.1 Language & structure

* Python-only unless explicitly instructed otherwise.
* Keep library code under `src/fidp/` (or the existing package root).
* Tests under `tests/`.
* Adhere to Type Discipline (§15.2): Use type hints and structured records (Dataclasses/TypedDict; Pydantic allowed only if already a dependency or explicitly required by a step prompt).

### 3.2 Style expectations

* Use clear names, docstrings for public functions/classes, type hints where practical.
* Favor `dataclasses` for structured records.
* Keep functions small and composable.
* Avoid overengineering internal abstractions unless the step explicitly needs it.

### 3.3 Public API stability

* When a module’s interface becomes “public” (used by multiple other modules/tests), avoid breaking changes unless required by the step prompt.
* If you must break an interface, update callers + tests in the same step.

---

## 4) Step Execution Protocol (do this every step)

For each step prompt (5–14), **execute all phases (A–D) sequentially in a single continuous output. Do NOT stop after Phase A to wait for approval.**

### Phase 0 — Pre-flight (GitHub-ready check)

Before coding, do a quick repo hygiene check:

* Ensure `ship.sh` exists and is executable (create/fix only if needed).
* Ensure `.gitignore` contains the baseline ignores (update only if needed).
* Ensure no cache/venv files are newly tracked.
* If any junk is tracked, remove with `git rm -r --cached ...` and include it as part of the step changes.
* If this step touches dependencies or environment:

  * confirm `pyproject.toml` `requires-python` matches `scripts/bootstrap.sh` `REQUIRED`
  * confirm the correct lock files exist under `requirements/`
  * confirm `scripts/bootstrap.sh` is executable

### Phase A — Understand & plan

1. Restate the step’s acceptance criteria in your own words.
2. Identify which modules/files are likely impacted.
3. Propose an implementation plan (2–6 bullets).
4. Identify the **test plan** (which tests you’ll add/update; which existing tests you’ll run).
5. **Hardening Check:** Explicitly list assumptions to ensure the implementation is "Operationally Complete" (§13.2).

### Phase B — Implement

1. Implement the requested functionality with minimal necessary changes, adhering to the **No Scaffolding** rules (§13.1).
2. Add/modify **targeted** tests that verify:

   * correctness on representative cases,
   * error handling for invalid inputs (structured errors, §14.3),
   * any invariants (e.g., passivity checks, monotonic metrics, shape constraints).
3. Update docs only if the step requires it or if changes would otherwise be unclear.

### Phase C — Verify with targeted tests

1. Select tests to run using §6.
2. Run the selected tests with pytest.
3. Ensure full pass. If failures occur:

   * fix the code or tests,
   * rerun the targeted test set,
   * do not proceed until green.

### Phase D — Report (must include commit message + push command)

Provide a final response that includes:

* **Recap**: what you changed and why (short, concrete).
* **Files changed**: a concise list (paths).
* **Tests added/updated**: list test files + what they cover.
* **Commands run**: exact `pytest ...` commands.
* **Results**: “PASS” summary (and any key logs if relevant).
* **Suggested commit message**: a single ideal line for the user to use.
* **Push command**: the exact command: `./ship.sh "..."`.
* **Hardening checklist status**: Confirmation of compliance with §14–§19.
* **Greenlight**: Explicitly state `GREENLIT: YES` or `GREENLIT: NO` (per stricter definition in §13.3).

---

## 5) Testing Requirements

### 5.1 Where tests go

* All tests must live under `tests/`.
* Prefer one test module per feature area:

  * `tests/test_mna_*.py`
  * `tests/test_mor_*.py`
  * `tests/test_vector_fit_*.py`
  * `tests/test_fractional_fit_*.py`
  * `tests/test_dsl_*.py`
  * `tests/test_search_*.py`
  * etc.

### 5.2 What tests must assert

Tests must be meaningful (§18.1). They should verify:

* core numerical correctness within tolerances,
* shape + type expectations,
* edge cases (empty graphs, singular matrices, invalid component values),
* determinism (seeded runs),
* regression for previously fixed issues,
* failure modes (structured errors).

### 5.3 Keep tests fast

* Unit tests should generally run in seconds.
* If a test needs heavier computation, mark it (e.g., `@pytest.mark.slow`) and only run it when the step requires it.

---

## 6) Test Selection Rubric (targeted, but not forgetful)

You must run **all tests likely affected** by the latest modifications.

### 6.1 Always run

* Tests in files you modified.
* Tests you added.

### 6.2 Run by dependency (practical heuristic)

For each modified source module `src/fidp/<area>/X.py`:

1. Run any tests in `tests/` whose filename matches the area or feature:

   * e.g., changing `mna/assemble.py` → run `tests/test_mna_*`

2. Search tests for imports/references to the module and run those test files too.

   * Example technique:

     * `rg "fidp\.mna\.assemble|from fidp\.mna\.assemble" tests -n`

3. If you changed a shared utility used broadly (e.g., `fidp/utils.py`, core schemas, DesignRecord):

   * run the broader set in that domain (often all `tests/test_*.py` in core areas).

### 6.3 Maintain a lightweight coverage memory

To avoid forgetting accumulated tests over time, keep a small file:

* `tests/TEST_INDEX.md`

Update it whenever you add a new test module. Each entry should say:

* what area it covers,
* which modules it targets.

This index is used to decide what’s “related” when selecting tests later.

**Example entry format:**

* `tests/test_mna_port_impedance.py` — covers `fidp/evaluators/mna/port_impedance.py` + edge cases for singular matrices.

### 6.4 Do not run the entire suite unless needed

Avoid indiscriminate full-suite runs every step. Only run full suite when:

* the step prompt explicitly asks,
* or you changed a foundational core that touches nearly everything (schemas, shared math utilities, package initialization).

---

## 7) Numerical Tolerance Standards (defaults)

Unless a step prompt specifies otherwise:

* Use relative/absolute tolerances appropriate to the calculation:

  * `pytest.approx` with `rel=1e-6` to `1e-3` depending on conditioning.

* For fits (fractional/rational), validate:

  * error curves are within the step’s defined thresholds,
  * monotonic or bounded properties when expected.

* **Passivity/Realizability:** See §16.2. Violations must be flagged.

* **Conditioning:** See §14.2. Include diagnostics for linear system solving.

---

## 8) Documentation & Artifacts (when relevant)

Only add documentation if:

* the step prompt requires it,
* or the code introduces a new public-facing interface that would be confusing without minimal docs.

When adding artifacts generation, keep outputs under:

* `artifacts/` (gitignored unless instructed)
* or a step-defined path.

**Artifact Integrity:** See §17.3. Artifacts must be content-addressed or versioned.

---

## 9) Performance Guidance (workstation constraints)

We are on a single workstation (GPU + CPU). Keep the pipeline:

* batch-friendly (vectorized ops where possible),
* cache-aware (avoid recomputing expensive sweeps),
* testable without long runtime.
* **Performance Engineering:** See §14.5. Inner loops must avoid avoidable allocations.

Avoid any design that requires “billions of explicit components.” Use hierarchical/recursive representations when needed.

---

## 10) Definition of Done (for each step)

**Note:** This definition is upgraded by §13.2 ("Operationally Complete"). A step is **DONE** only when:

* Implementation meets the step prompt’s acceptance criteria.

* The subsystem is callable, emits real artifacts, handles failures, and is deterministic (§13.2).

* Targeted tests are added/updated in `tests/`.

* Relevant tests (per §6) have been run and pass.

* Repo is GitHub-ready (no tracked venv/cache/secrets).

* If dependencies/environment were changed:

  * `./scripts/bootstrap.sh` must succeed (or fail only with explicit, actionable instructions)
  * `python -m fidp.env_check` must pass
  * locks must be updated and consistent (cpu + cu128)

* You provide a recap + `GREENLIT: YES` + suggested commit message + push command + Hardening Status.

If not done, explicitly say `GREENLIT: NO` and list the blocking issues.

---

## 11) Response Template (use at end of each step)

Use this exact structure (includes additions from §20):

1. **Recap**
2. **Files changed**
3. **Tests added/updated**
4. **Commands run**
5. **Results**
6. **Suggested commit message**
7. **Push command**
8. **Hardening checklist status** (list §14–§19 compliance)
9. **Known limitations** (must be empty for GREENLIT unless explicitly allowed)
10. **GREENLIT: YES/NO** (with reason)

---

## 12) Notes for Steps 5–14

The step prompts will reference sections from the design doc (evaluator stack, model extraction, AI search, novelty scoring, reproducibility, parallelism, validation gates, etc.). This AGENTS.md applies uniformly:

* Implement what the step asks.
* **Build the complete system for the step** (Meta-Rule §21).
* Add targeted tests for what you changed.
* Run all relevant tests (not necessarily all tests).
* Keep the repo clean for GitHub.
* Provide a suggested commit message + `./ship.sh "message"` command.
* Report and greenlight decision using the upgraded criteria.

---

## 13) “No Scaffolding” Addendum (MANDATORY for Steps 5–14)

This section **extends** (and where explicitly stated, **overrides**) earlier guidance. It exists to enforce the **No-Scaffolding** requirement: every step must land as a **fully fledged, hardened, state-of-the-art implementation**, not “minimal now, improve later.”

### 13.1 Non-Negotiable: Zero Scaffolding in Production Paths

**Forbidden in any production code path** (`src/fidp/**`), for steps 5–14:

* `pass` as a placeholder in any non-abstract method
* `raise NotImplementedError` (unless the step prompt explicitly requires a deliberately unsupported feature AND you provide an alternative fully working path)
* `TODO`, `FIXME`, `HACK`, or “stub” comments indicating incompleteness
* fake implementations to “make tests pass”
* feature flags that hide missing functionality
* silent fallbacks that reduce correctness

**If the step prompt explicitly asks for a subset**, treat that as the **floor**. Implement the complete subsystem needed for the feature to be reliable in the repo’s real workflow (evaluation → modeling → metrics → logging → reproducibility).

**Rule:** If the system can reach a runtime state where it fails due to missing code, the step is **NOT DONE**.

### 13.2 “Done” Means Operationally Complete

For steps 5–14, “DONE” means:

* The new subsystem can be exercised end-to-end via **an actual callable interface** (module API and/or script entry point)
* It emits **real artifacts** (structured outputs, logs, serialized records, plots if relevant)
* It supports **failure handling** and returns structured error diagnostics (no cryptic stack traces as the primary UX)
* It is deterministic under seed control (where randomness exists)
* It is test-verified with meaningful assertions (not just “it runs”)

### 13.3 Strict Definition of GREENLIT (Upgraded)

From this point onward:

* `GREENLIT: YES` means **you have done everything reasonably possible** to make the step’s deliverable **complete, robust, and state-of-the-art**, consistent with the step prompt.
* It is *not* sufficient to “implement the minimal asked behavior.” You must build the full implementation envelope:

  * correctness
  * determinism
  * performance
  * reproducibility
  * debug-ability
  * doc + usage clarity
  * tests that would catch regressions

If any of those are meaningfully lacking, you must output `GREENLIT: NO` and list the concrete missing items.

---

## 14) Hardening Checklist (apply to every step, always)

This checklist must be actively applied during Phase B/C and explicitly referenced in your recap if any item was relevant.

### 14.1 Correctness & Invariants

* Enforce domain invariants at boundaries (input validation):

  * component values must be finite and within bounds
  * ports must be well-defined
  * graph/circuit must be connected appropriately for the operation

* Add assertions/tests for invariants central to the step:

  * e.g., passivity margin non-negative for passive networks
  * canonical hashes stable under relabeling
  * fitted models reproduce the target response within tolerance

### 14.2 Numerical Stability

* Always include:

  * conditioning diagnostics (or at least warnings) when solving linear systems
  * controlled tolerances, consistent frequency grids, and log-spacing utilities
  * avoidance of catastrophic cancellation when computing metrics

* When fitting models:

  * record fit residuals + max error
  * reject bad fits explicitly (don’t silently accept)

### 14.3 Failure Modes Must Be First-Class

* Every “expected failure” must be representable as a structured error:

  * non-convergence
  * singular matrix
  * invalid topology
  * passivity failure
  * SPICE non-convergence or missing executable

* Tests must cover at least one representative failure mode per major module changed in the step.

### 14.4 Deterministic Artifacts

For any pipeline-like or search-like code:

* outputs must be stable under seed, including:

  * ordering of candidates
  * serialization ordering (stable dict ordering is not enough—must define canonical order)
  * selection of Pareto sets

### 14.5 Performance Engineering Is Part of “Done”

If the step touches “inner loop” code (evaluation, fitting, scoring, search), you must:

* avoid avoidable allocations in tight loops
* cache expensive intermediate results by content hash
* include at least one micro-benchmark script or benchmark test (fast) that exercises the core path

---

## 15) Quality Tooling Expectations (No-Scaffolding Mode)

This repo must be “professional-grade.” If the tooling is not already present, you may need to add it as part of a step **when it directly supports the step’s robustness**. You must follow the dependency process in §0.2.

### 15.1 Mandatory Baselines (when present, they must pass)

If configured in the repo:

* formatting/lint (e.g., ruff/black)
* type checking (mypy/pyright)
* import sorting (if configured)
* basic security scan (if configured)

**Policy:** if you introduce or modify this tooling, you must add tests or CI hooks only if the step prompt requires; otherwise keep it local but passing and documented.

### 15.2 Type Discipline

For steps 5–14:

* new public functions/classes must have type hints
* dataclasses and TypedDict/Pydantic-like schemas must be used for structured records
* avoid “Any” in public surfaces; if unavoidable, confine it internally and justify

---

## 16) Scientific Rigor Requirements (apply whenever relevant)

### 16.1 Cross-Method Consistency (when you add or change evaluators)

If a step modifies an evaluator or adds a new fidelity tier:

* create at least one test that cross-checks results against an independent method:

  * recurrence vs sparse MNA
  * sparse MNA vs SPICE
  * reduced model vs full model

* define explicit tolerances and justify them (conditioning-aware)

### 16.2 Passivity / Realizability Discipline

If a step touches:

* macromodel fitting
* MOR
* passivity metrics
* “passive effective element” claims

Then you must:

* compute and store passivity margin on an evaluation grid
* ensure passivity violations are flagged and cannot be silently ignored
* implement enforcement if the step prompt implies passivity as a requirement
* add tests that:

  * deliberately introduce a violation
  * verify detection
  * verify enforcement actually fixes it (or properly rejects)

### 16.3 “No Simulator Ghosts” Policy

When results could plausibly be artifacts (grid effects, resonant spikes, numerical instability):

* add at least one sanity check:

  * grid refinement check (coarser vs finer grid consistency)
  * perturbation check (small component perturbations don’t create “false novelty”)

* ensure reporting includes warnings for suspicious patterns

---

## 17) Reproducibility & Evidence Pack Requirements (No-Scaffolding Mode)

### 17.1 Every Step Must Leave the Repo More “Runnable”

Even if the step prompt is internal, your implementation must be runnable via:

* a module entry point, OR
* a `scripts/` entry point, OR
* a documented `python -m fidp.<module>` entry point

If you add a new subsystem, you must provide:

* a minimal usage example (docstring and/or `docs/` snippet)
* a reproducible “demo run” command

### 17.2 DesignRecord/Evidence Discipline (when relevant)

If the step generates candidates, evaluates circuits, fits models, or ranks results:

* you must store results in a structured artifact (DesignRecord or equivalent)
* you must include provenance fields:

  * seed
  * git SHA (or placeholder if unavailable in test env)
  * dependency lock fingerprint (if accessible)
  * evaluator versions

### 17.3 Artifact Integrity

Artifacts must be:

* content-addressed where practical (hash key)
* stored in stable locations (configurable path)
* never overwritten without versioning (append or new key)

---

## 18) Testing Upgrades for Strict GREENLIT

### 18.1 Tests Must Be Meaningful, Not Cosmetic

Prohibited patterns in tests:

* asserting only “no exception”
* asserting only shape/length when numeric correctness is the core purpose
* accepting overly loose tolerances without justification

Required patterns (when relevant):

* numeric validation on known circuits (goldens)
* roundtrip tests (export → parse → evaluate)
* failure-mode tests (structured errors)
* determinism tests

### 18.2 Golden Data Policy

If a step introduces a stable numerical output (e.g., baseline impedance curve, passivity margin for a canonical circuit):

* store a small golden fixture under `tests/fixtures/`
* use stable serialization (e.g., CSV/NPY/JSON with controlled formatting)
* add a regression test that compares against the golden within tolerance

### 18.3 Test Runtime Discipline

If you add slow tests:

* mark them `@pytest.mark.slow`
* keep default test runs fast
* in Phase C, run slow tests only if directly relevant to the modifications

---

## 19) Pipeline/DVC/MLflow Policy (Strict)

If a step touches pipelines or experiment orchestration, the outcome must be a **real pipeline**, not a decorative DAG:

* every stage must do real work
* every stage must have explicit inputs/outputs
* failure must be explicit and actionable
* stage outputs must be verifiable by tests (at least schema/shape checks)

If MLflow logging is involved:

* logging must include enough artifacts to reproduce results
* tests must verify that logging calls occur and that required keys are present (without depending on a real MLflow server)

---

## 20) Strict GREENLIT Report Additions (Required going forward)

In addition to the existing response template (§11), for steps 5–14 you must also include:

9. **Hardening checklist status**

   * list any relevant items from §14–§19 and confirm they were satisfied

10. **Known limitations (must be empty for GREENLIT unless explicitly allowed by step)**

* if any limitations remain, you must set `GREENLIT: NO` unless the step prompt explicitly permits deferral

**Rule:** “We can do it later” is not acceptable in No-Scaffolding Mode.

---

## 21) Meta-Rule: Interpret Prompts as “Build the Complete System for This Step”

For each step prompt:

* implement not only the direct feature, but also:

  * the surrounding reliability envelope (errors, determinism, logging, artifacts)
  * the performance envelope (caching, batching where appropriate)
  * the testing envelope (unit + regression + failure mode)
  * the documentation envelope (how to use the feature)

If you believe the prompt’s requested behavior is ambiguous, you must:

* choose the most rigorous interpretation,
* list your assumptions explicitly in Phase A,
* proceed without waiting for approval.

---

## 22) Summary of What Changed With This Addendum

* “GREENLIT” is now **far stricter**: it means the step is genuinely complete.
* “Scaffolding” is explicitly **forbidden** in production paths.
* Hardening, reproducibility, and scientific rigor are now **mandatory step deliverables**, not optional extras.
* Tests must be **meaningful**, include failures, and include golden regressions where appropriate.