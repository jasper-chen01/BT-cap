# Week 3 Local MedGemma and Gene Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the Week 3 local MedGemma consistency audit and an evidence-grounded LightGBM gene-ranking prototype.

**Architecture:** Extend the existing provider and evaluation boundaries with an Ollama adapter, trace mock, atomic per-call cache, deterministic replay/audit, and a separate tabular ranking module. Keep raw data and run caches external to Git while committing tested code and share-safe summaries.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, LightGBM, OpenAI Python client, Ollama, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-week3-local-medgemma-ranking-design.md`

## Global Constraints

- Use `C:\Users\gojas\BT-CAP-Capstone.git\data` as read-only scientific input.
- Use `medgemma:4b-it-q4_K_M`; do not make new Gemini or Vertex calls.
- Cache each answer before scoring and replay without model calls.
- Group validation folds by gene and average five fixed seeds.
- Do not commit raw data, run caches, model weights, credentials, or `.env`.

## Review Focus

- A partial or corrupt cache file must be rejected and re-generated, never scored as a valid run.
- Replay must not instantiate or call a live provider.
- Compound ligand/receptor fields must be split into genes consistently without turning pathway text into genes.
- Grouped cross-validation must keep every gene entirely within one fold.
- Metrics must remain defined or explicitly marked unavailable when a validation fold has one class.

---

### Task 1: Local providers and grounded context

**Files:**
- Modify: `rag_bioreasoning/src/ephys_rag/providers/factory.py`
- Create: `rag_bioreasoning/src/ephys_rag/providers/ollama.py`
- Create: `rag_bioreasoning/src/ephys_rag/providers/mock.py`
- Modify: `rag_bioreasoning/src/ephys_rag/chain.py`
- Modify: `rag_bioreasoning/src/ephys_rag/cli.py`
- Modify: `rag_bioreasoning/pyproject.toml`
- Test: `rag_bioreasoning/tests/test_local_providers.py`

**Interfaces:**
- Produces: `OllamaProvider.generate(prompt: str) -> GenerationResult`, `MockTraceProvider.generate(prompt: str) -> GenerationResult`, and `RAGEngine.ask(...)` result field `context`.

- [x] Write provider/factory/context tests and verify they fail because local providers do not exist.
- [x] Implement minimal Ollama and mock providers plus configuration and CLI choices.
- [x] Run the focused tests and the entire `rag_bioreasoning/tests` suite.

### Task 2: Atomic cache, replay, and Week 3 audit

**Files:**
- Create: `rag_bioreasoning/src/ephys_rag/week3_cache.py`
- Create: `rag_bioreasoning/src/ephys_rag/week3_audit.py`
- Create: `rag_bioreasoning/src/ephys_rag/week3_runner.py`
- Modify: `rag_bioreasoning/src/ephys_rag/cli.py`
- Test: `rag_bioreasoning/tests/test_week3_cache.py`
- Test: `rag_bioreasoning/tests/test_week3_audit.py`
- Test: `rag_bioreasoning/tests/test_week3_runner.py`

**Interfaces:**
- Consumes: `RAGEngine.ask(...)` results with exact `context`.
- Produces: `write_run_atomic`, `load_run`, `score_run`, `run_fivefold_evaluation`, and `build_week3_reports`.

- [x] Write cache/replay/scoring tests, including corrupt cache and no-live-call replay tests, and verify RED.
- [x] Implement atomic cache records and deterministic audit scoring.
- [x] Implement the five-run/resume runner and CLI `week3-run`/`week3-replay` commands.
- [x] Run focused tests and the full package suite.

### Task 3: Glioma evidence feature matrix and grouped LightGBM evaluation

**Files:**
- Create: `rag_bioreasoning/evaluation/glioma_seed_labels.csv`
- Create: `rag_bioreasoning/src/ephys_rag/ranking.py`
- Modify: `rag_bioreasoning/src/ephys_rag/cli.py`
- Modify: `rag_bioreasoning/pyproject.toml`
- Test: `rag_bioreasoning/tests/test_ranking.py`

**Interfaces:**
- Produces: `build_candidate_features(data_dir) -> DataFrame`, `evaluate_rankers(features, seeds) -> RankingEvaluation`, and `write_ranking_outputs(...)`.

- [x] Write parsing, 1,368-style candidate construction, 39-feature schema, label, leakage, baseline, and metric tests; verify RED.
- [x] Implement the minimum feature builder and documented compartment-aware seed labels.
- [x] Implement grouped five-fold/five-seed LightGBM evaluation and two baselines.
- [x] Export feature matrix, OOF scores, rankings, importances, metrics, and model card.
- [x] Run focused tests and the full package suite.

### Task 4: Installation, live runs, reports, and documentation

**Files:**
- Modify: `rag_bioreasoning/README.md`
- Modify: `rag_bioreasoning/.env.example`
- Modify: `.gitignore`
- Create: `rag_bioreasoning/WEEK3_REPORT.md`
- Create: share-safe generated summaries under `rag_bioreasoning/week3_outputs/`

**Interfaces:**
- Consumes: Tasks 1-3 CLIs and output schemas.
- Produces: replayable audit/ranking deliverables and exact reproduction commands.

- [x] Install Ollama and pull the pinned 4B quantization; record the version and model digest.
- [x] Run the mock loop, then five local MedGemma runs for Q9-Q38 with resume enabled.
- [x] Replay Q9-Q32 plus appendix and audit unstable/wrong runs.
- [x] Run the LightGBM benchmark and generate the model card.
- [x] Update documentation and build a Box-ready answer archive.
- [x] Run the full focused suite, replay command, ranking command, and a repository-wide test inventory.

### Task 5: Review and delivery

**Files:**
- Review all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: verified code and deliverables.
- Produces: scoped Git commit(s), remote branch, and Box upload or ready archive.

- [x] Perform a whole-branch review against the spec and fix Important findings with RED-GREEN tests.
- [x] Verify no secrets, raw data, model weights, or run cache files are staged.
- [ ] Commit scoped Week 3 files and push `codex/week3-local-medgemma-ranking` to the user's GitHub remote.
- [x] Upload answer artifacts to the user-selected Box folder when authenticated; otherwise report the exact ready archive path.
