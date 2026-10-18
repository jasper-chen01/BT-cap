# Week 6 Portal Integration and Midterm Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the grounded Q1–Q44 reasoning agent and held-out gene ranker into the existing portal, prove that portal results match the CLI, and produce an editable midterm presentation from the verified integration.

**Architecture:** A shared RAG package owns biological routing, score loading, and scETM neighbours. A cached backend service exposes that package through FastAPI, while the React Genes tab and chat only render endpoint data. A regression script compares API behavior with the CLI before the architecture overview and PowerPoint report Week 6 as complete.

**Tech Stack:** Python 3.9+, FastAPI, Pydantic 2, pandas, NumPy, scikit-learn, React 18, Vite 5, Tailwind CSS, Node test runner, JavaScript `@oai/artifact-tool` for PowerPoint creation.

**Spec:** `docs/superpowers/specs/2026-10-18-week6-portal-integration-design.md`

## Global Constraints

- Do not retrain scETM or LightGBM in the portal.
- `EPHYS_RAG_DATA_DIR` points to `data/reasoning/`; `DATA_DIR` remains a backwards-compatible fallback.
- The Genes tab and `nominate_lookup` must use the same Python score loader.
- A model that never scored a context returns no row, never probability zero.
- Biological questions never fall through to bare Gemini; empty evidence returns `not in the tables`.
- `model` accepts only `none`, `medgemma`, or `gemini`.
- LLM cache keys include normalized question, model, retrieved context, and cache schema version.
- Browser code renders endpoint results and performs no biological score, rank, similarity, or rule computation.
- The Week 6 regression target is 44 of 44 API/CLI matches.
- Preserve all unrelated local changes in the existing `BT-cap` checkout.

## Review Focus

1. **Missing model output:** an unscored gene-context combination must remain absent instead of becoming `0`; Task 2 adds `test_gene_rows_preserve_missing_model_scores` and Task 4 checks the API shape.
2. **Missing runtime data:** a partial or absent `data/reasoning` directory must produce a clear preparation error or HTTP 503, never a half-loaded service; Tasks 1 and 4 test this.
3. **Grounding failure:** a biological question with no matching rows must return `not in the tables` and must not call Gemini; Task 5 tests both assertions.
4. **Cache collision or corruption:** model and context changes must produce different keys, and interrupted writes must not create valid cache entries; Task 3 tests key separation and atomic writes.
5. **Router regression:** Week 5 words such as `score`, `rank`, and `candidate` must not take over Q1–Q38; Task 8 compares API and CLI tools for all 44 questions.

---

### Task 1: Runtime data layout and configuration

**Files:**
- Create: `scripts/prepare_reasoning_data.py`
- Create: `tests/test_prepare_reasoning_data.py`
- Modify: `fall_semester/rag_bioreasoning/src/ephys_rag/config.py`
- Modify: `backend/config.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `prepare_reasoning_data(source_root: Path, destination: Path) -> dict[str, Path]`
- Produces: `settings.EPHYS_RAG_DATA_DIR: Path`, `settings.EPHYS_RAG_RUNS_DIR: Path`
- Produces: RAG configuration constants for three arm score files, gene ranks, and scETM embeddings.

- [ ] **Step 1: Write failing preparation and configuration tests**

Add tests named `test_prepare_reasoning_data_copies_required_files`, `test_prepare_reasoning_data_lists_all_missing_sources`, and `test_config_prefers_ephys_rag_data_dir`. Assert the expected directory names from the specification and verify that a missing source leaves the destination incomplete.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest tests/test_prepare_reasoning_data.py -v`

Expected: FAIL because the preparation function and new configuration fields do not exist.

- [ ] **Step 3: Implement the preparation script and environment resolution**

Implement the exact function signature above. Copy tracked Weeks 1–5 outputs into `data/reasoning` and validate all inputs before copying any file. Update agent configuration to prefer `EPHYS_RAG_DATA_DIR`, then `DATA_DIR`, then existing repository defaults.

- [ ] **Step 4: Run the focused tests and create local runtime data**

Run: `pytest tests/test_prepare_reasoning_data.py -v`

Run: `python scripts/prepare_reasoning_data.py --source . --destination data/reasoning`

Expected: tests PASS and the script reports every prepared path.

- [ ] **Step 5: Commit the runtime setup**

```bash
git add scripts/prepare_reasoning_data.py tests/test_prepare_reasoning_data.py fall_semester/rag_bioreasoning/src/ephys_rag/config.py backend/config.py .env.example
git commit -m "feat: prepare Week 6 reasoning data"
```

### Task 2: Shared nomination and neighbour loader

**Files:**
- Create: `fall_semester/rag_bioreasoning/src/ephys_rag/nomination.py`
- Create: `fall_semester/rag_bioreasoning/tests/test_nomination_store.py`
- Modify: `fall_semester/rag_bioreasoning/src/ephys_rag/tools/registry.py`
- Modify: `fall_semester/rag_bioreasoning/src/ephys_rag/tools/__init__.py`
- Modify: `fall_semester/rag_bioreasoning/tests/test_nominate_lookup.py`

**Interfaces:**
- Produces: `NominationStore.from_config() -> NominationStore`
- Produces: `NominationStore.gene_rows(symbol: str) -> list[dict]`
- Produces: `NominationStore.ranked_rows(*, model: str, arm: str, exclude_housekeeping: bool, novel_only: bool, rule_pass: bool | None, min_n: int, limit: int) -> list[dict]`
- Produces: `NominationStore.nominate_rows(*, gene: str | None, celltype: str | None, idh: str | None, order_by: str, exclude_syngo: bool, limit: int) -> list[dict]`
- Produces: `NominationStore.neighbors(symbol: str, *, model: str, k: int = 20) -> list[dict]`
- Consumes: file paths produced by Task 1.

- [ ] **Step 1: Write failing store tests**

Test all three arms, case-insensitive gene lookup, missing probability preservation, stable ranking, filters, `k` exclusion of the query gene, descending cosine similarity, label flags, and an unknown-gene exception.

- [ ] **Step 2: Run the store tests and confirm failure**

Run: `pytest fall_semester/rag_bioreasoning/tests/test_nomination_store.py -v`

Expected: FAIL because `NominationStore` does not exist.

- [ ] **Step 3: Implement `NominationStore` and refactor `nominate_lookup`**

Use pandas for score tables and NumPy for normalized cosine similarity. Keep held-out scores, ranks, explanation strings, and fold strings unchanged. Inject a store into `ToolRegistry` for tests and default to `NominationStore.from_config()` in production.

- [ ] **Step 4: Run store and Week 5 regression tests**

Run: `pytest fall_semester/rag_bioreasoning/tests/test_nomination_store.py fall_semester/rag_bioreasoning/tests/test_nominate_lookup.py fall_semester/rag_bioreasoning/tests/test_oof_fold_provenance.py -v`

Expected: all tests PASS and Q39–Q44 values remain unchanged.

- [ ] **Step 5: Commit the shared loader**

```bash
git add fall_semester/rag_bioreasoning/src/ephys_rag/nomination.py fall_semester/rag_bioreasoning/src/ephys_rag/tools fall_semester/rag_bioreasoning/tests
git commit -m "refactor: share nomination data loader"
```

### Task 3: Cached backend reasoning service

**Files:**
- Create: `backend/services/reasoning_service.py`
- Create: `backend/tests/test_reasoning_service.py`
- Create: `backend/tests/__init__.py`

**Interfaces:**
- Produces: `get_rag_engine() -> RAGEngine` with `lru_cache(maxsize=1)`
- Produces: `get_nomination_store() -> NominationStore` with `lru_cache(maxsize=1)`
- Produces: `serialize_tool_result(result: ToolResult) -> dict`
- Produces: `PortalAnswerCache.make_key(question: str, model: str, context: str) -> str`
- Produces: `ReasoningService.ask(question: str, model: str) -> dict`
- Consumes: `NominationStore` from Task 2.

- [ ] **Step 1: Write failing service and cache tests**

Cover engine singleton behavior, JSON trace fields, deterministic `none` answers, distinct cache keys for model and context changes, cache hits that avoid the provider, and temporary files that never count as valid cache entries.

- [ ] **Step 2: Run the service tests and confirm failure**

Run: `pytest backend/tests/test_reasoning_service.py -v`

Expected: FAIL because the service and cache do not exist.

- [ ] **Step 3: Implement the service and atomic cache**

Use SHA-256 over a canonical JSON payload with schema version `1`. Serialize traces as `{tool, source_file, n_hits, filters, rows, note}` and hits as `{title, score, kind, source_file}`. Return `not in the tables` when traces and hits contain no evidence.

- [ ] **Step 4: Import existing Q9–Q32 MedGemma results into the portal cache**

Add `import_week2_cache(source_dir: Path, cache: PortalAnswerCache, engine: RAGEngine) -> int` to the service module. Use each saved question, MedGemma answer, and current grounded context so cache keys preserve provenance.

- [ ] **Step 5: Run the service tests**

Run: `pytest backend/tests/test_reasoning_service.py -v`

Expected: PASS, including provider call count zero on a cache hit.

- [ ] **Step 6: Commit the reasoning service**

```bash
git add backend/services/reasoning_service.py backend/tests
git commit -m "feat: add cached portal reasoning service"
```

### Task 4: Reasoning and gene API

**Files:**
- Create: `backend/api/reasoning.py`
- Create: `backend/tests/test_reasoning_api.py`
- Modify: `backend/models/schemas.py`
- Modify: `backend/main.py`

**Interfaces:**
- Produces: `POST /api/reasoning/ask`
- Produces: `GET /api/genes/{symbol}`
- Produces: `GET /api/genes/ranked`
- Produces: `GET /api/genes/{symbol}/neighbors`
- Consumes: cached services from Task 3.

- [ ] **Step 1: Write failing FastAPI endpoint tests**

Use `TestClient` and dependency overrides. Assert response keys, valid model values, missing data as HTTP 503, unknown gene as HTTP 404, missing scores as absent rows, ranking filter semantics, `k` bounds of 1–100, and stable CSV-ready keys.

- [ ] **Step 2: Run the API tests and confirm failure**

Run: `pytest backend/tests/test_reasoning_api.py -v`

Expected: FAIL with missing router or 404 responses.

- [ ] **Step 3: Add request schemas, endpoints, and router registration**

Add `ReasoningAskRequest(question: str, model: Literal["none", "medgemma", "gemini"] = "none")`. Keep gene endpoint responses as JSON dictionaries whose biological fields come directly from `NominationStore` and the RAG registry.

- [ ] **Step 4: Run service and API tests**

Run: `pytest backend/tests/test_reasoning_service.py backend/tests/test_reasoning_api.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the API**

```bash
git add backend/api/reasoning.py backend/models/schemas.py backend/main.py backend/tests/test_reasoning_api.py
git commit -m "feat: expose reasoning and gene endpoints"
```

### Task 5: Grounded portal chat and trace schema

**Files:**
- Create: `backend/tests/test_chat_grounding.py`
- Modify: `backend/models/schemas.py`
- Modify: `backend/services/chat_agent.py`
- Modify: `backend/api/chat.py`

**Interfaces:**
- Produces: `ChatMessage.traces: list[dict] | None`
- Produces: `ChatAgent._is_grounded_question(message: str) -> bool`
- Consumes: `infer_query_slots` and `ReasoningService.ask`.

- [ ] **Step 1: Write failing routing tests**

Assert that NRXN1, ligand–receptor, pathway, Ephys, IDH, and cell-type questions call the reasoning service; an unsupported biological question returns `not in the tables`; no grounded branch invokes Gemini; upload, annotate, status, portal help, and UMAP summary behavior remains unchanged.

- [ ] **Step 2: Run the routing tests and confirm failure**

Run: `pytest backend/tests/test_chat_grounding.py -v`

Expected: FAIL because biological messages currently fall through to Gemini.

- [ ] **Step 3: Implement grounded routing before general Gemini**

Retain the existing branch order for file and portal operations. Add the grounded branch immediately before general conversation. Attach structured traces to the returned `ChatMessage`.

- [ ] **Step 4: Run chat and backend tests**

Run: `pytest backend/tests -v`

Expected: PASS.

- [ ] **Step 5: Commit grounded chat**

```bash
git add backend/models/schemas.py backend/services/chat_agent.py backend/api/chat.py backend/tests/test_chat_grounding.py
git commit -m "feat: ground biological portal chat"
```

### Task 6: Frontend API client, trace panel, and Genes tab

**Files:**
- Create: `frontend-react/src/lib/reasoningApi.js`
- Create: `frontend-react/src/lib/reasoningApi.test.js`
- Create: `frontend-react/src/components/TracePanel.jsx`
- Create: `frontend-react/src/components/GenesTab.jsx`
- Modify: `frontend-react/package.json`
- Modify: `frontend-react/src/components/PortalPage.jsx`

**Interfaces:**
- Produces: `getGene(symbol)`, `getRankedGenes(filters)`, `getNeighbors(symbol, model, k)`, `askReasoning(question, model)`
- Produces: `rowsToCsv(rows: object[]) -> string`
- Produces: `normalizeChatMessage(message: object) -> object`
- Produces: `<TracePanel traces={[]} defaultOpen={false} />`
- Produces: `<GenesTab />`
- Consumes: endpoints from Task 4.

- [ ] **Step 1: Write failing Node tests for API parameter and CSV behavior**

Use `node:test` to assert URL encoding, default filters, preservation of blank scores, CSV column order, quoting, and an empty result set.

- [ ] **Step 2: Run the frontend tests and confirm failure**

Run: `npm test --prefix frontend-react`

Expected: FAIL because the client module and test script do not exist.

- [ ] **Step 3: Implement the API client and trace panel**

Use `VITE_API_BASE_URL` with the existing localhost fallback. Keep CSV conversion limited to endpoint rows.

- [ ] **Step 4: Implement the four-panel Genes tab and minimal PortalPage wiring**

Render the 6-by-2 context grid, required badges, selected-context evidence, ranked filters and download, top 20 neighbours, and ask panel. Add only the import, button, and conditional render to `PortalPage.jsx`.

- [ ] **Step 5: Run frontend tests and production build**

Run: `npm test --prefix frontend-react`

Run: `npm run build --prefix frontend-react`

Expected: tests PASS and Vite completes without errors.

- [ ] **Step 6: Commit the Genes tab**

```bash
git add frontend-react/package.json frontend-react/src/lib frontend-react/src/components/GenesTab.jsx frontend-react/src/components/TracePanel.jsx frontend-react/src/components/PortalPage.jsx
git commit -m "feat: add portal Genes tab"
```

### Task 7: Chat trace rendering

**Files:**
- Modify: `frontend-react/src/components/ChatPage.jsx`
- Modify: `frontend-react/src/lib/reasoningApi.test.js`

**Interfaces:**
- Consumes: `ChatMessage.traces` from Task 5 and `TracePanel` from Task 6.
- Produces: normalized frontend messages that preserve `traces`.

- [ ] **Step 1: Add a failing message-normalization test**

Assert that `normalizeChatMessage` preserves assistant content and trace arrays and that messages without traces remain valid.

- [ ] **Step 2: Run the frontend test and confirm failure**

Run: `npm test --prefix frontend-react`

Expected: FAIL because normalization currently drops traces.

- [ ] **Step 3: Preserve and render collapsed traces**

Remove the duplicate hard-coded `API_BASE`, use the shared client and `normalizeChatMessage`, preserve `data.message.traces`, and render `TracePanel` only for assistant messages with traces.

- [ ] **Step 4: Run frontend tests and build**

Run: `npm test --prefix frontend-react`

Run: `npm run build --prefix frontend-react`

Expected: PASS.

- [ ] **Step 5: Commit chat traces**

```bash
git add frontend-react/src/components/ChatPage.jsx frontend-react/src/lib/reasoningApi.test.js
git commit -m "feat: display grounded chat traces"
```

### Task 8: Portal regression suite and evidence files

**Files:**
- Create: `scripts/portal_regression.py`
- Create: `tests/test_portal_regression.py`
- Create at runtime: `fall_semester/turnin/week6_portal/portal_regression.json`
- Create at runtime: `fall_semester/turnin/week6_portal/portal_regression.csv`
- Create at runtime: `fall_semester/turnin/week6_portal/WEEK6_PORTAL_REPORT.md`

**Interfaces:**
- Produces: `load_question_cases(path: Path) -> list[QuestionCase]`
- Produces: `compare_api_and_cli(question: QuestionCase, api_result: dict, cli_result: dict) -> dict`
- Produces: command `python scripts/portal_regression.py --base-url ... --questions fall_semester/questions.md`
- Consumes: API from Task 4 and the same CLI package used by the portal.

- [ ] **Step 1: Write failing parser and comparison tests**

Assert exactly 44 parsed questions, expected tools and required terms for Q1, Q21, and Q39, order-insensitive tool comparison, required-term comparison, published-33 score equality, and cache files unchanged during cached Q9–Q32 runs.

- [ ] **Step 2: Run regression unit tests and confirm failure**

Run: `pytest tests/test_portal_regression.py -v`

Expected: FAIL because the regression module does not exist.

- [ ] **Step 3: Implement regression parsing, comparison, and reports**

Run API checks against a live base URL or FastAPI `TestClient`. Invoke CLI logic in-process so both sides use the same environment while keeping independently collected outputs.

- [ ] **Step 4: Run all 44 extractive comparisons and supporting checks**

Run: `python scripts/portal_regression.py --questions fall_semester/questions.md --output-dir fall_semester/turnin/week6_portal`

Expected: `44/44` API/CLI matches, published-33 equality, and Q9–Q32 cache hits with no new model calls.

- [ ] **Step 5: Commit regression code and evidence**

```bash
git add scripts/portal_regression.py tests/test_portal_regression.py fall_semester/turnin/week6_portal
git commit -m "test: verify portal matches reasoning CLI"
```

### Task 9: Full-stack verification and portal screenshots

**Files:**
- Create: `docs/presentations/screenshots/genes-nrxn1.png`
- Create: `docs/presentations/screenshots/ranked-novel.png`
- Create: `docs/presentations/screenshots/chat-q39-trace.png`
- Create: `docs/presentations/screenshots/not-in-tables.png`
- Modify: `README.md`

**Interfaces:**
- Consumes: completed backend and frontend.
- Produces: verified demo startup instructions and presentation-ready screenshots.

- [ ] **Step 1: Run the complete automated test suite**

Run: `pytest fall_semester/rag_bioreasoning/tests backend/tests tests -v`

Run: `npm test --prefix frontend-react`

Run: `npm run build --prefix frontend-react`

Expected: all tests and build PASS.

- [ ] **Step 2: Start the backend and frontend for browser verification**

Run the backend on port 8000 and Vite on port 5173 with `EPHYS_RAG_DATA_DIR` set to the prepared directory.

- [ ] **Step 3: Execute the six-step Week 6 demo in the browser**

Verify NRXN1 scores and evidence, novel ranking filters, RPL5 neighbours, cached MedGemma traces, Q39 fold provenance, and an unsupported question.

- [ ] **Step 4: Capture the four named screenshots and update setup documentation**

Document data preparation, editable package installation, environment variables, startup, regression, and demo commands.

- [ ] **Step 5: Commit verification artifacts**

```bash
git add README.md docs/presentations/screenshots
git commit -m "docs: add Week 6 portal demo evidence"
```

### Task 10: Updated overview and editable midterm deck

**Files:**
- Create: `docs/presentations/BioReasoning Capstone.html`
- Create: `docs/presentations/BTCAP_Bioreasoning_Overview.png`
- Create: `docs/presentations/build_midterm_deck.mjs`
- Create: `docs/presentations/BT_CAP_Midterm_2026-10-19.pptx`
- Create: `docs/presentations/BT_CAP_Midterm_Speaker_Notes.md`

**Interfaces:**
- Consumes: regression report and screenshots from Tasks 8–9.
- Produces: a reproducible editable PowerPoint and its speaker notes.

- [ ] **Step 1: Copy the supplied overview source and update only verified status**

Preserve the existing architecture and visual language. Change Week 6 from in progress to done only if Task 8 reports 44/44 and Task 9 passes. Update the status date to 18 October 2026.

- [ ] **Step 2: Render and inspect the updated overview**

Open the HTML through a local server, capture the full diagram at presentation resolution, and verify text remains legible without clipping.

- [ ] **Step 3: Build the seven-slide deck with `@oai/artifact-tool`**

Use the overview image, verified metrics, and portal screenshots. Keep titles at 32 points or larger and body text at 17 points or larger. Add concise citations and distinguish measured results from planned next steps.

- [ ] **Step 4: Render every slide and perform visual QA**

Check text fit, image cropping, table readability, slide order, speaker-note alignment, and editable evidence objects. Fix every overlap or overflow before export.

- [ ] **Step 5: Commit presentation deliverables**

```bash
git add docs/presentations
git commit -m "docs: add verified midterm presentation"
```

### Task 11: Final branch verification and handoff

**Files:**
- Modify if needed: files implicated by final verification only

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: clean branch, final verification record, and upload-ready deliverables.

- [ ] **Step 1: Run final verification from a clean process**

Run: `pytest fall_semester/rag_bioreasoning/tests backend/tests tests -v`

Run: `npm test --prefix frontend-react`

Run: `npm run build --prefix frontend-react`

Run: `python scripts/portal_regression.py --questions fall_semester/questions.md --output-dir fall_semester/turnin/week6_portal`

Expected: all automated checks PASS and portal regression reports `44/44`.

- [ ] **Step 2: Confirm repository state and deliverable presence**

Run: `git status --short`

Run: `git log --oneline upstream/codex/week5-nominate-lookup..HEAD`

Expected: clean working tree and all Week 6 commits visible.

- [ ] **Step 3: Review the complete branch and fix only verified findings**

Review API contracts, grounding boundary, score consistency, trace privacy, regression evidence, and presentation claims.

- [ ] **Step 4: Commit any final verified fixes**

Use a narrowly scoped commit message that names the corrected behavior. Skip this step if review finds no issues.

- [ ] **Step 5: Present the portal, regression report, screenshots, deck, notes, and Box-ready result files to the user**

List exact local paths, the 44/44 result, remaining limitations, and the recommended GitHub integration action.
