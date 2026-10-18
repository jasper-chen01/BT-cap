# Week 6 Portal Integration and Midterm Presentation Design

**Date:** 2026-10-18  
**Branch:** `codex/week6-portal-integration`  
**Base:** `upstream/codex/week5-nominate-lookup`  
**Primary requirement:** Integrate the grounded reasoning agent and the gene ranker into the existing portal without retraining either system.

## 1. Outcome and acceptance criteria

The portal will expose the Weeks 1–3 reasoning agent and the Weeks 4–5 gene ranker through one backend and one React interface. The portal must read the same derived tables and held-out ranker outputs as the command-line agent so that chat answers and the Genes tab cannot disagree.

The work is complete when:

1. The reasoning and gene endpoints defined in `WEEK6.md` respond with structured JSON.
2. Biological questions in portal chat use the grounded agent and include structured tool traces.
3. The Genes tab contains the gene card, ranked list, neighbours, and grounded question panels.
4. The portal regression script reports 44 of 44 Q1–Q44 questions equivalent to the CLI for tools fired and required-term retrieval.
5. Published-33 endpoint scores match `nominate_lookup` for probability and gene rank.
6. Cached MedGemma questions return without a new model call.
7. The React production build and backend test suite pass.
8. An editable midterm PowerPoint, speaker notes, updated architecture image, and portal screenshots are available in the repository.

## 2. Repository and branch strategy

Implementation will occur in an isolated worktree based on the completed Week 5 branch. This base contains current portal `main`, the fall-semester RAG package, the three LightGBM arms, scETM embeddings, and `nominate_lookup` with fold provenance.

The existing `BT-cap` checkout remains untouched because it contains unrelated local changes on `codex/week3-local-medgemma-ranking`.

The user-supplied `WEEK6.md`, overview HTML, and overview PNG remain unchanged at their original location. During implementation, presentation copies will be added under `docs/presentations/` so Git can track the midterm artifacts with the Week 6 code.

## 3. Data layout and configuration

The portal will use `data/reasoning/` as its runtime data root. A preparation script will copy the required existing outputs into this structure without recomputing them:

```text
data/reasoning/
  cellchat_ephys_plus_celltype/
  glioma_compartment_ephys_clustering/
  within_cluster_ephys_DEGs/
  within_cluster_ephys_DEGs_by_IDH/
  ranker/
    arms_gene_ranks.csv
    tables/
    tables_scetm/
    scetm_only/
  gene_embeddings/
    scetm_gene_embeddings_d32.csv
  runs/
    portal/
```

The agent configuration will prefer `EPHYS_RAG_DATA_DIR` and retain `DATA_DIR` as a backwards-compatible fallback. Ranker and embedding paths will also allow environment overrides while defaulting to folders below `EPHYS_RAG_DATA_DIR`.

The data preparation script will validate every required source before copying. It will fail with a list of missing inputs rather than create a partially usable data directory.

## 4. Shared runtime services

### 4.1 Reasoning engine

A backend service will wrap `RAGEngine.from_disk()` with `functools.lru_cache(maxsize=1)`. It will serialize retrieval hits and tool results into plain JSON objects containing tool name, source file, hit count, and row data.

The API model selector will accept exactly:

- `none`: deterministic extractive answer
- `medgemma`: MedGemma invoked from the backend process
- `gemini`: Vertex Gemini invoked from the backend process

Unsupported values return HTTP 422.

### 4.2 Shared nomination loader

The score-loading and row-selection logic currently used by `nominate_lookup` will move behind a reusable loader in the RAG package. Both `nominate_lookup` and the portal gene endpoints will call this loader.

The loader will:

- load all three arms once
- normalize gene, cell-type, IDH, model, score, rank, feature, and fold columns
- preserve missing model scores as missing values
- expose label-set and housekeeping flags
- expose three-table support fields
- provide deterministic filtering and ranking

This boundary prevents the Genes tab from reproducing or reinterpreting model logic in JavaScript.

### 4.3 scETM neighbour service

The backend will load the scETM `rho` matrix once and normalize each vector once. A gene-neighbour query will use cosine similarity, exclude the query gene, and return the top `k` rows with the relevant label-set flag.

### 4.4 Portal response cache

For MedGemma and Gemini, the cache key will hash the normalized question, model, retrieved context, and an explicit cache schema version. Cache files will live under `runs/portal/<hash>.json`.

The service will check the cache before calling a model. A successful new response will be written through a temporary file and atomically renamed before the endpoint returns. Extractive `none` responses remain deterministic and do not require the LLM cache.

## 5. Backend API

A new FastAPI router at `backend/api/reasoning.py` will be included by `backend/main.py` under `/api`.

### `POST /api/reasoning/ask`

Request:

```json
{"question": "...", "model": "none"}
```

Response:

```json
{
  "answer": "...",
  "traces": [],
  "hits": [],
  "model": "none",
  "cached": false
}
```

If the grounded tools and retrieval return no relevant evidence, the endpoint returns `not in the tables` rather than asking an LLM to guess.

### `GET /api/genes/{symbol}`

Returns all available score rows across contexts and arms, gene-level badges, CellChat rows, DEG rows, and the explanations required by the gene card. A model that did not score a context contributes no row.

### `GET /api/genes/ranked`

Supports the required model, arm, housekeeping, novelty, three-table, minimum-cell, and limit parameters. The endpoint returns ranked rows and a stable column order suitable for CSV download.

### `GET /api/genes/{symbol}/neighbors`

Returns the top scETM cosine neighbours and `in_label_set` for the selected model. The default and maximum `k` values are 20 and 100.

Errors will use FastAPI validation for malformed parameters, HTTP 404 for an unknown gene, and HTTP 503 for missing runtime data or an unavailable configured model.

## 6. Portal chat integration

`backend/services/chat_agent.py` will preserve the existing greeting, help, status, upload, and annotation branches.

Before the general Gemini branch, it will call `infer_query_slots`. If the parser detects a gene, ligand–receptor pair, cell type, Ephys state, IDH group, or pathway, the service will call the shared reasoning service. It will never send that biological question to bare Gemini.

`ChatMessage` will gain an optional `traces: list[dict]` field. `ChatPage.jsx` will preserve traces returned by the backend and render them in a collapsed section below the corresponding assistant message.

Portal how-to questions and the existing UMAP summary flow may continue to use Gemini because they do not make claims about the reasoning tables.

## 7. Genes tab

`frontend-react/src/components/GenesTab.jsx` will own the new interface. `PortalPage.jsx` will only import it, add the tab button, and render it when active.

### 7.1 Gene card

The gene search will render a 6-cell-type by 2-IDH-context grid. Each context will show table-only and tables-plus-scETM scores side by side when available. Missing model output will display as blank.

Badges will show SynGO, GO:0001772, published-33, housekeeping, and three-table-rule membership. The selected context will expose top explanation features, CellChat edges, and its DEG row.

### 7.2 Ranked list

Controls will select program model and feature arm. Filters will exclude housekeeping genes by default and support novel-only, three-table-rule, minimum cell count, and result limit. CSV export will contain the currently displayed rows.

### 7.3 Neighbours

The neighbours panel will display the top 20 scETM neighbours, cosine similarity, and label membership. It will make the Level 2 feature interpretable without calculating neighbours in the browser.

### 7.4 Ask

The ask panel will call `/api/reasoning/ask`, provide No LLM, MedGemma, and Gemini options, and show the answer, retrieved titles, and collapsed JSON traces.

The browser will not compute probabilities, ranks, similarities, or rule checks.

## 8. Regression and tests

### 8.1 Backend unit tests

Tests will cover:

- shared score loading and preservation of missing scores
- gene lookup, ranking filters, and neighbour ordering
- structured trace serialization
- cache hits and misses
- chat routing for grounded and non-grounded questions
- `not in the tables` behavior

### 8.2 Frontend verification

The React production build must succeed. Component behavior will be verified through API-backed smoke checks and browser inspection of the Genes and Chat views.

### 8.3 Portal regression script

`scripts/portal_regression.py` will use the portal API and CLI in the same environment.

It will:

1. Run Q1–Q44 with `model=none` through the API.
2. Compare tools fired and required-term retrieval with CLI output.
3. Fail unless all 44 match.
4. Check published-33 probability and rank against `nominate_lookup`.
5. Check that Q9–Q32 MedGemma responses come from cache and do not create new model calls.

The script will produce JSON, CSV, and a short Markdown report under `fall_semester/turnin/week6_portal/`.

## 9. Midterm presentation deliverables

The repository will contain:

- `docs/presentations/BT_CAP_Midterm_2026-10-19.pptx`
- `docs/presentations/BT_CAP_Midterm_Speaker_Notes.md`
- an updated overview HTML and PNG
- portal screenshots used in the deck

The deck will use the visual language of the supplied overview and contain approximately seven slides:

1. Clinical and computational objective
2. Weeks 1–6 architecture
3. Grounded agent and evidence tables
4. LightGBM and scETM gene ranking
5. Week 6 portal integration
6. Regression and representative biological results
7. Five-minute demonstration sequence and next steps

The final architecture visual will update the Week 6 status only after the implementation and regression evidence support that change. Speaker notes will distinguish completed work, measured results, and remaining limitations.

## 10. Scope limits

This work will not retrain scETM or LightGBM, alter the biological labels, replace the existing annotation workflow, or claim that model probabilities establish biological causality. It will not require live MedGemma inference for cached Week 3 questions during the presentation.

If the existing cached runs cannot map reliably to the Week 6 cache key, the portal will import them through a documented conversion step and record their original provenance.

## 11. Demonstration sequence

The final five-minute demonstration will follow the Week 6 brief:

1. Search NRXN1 and inspect both scores, explanation features, and CellChat evidence.
2. Filter tumor-synaptic rankings to novel, non-housekeeping, three-table-supported genes and show the top ten.
3. Inspect RPL5 neighbours to explain the ribosomal block.
4. Ask the NRXN1 DEG question with MedGemma and open its traces.
5. Ask Q39 in chat and show the `nominate_lookup` arm and fold.
6. Ask an unsupported question and show the `not in the tables` response.
