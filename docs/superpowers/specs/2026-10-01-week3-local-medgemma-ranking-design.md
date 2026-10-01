# Week 3 Local MedGemma and Gene Ranking Design

## Goal

Complete the Week 3 assignment with a free local MedGemma workflow, five-run consistency audit, replayable cached reports, and a LightGBM gene-prioritization prototype built from the same CellChat, DEG, and cell-count evidence used by the RAG agent.

## Inputs and boundaries

- Use the scientific tables in `C:\Users\gojas\BT-CAP-Capstone.git\data` without copying them into Git or sending full tables to a model.
- Use the Q9-Q38 manifest already in `rag_bioreasoning/evaluation/questions.json`.
- The required Week 3 comparison table covers Q9-Q32. Q33-Q38 are an appendix because the assignment explicitly identifies Q36 as an instability check.
- Reuse cached Gemini answers from Week 2. Make no new paid Gemini or Vertex calls.
- Use `medgemma:4b-it-q4_K_M` through Ollama's OpenAI-compatible endpoint at `http://localhost:11434/v1`.
- Treat CellChat probabilities and the LightGBM output as prioritization evidence, not causal or experimental validation.

## Local model and reproducible evaluation

Add an Ollama MedGemma provider without changing the existing Vertex MedGemma provider. Add a deterministic mock provider that renders facts directly from tool traces. The mock validates the full question-to-tools-to-cache-to-report loop without model calls.

The Week 3 runner executes each question five times. After each successful generation it atomically writes one JSON file named `Q<id>_medgemma_run<run>.json`. Each record includes question identity, provider/model, run number, timestamp, exact grounded context, answer, tool traces, and retrieval titles. Existing valid files are reused; only missing or invalid files trigger calls.

Replay mode performs no model calls. It reads cache files, scores them, joins the cached Gemini record, and rebuilds CSV and Markdown reports.

## Audit semantics

- `trace_terms`: whether required terms are present in deterministic tool traces.
- `hit`: whether all required answer terms are present, case-insensitively.
- `invented_pair`: a ligand-receptor interaction asserted in the answer but absent from that run's CellChat trace.
- `invented_deg`: a gene/direction/context DEG assertion absent from that run's DEG trace.
- `faithful`: no invented pair/DEG, no contradiction of explicit trace facts, and an answer that addresses the question. The automatic conservative checks produce a review flag; the committed audited table contains the final human-audited judgment and reason.
- `unstable`: faithful or hit count is between one and four of five. The narrative must discuss Q25, Q27, and Q36 even if one is stable.

## LightGBM prototype

The unit is a CellChat ligand/receptor gene in one available cell-type and IDH context. Candidate genes come only from parsed ligand and receptor components in the CellChat table. Features are generated only from the three source-table families:

- CellChat edge counts, probabilities, sender/receiver roles, pathway diversity, and Ephys lean.
- DEG presence, direction, log2 fold change, adjusted p-value, expression fractions, and agreement across IDH groups.
- Ephys group sizes and low-count flags.

Labels come from a committed, documented seed list derived from the supervisor-provided hypotheses and pathway-biology knowledge files. Each label is compartment-aware. The list is an internal biological prior, not a comprehensive published gold standard.

Evaluation uses binary LightGBM classification with five-fold group cross-validation grouped by gene, repeated over five fixed seeds. A gene never appears in both training and validation. Report AUROC, AUPRC, and Top-25 precision for LightGBM, maximum CellChat probability alone, and a deterministic three-table evidence rule. Export the feature matrix, out-of-fold scores, ranked candidates, feature importance, metrics, and model card.

## Deliverables

- Local Ollama and mock providers with tests.
- Replay-safe five-run cache under `rag_bioreasoning/runs/`.
- `WEEK3_Q9_Q32_AUDIT.csv` and `.md`, Q33-Q38 appendix, and instability narrative.
- LightGBM feature, ranking, importance, and metrics CSV files plus a model card.
- A single command that regenerates reports from cached runs without network or model calls.
- Code and share-safe summaries pushed to a `codex/` GitHub branch.
- Answer/report bundle uploaded to Box if an authenticated Box path is available; otherwise retain a ready-to-upload ZIP and report the authentication blocker.

## Safety and reproducibility

- Never commit raw scientific data, model weights, access tokens, `.env`, credentials, or paid endpoint identifiers.
- Keep generated raw run caches out of Git; include them in the Box bundle.
- Store model name, generation parameters, data fingerprints, package versions, seeds, and commands in the reports.
- Installation and model download are external machine changes authorized by the user's Week 3 request.
