# Week 3 report — local MedGemma consistency and gene ranking

Date: 2026-10-01

## Completion summary

- Installed Ollama 0.35.0 and downloaded `medgemma:4b-it-q4_K_M` (Ollama model ID `9fe4e9a6c9bd`, 3.3 GB; model blob SHA-256 `d1d201b9e957ab5f47b30f765d17ed3f3224a51f4a625e8730f6b60553417f23`).
- Built and tested an Ollama provider using the OpenAI-compatible local API.
- Built an atomic per-question/run cache, cache validation, resume behavior, and replay-only table generation.
- Ran the full mock pipeline 5 times over Q9-Q38 (150 cached mock responses).
- Ran local MedGemma 5 times over Q9-Q38 (150 cached model responses). No Vertex, Gemini, or other paid model call was made for Week 3.
- Reused the existing Week 2 Gemini cache for the comparison column.
- Human-audited every MedGemma run for biological role and logical correctness, separately from exact required-term matching.
- Built a 39-feature LightGBM gene-ranking prototype with gene-grouped cross-validation and two baselines.
- Generated a Box-ready archive containing the report, tables, rankings, and raw run cache.

## Local model setup

The laptop has 31.7 GB RAM and an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM. The official multimodal MedGemma build could not use the CUDA path because Ollama attempted a 1.85 GB pinned host-memory allocation. The verified run therefore used CPU-only Ollama with CUDA and Vulkan disabled and a 4,096-token context window. This changes speed, not model weights or answers.

The raw cache records the actual provider (`ollama`), model (`medgemma:4b-it-q4_K_M`), prompt context, tool traces, answer, run number, timestamp, and latency. Valid cached files are never called again during resume. Replay loads only these files and does not instantiate a provider.

## Q9-Q38 consistency results

The main assignment table covers Q9-Q32 (24 questions, 120 runs). Human review found 83/120 faithful answers. Fifteen questions were faithful in all five runs, seven were unfaithful in all five runs, and two had mixed faithful outcomes. Exact required-term matching was 61/120; this lower number is intentionally reported separately because a correct contextual answer can omit a literal term, while an incorrect answer can repeat every required term.

The Q33-Q38 appendix contains 6 additional questions and 30 runs. Human review found 6/30 faithful answers. One question was faithful in all five runs, four were unfaithful in all five runs, and Q36 was faithful in only one run. Exact required-term matching was 17/30.

No run invented a ligand-receptor pair or DEG direction under the trace-based invention audit. The important failures were instead omissions, role reversal, or incorrect logical polarity:

- **Q25:** faithful 5/5, exact-term hit 0/5. All answers correctly distinguished EGFR as Ephys_2-high in IDH-mutant cycling tumor and Ephys_1-high in IDH-WT OPC-like tumor, but did not repeat every manifest spelling.
- **Q27:** faithful 5/5, exact-term hit 5/5. All answers identified `MES_like_tumor/TAM1/microglia / Ephys_2` as the six-cell group.
- **Q36:** faithful 1/5, exact-term hit 5/5. Only run 3 correctly said T cells do not send NRXN or Glutamate; the other four runs asserted the opposite despite the zero-hit traces.
- **Q10:** faithful 0/5 despite exact-term hit 5/5. Every answer repeated `HLA-DRA_CD4` but reversed the ligand/receptor roles and called HLA-DRA the receptor.
- **Q16:** faithful 4/5. One run contradicted its own list of varying p-values.
- **Q29:** faithful 4/5. One response was truncated before reaching a supported conclusion.
- **Q38:** faithful 0/5. Every response said the evidence “does not refute” the claim even while citing evidence that does refute it.

This comparison demonstrates why the audited faithfulness column is the primary evaluation and raw term-hit rate is only a diagnostic.

## LightGBM gene-ranking prototype

The ranked unit is one CellChat ligand/receptor gene in one IDH status × transcriptomic cell-type context. Splitting CellChat complexes produced 114 biological genes. Crossing them with 12 contexts produced 1,368 candidate rows.

Each row has 39 numeric features from the same three evidence sources used by the RAG tools:

- CellChat edge counts, probabilities, direction, pathway breadth, partners, and Ephys lean.
- DEG presence, log2 fold change, adjusted p-value, direction, expression fractions, and cross-IDH agreement.
- Ephys_1/Ephys_2 group sizes and a low-count warning.

The prototype labels are an explicit 33-gene, compartment-aware seed list derived from the project hypotheses. They are not an independently curated published gold standard. All 33 seeds occur among the 114 CellChat genes. Validation uses five gene-grouped folds and five fixed seeds, so a candidate is never scored by a model trained on the same gene.

| Scorer | AUROC | AUPRC | Top-25 precision |
|---|---:|---:|---:|
| LightGBM | 0.821 | 0.540 | 1.00 |
| CellChat probability alone | 0.731 | 0.385 | 1.00 |
| Three-table rule alone | 0.730 | 0.241 | 0.16 |

These values measure recovery of the internal seed list. They do not establish external biological validity, clinical utility, or experimental causality.

## Deliverables

- `week3_outputs/WEEK3_Q9_Q32_AUDIT.csv` and `.md`: required main comparison table.
- `week3_outputs/WEEK3_Q33_Q38_APPENDIX.csv` and `.md`: additional question results.
- `week3_outputs/WEEK3_RUN_LEVEL_AUDIT.csv`: all 150 run-level judgments and raw answers.
- `week3_outputs/WEEK3_INSTABILITY_NOTES.md`: unstable questions plus required Q25/Q27/Q36 discussion.
- `week3_outputs/ranking/`: candidate feature matrix, OOF rankings, metrics, feature importance, and model card.
- `evaluation/week3_human_audit.json`: transparent human-audit decisions and reasons.
- `runs/`: raw mock and MedGemma caches (included in the share archive, excluded from Git).

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[local,ranking,dev]"
.\.venv\Scripts\python.exe -m ephys_rag.cli week3-run --provider mock --cache-label mock
.\.venv\Scripts\python.exe -m ephys_rag.cli week3-run --provider ollama --cache-label medgemma
.\.venv\Scripts\python.exe -m ephys_rag.cli week3-replay --table --gemini-cache <week2-evaluation.json>
.\.venv\Scripts\python.exe -m ephys_rag.cli rank-genes
.\.venv\Scripts\python.exe -m pytest tests -q
```

Raw scientific data, model weights, credentials, `.env`, and run caches are not committed to Git.

## Verification

- The focused bio-reasoning test suite passes: 79 tests passed.
- The repository-wide inventory reports 85 passed and 3 failures. All three failures are in the historical CellChat analysis tests and are caused by the optional `anndata` package not being installed; the same failures existed before the Week 3 implementation.
- The replayed CSVs contain 24 main questions, 6 appendix questions, and 150 unique MedGemma question/run records with a completed human-audit decision for every run.
- The ranking CSVs contain 1,368 unique gene-context candidates, 39 numeric evidence features, 3 scorer rows, and 39 feature-importance rows.
- `week3_bundle.zip` contains 314 entries, including all 150 MedGemma caches and 150 mock caches.
