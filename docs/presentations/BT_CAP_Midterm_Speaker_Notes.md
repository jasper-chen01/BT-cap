# BT-CAP Midterm Speaker Notes

Presentation date: 19 October 2026

## Slide 1 — From single-cell evidence to auditable gene nominees

- The project goal is to turn large single-cell tables into traceable evidence and ranked candidates for experimental validation.
- Weeks 1–5 established the table tools, grounded MedGemma workflow, and gene ranker.
- Week 6 integrates those pieces into the portal. The portal does not retrain the models; it exposes verified outputs through one interface.

## Slide 2 — One evidence base powers two paths

- Single-cell data and public gene lists feed the derived CellChat, DEG, count, and annotation tables.
- The upper lane answers biological questions through explicit tools and preserved traces.
- The lower lane builds tumor-synaptic and immune-synapse rankers with table and scETM features.
- `nominate_lookup` links ranker scores back into the reasoning agent. The portal then exposes both lanes.

Source: `docs/presentations/BioReasoning Capstone.html`

## Slide 3 — Grounded reasoning

- The router selects the relevant table tools before any model answer is generated.
- Each trace records the tool, filters, returned evidence, and source file.
- MedGemma writes from those traces. For Q9–Q32, all 24 cached answers were replayed with zero provider calls.
- If a question is unsupported by the tables, the system returns “not in the tables.”

Sources: `fall_semester/questions.md`; `fall_semester/turnin/week6_portal/WEEK6_PORTAL_REPORT.md`

## Slide 4 — Gene nomination model

- The ranker scores a gene in a cell-type × IDH context.
- Inputs combine CellChat interactions, DEG support, cell-count safeguards, a 32-dimensional scETM embedding, and neighbor-label fractions.
- Evaluation is out of fold across five seeds and grouped folds, so a gene is never scored by a model trained on that gene.
- The best combined arm achieved AUROC 0.742 for tumor synaptic and 0.725 for immune synapse.
- Example: NRXN1 in IDH-mutant cycling tumor has `p_tumor_synaptic = 0.900521`, rank 9 in context, and is driven by the synaptic-neighbor and embedding features.

Source: `fall_semester/turnin/lgbm_scetm/WEEK5_RESULTS.md`

## Slide 5 — Portal integration

- The new Genes tab includes a gene card, ranked list, scETM neighbors, and evidence tables.
- Users can filter by model arm, threshold, novel status, housekeeping status, and three-table support, then download the ranking as CSV.
- The chat and Genes tab use the same saved score files and loader as `nominate_lookup`, preventing mismatched values.

Sources: `frontend-react/src/components/GenesTab.jsx`; `backend/api/reasoning.py`

## Slide 6 — Verification

- Q1–Q44 passed 44/44 when comparing API and CLI tool sets, required evidence terms, and extractive answer text.
- The published-gene consistency check compared 310 scored outputs. AXL was absent from both interfaces and correctly returned no score.
- Q9–Q32 produced 24/24 cache hits with zero provider calls and unchanged cache files.
- The full suite passed 294 Python tests, 8 frontend tests, and the production frontend build.
- These checks establish software consistency and provenance. They do not establish biological truth.

Source: `fall_semester/turnin/week6_portal/WEEK6_PORTAL_REPORT.md`

## Slide 7 — Conclusions and next steps

- The project now connects grounded table retrieval, model-assisted answers, gene ranking, and a reviewable web portal.
- The portal preserves the main safety boundary: unsupported questions are not answered from model memory.
- The next step is to review top novel nominees with the team and select candidates for experimental validation.
- Remaining limitations include biological validation, incomplete scores for genes outside the trained universe, and dependence on the quality of the upstream annotations and tables.

Source: verified portal demonstration on 18 October 2026.
