# Week 6 portal regression

Generated: 2026-10-18T22:01:47.520559+00:00

Result: **PASS**. Q1–Q44: **44/44** API/CLI tool-and-term matches.

Each match requires equal tool sets, an allowed expected tool combination, and every required evidence group on both sides. Tool order is ignored; alternatives use OR and groups use AND. Echoed questions and lookup filters do not count as evidence, except Q36's explicitly allowed empty T-cell NRXN and Glutamate lookups, which must both execute against the requested cell group.

Transport: FastAPI TestClient /api production reasoning router. CLI results come from a separately constructed engine using the CLI ask parser with `--no-llm`. No API result is reused as CLI evidence.

Extractive answer text equality: 44/44.

Published-33: PASS; 33/33 canonical literature genes checked against `nominate_lookup` in tables_scetm, for both program models and every scored context. 32 genes have scores; unscored genes (AXL) must return HTTP 404 and an empty tool result. Context probabilities use the tool's six-decimal precision; context ranks are exact. Gene-level probabilities and ranks are checked separately against saved `arms_gene_ranks.csv`. Absent model scores remain absent.

MedGemma Q9–Q32: PASS; hits=24/24; provider calls=0; cache unchanged=True.

The in-process cache check blocks provider entry points and snapshots every cache file's SHA-256, byte size, and modification time before and after. It neither imports nor primes the runtime cache. Historical answer provenance remains in the existing cache entries; these results establish cache reuse, not a new inference or biological validation.

| Question | Match | API tools | Missing API requirements |
|---|---|---|---|
| Q1 | PASS | exclusive_pathway_lookup |  |
| Q2 | PASS | cellchat_lookup, pathway_contrast_lookup, pathway_filter |  |
| Q3 | PASS | annotation_lookup |  |
| Q4 | PASS | exclusive_pathway_lookup, pathway_contrast_lookup |  |
| Q5 | PASS | cellchat_lookup |  |
| Q6 | PASS | count_lookup, dataset_stats_lookup |  |
| Q7 | PASS | cellchat_lookup, pathway_filter |  |
| Q8 | PASS | cellchat_lookup, pathway_filter |  |
| Q9 | PASS | cellchat_lookup, pathway_contrast_lookup, pathway_filter |  |
| Q10 | PASS | cellchat_lookup, pathway_filter |  |
| Q11 | PASS | cellchat_lookup, pathway_filter |  |
| Q12 | PASS | exclusive_pathway_lookup |  |
| Q13 | PASS | exclusive_pathway_lookup |  |
| Q14 | PASS | cellchat_lookup, pathway_filter |  |
| Q15 | PASS | cellchat_lookup, pathway_filter |  |
| Q16 | PASS | cellchat_lookup |  |
| Q17 | PASS | cellchat_lookup |  |
| Q18 | PASS | cellchat_lookup |  |
| Q19 | PASS | annotation_lookup, cellchat_lookup, pathway_filter, support_join |  |
| Q20 | PASS | cellchat_lookup, pathway_contrast_lookup, pathway_filter |  |
| Q21 | PASS | cellchat_lookup, deg_lookup |  |
| Q22 | PASS | cellchat_lookup, deg_lookup |  |
| Q23 | PASS | cellchat_lookup, deg_lookup |  |
| Q24 | PASS | cellchat_lookup, deg_lookup |  |
| Q25 | PASS | cellchat_lookup, deg_lookup |  |
| Q26 | PASS | cellchat_lookup, deg_lookup |  |
| Q27 | PASS | annotation_lookup, count_lookup |  |
| Q28 | PASS | count_lookup, dataset_stats_lookup |  |
| Q29 | PASS | count_lookup |  |
| Q30 | PASS | cellchat_lookup, deg_lookup |  |
| Q31 | PASS | cellchat_lookup, deg_lookup, pathway_filter |  |
| Q32 | PASS | cellchat_lookup, deg_lookup, pathway_filter |  |
| Q33 | PASS | cellchat_lookup, pathway_filter |  |
| Q34 | PASS | cellchat_lookup, pathway_filter |  |
| Q35 | PASS | cellchat_lookup, pathway_filter |  |
| Q36 | PASS | cellchat_lookup, deg_lookup, pathway_filter |  |
| Q37 | PASS | cellchat_lookup, pathway_filter |  |
| Q38 | PASS | cellchat_lookup, deg_lookup, pathway_contrast_lookup, pathway_filter |  |
| Q39 | PASS | cellchat_lookup, deg_lookup, nominate_lookup, pathway_filter |  |
| Q40 | PASS | cellchat_lookup, deg_lookup, nominate_lookup, pathway_filter |  |
| Q41 | PASS | cellchat_lookup, nominate_lookup, pathway_filter |  |
| Q42 | PASS | nominate_lookup, pathway_filter |  |
| Q43 | PASS | cellchat_lookup, deg_lookup, nominate_lookup |  |
| Q44 | PASS | cellchat_lookup, deg_lookup, nominate_lookup |  |
