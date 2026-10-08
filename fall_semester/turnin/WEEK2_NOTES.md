# Week 2 turn-in notes

## What was implemented
- `rag_bioreasoning/src/ephys_rag/llm.py`: MedGemma (Vertex dedicated `:rawPredict` + `@requestFormat=chatCompletions`) and Gemini (`gemini-2.5-flash` on Vertex).
- Same tool traces fed to both models (`ask --dual` / `cli week2`).
- Env (repo `.env`): `MEDGEMMA_ENDPOINT_ID`, `MEDGEMMA_DEDICATED_DNS`, existing `VERTEX_*` + `GEMINI_MODEL`.

## How to re-run
```bash
cd fall_semester/rag_bioreasoning
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m ephys_rag.cli ask --dual "Is NRXN1 DEG-supported as Ephys_2-high in IDH-mutant cycling tumor?"
python -m ephys_rag.cli week2
```

## Deliverable
`fall_semester/turnin/week2_traces/`
- `Q9.json` … `Q32.json` (traces + both answers)
- `WEEK2_Q9_Q32_SUMMARY.md` / `.json` (assignment table)

## Notes
- Required terms: present for Q9–Q32 after router updates.
- MedGemma 1.5-4B sometimes emits long chain-of-thought; answers are stripped when possible. Prefer Gemini wording when MedGemma truncates mid-thought.
- Empty lookups remain allowed; do not treat model speculation as a DEG/pair.
