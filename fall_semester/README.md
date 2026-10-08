# Ephys RAG / BioReasoning

Ask questions over **`data/`** with the agent in `rag_bioreasoning/`.
Week 1 was traces only (`--no-llm`). Week 2 is MedGemma + Gemini on those traces.

```bash
cd rag_bioreasoning
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m ephys_rag.cli ask --no-llm "Is NRXN1 DEG-supported in IDH-mutant cycling Ephys_2?"
streamlit run app.py
```

- [WEEK2.md](WEEK2.md) — **this week**
- [ASSIGNMENT.md](ASSIGNMENT.md) — week 1 (already done)
- [WEEKLY_TODOS.md](WEEKLY_TODOS.md) — week 1 checklist
- [questions.md](questions.md) — Q1–Q38 (this week: Q9–Q32; Q33–Q38 stretch)
- [hypotheses.md](hypotheses.md) — priors, not answers
