
Build nothing from scratch this week. Run the RAG / BioReasoning agent on the `fall_semester/data` tables and turn in **tool traces**

The agent asks a biology question, looks up CellChat / DEGs / counts with tools, and retrieves supporting chunks. No LLM. Do not filter the CSVs by hand. Do not paste tables into Gemini.

## Setup

```bash
cd fall_semester/rag_bioreasoning
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m ephys_rag.cli ask --no-llm "Is NRXN1 DEG-supported in IDH-mutant cycling Ephys_2?"
python -m ephys_rag.cli ask --no-llm "Are neurons present as a labeled identity?"
streamlit run app.py
```

`DATA_DIR` defaults to `../data`.


1. `ask --no-llm` on Q1–Q8 in [questions.md](questions.md). Save the tool traces and the top retrieved titles.
2. For each question: which tool fired, which file, which gene/pair, and whether the required term appeared.
3. Half-page note: what the agent retrieved well vs poorly. If a required term is missing, **fix the router or retriever** — do not grep the CSV and write the answer in.

Priors (not answers): [hypotheses.md](hypotheses.md). Checklist: [WEEKLY_TODOS.md](WEEKLY_TODOS.md).
