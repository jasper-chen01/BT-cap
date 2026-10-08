# rag_bioreasoning

Starter agent for the fall data drop. Tables first, tools second, LLM last.

```bash
cd rag_bioreasoning
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# DATA_DIR defaults to ../data

python -m ephys_rag.cli stats
python -m ephys_rag.cli contrast --compartment tumor
python -m ephys_rag.cli tools --gene NRXN1 --celltype cycling_tumor --idh IDH_Mutant
python -m ephys_rag.cli ask --no-llm "Is NRXN1 DEG-supported in IDH-mutant cycling Ephys_2?"
streamlit run app.py
```

`llm.py` is a stub until week 3 (MedGemma + Gemini on the same tool traces).
