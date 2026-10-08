# Week 2 — MedGemma + Gemini on the same traces

Same repo. Same tables. Same tools. Last week the agent only retrieved. This week it **writes**.

## What they do

1. Open `rag_bioreasoning/src/ephys_rag/llm.py`.
   `generate_answer()` is a stub. Fill it in.
2. Reuse Challenge B’s **LLM client** (the `chat.completions` call), not Challenge B’s data.
3. Send the **same** tool traces to:
   - MedGemma 
   - Gemini Flash 
   - or some other knowledge tools
4.  Run [questions.md](questions.md) **Q9–Q32**. Q1–Q8 can be re-run as a check. Q33–Q38 are stretch.

## What they turn in

For each of Q9–Q32:

| Question | Tool that fired | Required term in traces? | MedGemma answer | Gemini answer | Either model invented a pair/DEG? |
|---|---|---|---|---|---|

## Still true

Empty lookups are allowed (Q36: T cells do not send NRXN).  
Neurons are not a labeled identity unless `annotation_lookup` says so.  


## SOME REFERENCES 
https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1267s

We can build a similar framework but for cancer neuroscience!
https://www.cell.com/cell/abstract/S0092-8674(26)00651-3
