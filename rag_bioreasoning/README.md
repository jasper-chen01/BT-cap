# Glioma Ephys bio-reasoning

This standalone package retrieves bounded evidence from the updated CellChat, DEG, annotation, and cell-count tables, then answers a biological question with one of five modes:

- **Local MedGemma** through Ollama (the Week 3, no-per-call-cost path).
- **Mock trace client** for testing the cache/replay loop without model calls.
- **MedGemma** through an already deployed Vertex AI endpoint (primary model).
- **Gemini** through Vertex AI (comparison model).
- **None / extractive** for offline inspection of the exact evidence sent to a model.

It does not modify the existing website, upload entire CSVs, deploy cloud resources, or treat CellChat scores as experimental validation. The output is a ranked, traceable reasoning aid for choosing genes and ligand–receptor pairs for later validation.

## 1. Install

From this folder on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Install local-model and ranking dependencies for Week 3:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[local,ranking,dev]"
```

Install the optional Google Cloud clients before a live MedGemma or Gemini run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[cloud,dev]"
```

Copy `.env.example` to `.env`, then set `DATA_DIR` to the external folder containing `cellchat_ephys_plus_celltype/`, `glioma_compartment_ephys_clustering/`, and the DEG folders. Do not copy the raw scientific tables or credentials into this repository.

## 2. Verify the offline workflow

Provider status is safe to share; it prints only readiness booleans:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli providers
```

Inspect an answer and the underlying tool traces without calling a model:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli ask --provider none "What does NRXN bind in this table?"
```

Run all seven committed questions and write JSON plus Markdown reports:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli evaluate --provider none --output-dir analysis_runs/offline_smoke
```

The report stores the actual provider and model, answer, latency, required-term coverage, tool provenance, and retrieved chunk titles. A failed question is recorded without discarding the successful records.

## 3. Run Week 3 locally with Ollama

Install Ollama from [the official Windows download](https://ollama.com/download/windows), then pull a MedGemma 4B instruction build:

```powershell
ollama pull medgemma:4b-it-q4_K_M
```

The default local endpoint is `http://localhost:11434/v1`. On the 4 GB RTX 3050 test laptop, the multimodal build required CPU-only mode because its CUDA loading path could not allocate a 1.85 GB pinned host buffer. Start a second CPU-only Ollama server when needed:

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
$env:GGML_VK_VISIBLE_DEVICES="-1"
$env:OLLAMA_VULKAN="0"
$env:OLLAMA_HOST="127.0.0.1:11435"
$env:OLLAMA_CONTEXT_LENGTH="4096"
ollama serve
```

Point `.env` at that server and run the zero-cost mock loop first:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli week3-run --provider mock --cache-label mock
```

Then run local MedGemma five times over Q9-Q38. Every completed response is atomically written under `runs/`; rerunning this command skips valid cache files and only regenerates missing or invalid ones.

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli week3-run --provider ollama --cache-label medgemma
```

Rebuild all tables from cache without any model call:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli week3-replay --table --gemini-cache <week2-evaluation.json>
```

The committed `evaluation/week3_human_audit.json` separates exact required-term matching from human review of biological roles and logic. Raw model answers remain in the ignored `runs/` directory; share-safe tables are written under `week3_outputs/`.

Build the 39-feature, gene-grouped LightGBM ranking:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli rank-genes
```

The ranked unit is a CellChat gene × IDH/cell-type context. The included 33-gene labels are internal hypothesis-derived seeds, not an independently curated published gold standard.

## 4. Configure MedGemma on Vertex AI

This client expects an **instruction-tuned MedGemma model that is already deployed to a Vertex AI endpoint**. Deployment is intentionally manual because endpoint compute is billable while the model remains deployed.

Before the first live call:

1. Use the `elec-594-bt-cap` Google Cloud project, confirm that billing/project credits are active, and enable the Vertex AI API (`aiplatform.googleapis.com`).
2. In Vertex AI Model Garden, open MedGemma, accept any required model terms, choose an instruction-tuned variant, review the proposed region, machine/accelerator, quota, and hourly cost, and deploy it only after the team approves that cost.
3. Give the person or service account running this client permission to use the project and Vertex endpoint. For local development, configure Application Default Credentials or set `GOOGLE_APPLICATION_CREDENTIALS` to an approved service-account JSON stored outside the repository.
4. Put the endpoint's numeric ID and its region in `.env` as `MEDGEMMA_ENDPOINT_ID` and `MEDGEMMA_ENDPOINT_LOCATION`. Keep `MEDGEMMA_USE_DEDICATED_ENDPOINT` consistent with the endpoint type shown in Vertex AI.
5. Confirm readiness with the `providers` command. A configured status only means required settings are present; the live smoke test verifies authentication, IAM, endpoint health, and response compatibility.

Run one bounded live question:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli ask --provider medgemma "What does NRXN bind in this table?"
```

Then run the reproducible MedGemma evaluation:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli evaluate --provider medgemma --output-dir analysis_runs/medgemma_live
```

An explicit MedGemma error remains labeled as a MedGemma error. The application never silently substitutes Gemini after a failed MedGemma request.

## 5. Run the Gemini comparison

Gemini uses the same grounded prompt and question manifest, so its output is directly comparable:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli evaluate --provider gemini --output-dir analysis_runs/gemini_live
```

To evaluate both in one run:

```powershell
.\.venv\Scripts\python.exe -m ephys_rag.cli evaluate --provider medgemma --provider gemini --output-dir analysis_runs/model_comparison
```

`auto` chooses configured MedGemma first, then configured Gemini, then extractive mode. It selects once before generation and does not hide request-time failures.

## 6. Streamlit interface

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The sidebar shows non-secret configuration status and lets you choose `none`, `medgemma`, `gemini`, or `auto`. Every answer displays the actual provider, model, and elapsed time.

## 7. Scientific and security boundaries

- The models receive retrieved rows and summarized tool traces, not whole expression matrices or CSV files.
- CellChat communication probability is a model-derived score, not causal or experimental proof.
- No labeled neuron identity means the system must not claim direct neuron-to-tumor signaling from synaptic-like tumor programs.
- Candidate genes still need the right cell type/IDH DEG support and adequate group cell counts.
- Keep `.env`, service-account JSON, access tokens, raw data, model weights, and raw run caches out of version control. Reviewed, share-safe Week 3 summary tables are committed; inspect any new report before sharing.

## 8. Stop charges after testing

A self-deployed Model Garden endpoint can keep consuming billable accelerator/compute capacity even when no requests are running. After the approved test window, open **Vertex AI → Online prediction → Endpoints**, select the endpoint, and **undeploy the model**. Delete the now-empty endpoint and unused registered model only if the team no longer needs them. Confirm in the console that no deployed model remains; deleting an endpoint generally requires undeploying its model first.

Official references:

- [Vertex AI Model Garden overview](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models)
- [Online prediction and dedicated endpoints](https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions)
- [Delete an endpoint](https://cloud.google.com/vertex-ai/docs/samples/aiplatform-delete-endpoint-sample)
