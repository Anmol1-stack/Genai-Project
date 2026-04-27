# Hospital Multi-LLM RAG Triage Pipeline

An AI-powered hospital complaint triage system using a multi-LLM pipeline (fine-tuned Qwen2.5-1.5B + Mistral-7B) with ChromaDB RAG, FastAPI backend, and a web frontend supporting image and voice input.

## Repository Structure

```
.
├── running_files/          ← CORE SYSTEM — start here to run the project
│   ├── hospital_multillm_rag.py
│   ├── main.py
│   ├── database.py
│   ├── schemas.py
│   ├── blip_handler.py
│   ├── whisper_handler.py
│   ├── chroma_setup.py
│   └── static/             ← Frontend (complaint form + admin dashboard)
│
├── dataset/                ← Train/val/test splits + master JSON
├── sop_docs/               ← Standard Operating Procedure documents (RAG source)
├── vectordb/               ← ChromaDB persistent vector store
├── finetuned_model/        ← Fine-tuned adapter weights (Qwen + Mistral)
├── Image Dataset/          ← Labelled hospital complaint images
│
├── baseline_llm_comparison/  ← Zero-Shot vs Prompted baseline notebooks
│   ├── Mistral_7B_ZeroShot_vs_Prompted.ipynb
│   └── Qwen_1_5B_ZeroShot_vs_Prompted.ipynb
│
├── older_baseline_attempts/  ← Earlier experiments (NOT part of final system)
│
├── SOP_Creation_&_Ingestion_in_Chroma.ipynb  ← SOP ingestion notebook
├── mistral_7b_qlora_finetuning.ipynb         ← Mistral fine-tuning notebook
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd running_files
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000
```

## Pipeline Overview

1. **Input**: Voice text (Whisper) + Image caption (BLIP) + Direct complaint text
2. **Fusion**: Cosine similarity-based weighted fusion of voice + image signals
3. **Classification**: Fine-tuned Qwen2.5-1.5B → category, severity, department
4. **RAG Retrieval**: ChromaDB query on SOP docs + train/val corpus
5. **Description**: Fine-tuned Mistral-7B (via Groq) → formal complaint description
6. **Guardrails**: Hybrid severity floor + category keyword validation
7. **Storage**: SQLite + ChromaDB predictions collection

## Note on Older Baseline Attempts

The `older_baseline_attempts/` folder contains **earlier experimental baselines** (LLaMA zero-shot, hybrid LLaMA+Mistral pipelines) that were evaluated during Phase 2 of the project. These are **not part of the final implemented system** and are kept for reference only.
