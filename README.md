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
│   ├── blip_h andler.py
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
