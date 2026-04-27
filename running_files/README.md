# Running Files — Core System

This folder contains the **complete runnable system** for the Hospital Multi-LLM RAG Triage Pipeline.

## How to Run

```bash
# From the repo root
cd running_files
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser.

## Contents

| File/Folder | Description |
|-------------|-------------|
| `hospital_multillm_rag.py` | Main pipeline — Qwen (classification) + Mistral (description) + ChromaDB RAG |
| `main.py` | FastAPI server — REST API + static file serving |
| `database.py` | SQLAlchemy models for complaint storage |
| `schemas.py` | Pydantic request/response schemas |
| `blip_handler.py` | BLIP image captioning (Salesforce/blip-image-captioning-base) |
| `whisper_handler.py` | Whisper speech-to-text transcription |
| `chroma_setup.py` | ChromaDB collection setup utility |
| `static/` | Frontend HTML/CSS/JS — public complaint form + admin dashboard |

## Environment

Copy `.env.example` or create `.env` at the repo root with:
```
CHROMA_PATH=/absolute/path/to/vectordb/chromadb
CHROMA_COLLECTION=hospital_sops
HF_TOKEN=your_hf_token
GROQ_API_KEY=your_groq_key
MISTRAL_BACKEND=groq
```
