import argparse
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()


@dataclass
class Config:
    raw_dir: Path = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
    index_dir: Path = Path(os.getenv("VECTORDB_DIR", "vectordb/chromadb"))
    collection_name: str = os.getenv("COLLECTION_NAME", "healthcare_docs")
    sqlite_path: Path = Path(os.getenv("SQLITE_PATH", "rag_runs.db"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE", 512))
    overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP", 50))
    token_to_char: int = int(os.getenv("TOKEN_TO_CHAR", 4))
    top_k: int = int(os.getenv("TOP_K_RESULTS", 5))
    llm_backend: str = os.getenv("LLM_BACKEND", "groq")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    base_model_id: str = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    adapter_dir: str = os.getenv("ADAPTER_DIR", "")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 512))


def clean_text(text: str) -> str:
    text = re.sub(r"[-–]\s*\d+\s*[-–]", "", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    lines = text.split("\n")
    kept: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append("")
        elif len(s) > 3 and re.search(r"[a-zA-Z]", s):
            kept.append(line)
    return "\n".join(kept).strip()


def load_pdf(filepath: Path) -> str:
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text += page_text + "\n"
    return text


def load_text_file(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="ignore")


def ingest_documents(raw_dir: Path) -> List[Dict[str, Any]]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW_DATA_DIR not found: {raw_dir}")

    supported = {".pdf", ".txt", ".md"}
    files = [f for f in sorted(raw_dir.iterdir()) if f.is_file() and f.suffix.lower() in supported]
    if not files:
        raise FileNotFoundError(f"No .pdf/.txt/.md files found in {raw_dir}")

    docs: List[Dict[str, Any]] = []
    for filepath in files:
        if filepath.suffix.lower() == ".pdf":
            raw = load_pdf(filepath)
            source_type = "pdf"
        else:
            raw = load_text_file(filepath)
            source_type = "text"

        raw = raw.strip()
        if not raw:
            continue

        cleaned = clean_text(raw)
        if not cleaned:
            continue

        docs.append(
            {
                "filename": filepath.name,
                "source_type": source_type,
                "text": cleaned,
                "raw_char_count": len(raw),
                "clean_char_count": len(cleaned),
            }
        )
    if not docs:
        raise RuntimeError("Documents were found, but all became empty after cleaning.")
    return docs


def chunk_text(text: str, filename: str, chunk_size_tokens: int, overlap_tokens: int, token_to_char: int) -> List[Dict[str, Any]]:
    chunk_chars = chunk_size_tokens * token_to_char
    overlap_chars = overlap_tokens * token_to_char
    step = max(1, chunk_chars - overlap_chars)

    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_chars, n)
        piece = text[start:end].strip()
        if len(piece) > 50:
            chunks.append(
                {
                    "chunk_id": f"{filename}_chunk_{idx}",
                    "source": filename,
                    "chunk_index": idx,
                    "start_char": start,
                    "end_char": end,
                    "text": piece,
                    "char_length": len(piece),
                }
            )
            idx += 1
        if end == n:
            break
        start += step
    return chunks


def make_chunks(docs: List[Dict[str, Any]], chunk_size_tokens: int, overlap_tokens: int, token_to_char: int) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for d in docs:
        stem = Path(d["filename"]).stem
        all_chunks.extend(chunk_text(d["text"], stem, chunk_size_tokens, overlap_tokens, token_to_char))
    if not all_chunks:
        raise RuntimeError("Chunking produced zero chunks. Reduce chunk size or check input docs.")
    return all_chunks


def build_index(cfg: Config) -> Dict[str, Any]:
    docs = ingest_documents(cfg.raw_dir)
    chunks = make_chunks(docs, cfg.chunk_size_tokens, cfg.overlap_tokens, cfg.token_to_char)

    embedder = SentenceTransformer(cfg.embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.index_dir))

    try:
        client.delete_collection(cfg.collection_name)
    except Exception:
        pass

    coll = client.create_collection(name=cfg.collection_name, metadata={"hnsw:space": "cosine"})

    batch = 100
    ids = [c["chunk_id"] for c in chunks]
    metas = [
        {
            "source": c["source"],
            "chunk_index": int(c["chunk_index"]),
            "start_char": int(c["start_char"]),
            "end_char": int(c["end_char"]),
        }
        for c in chunks
    ]
    emb_list = embeddings.tolist()
    for i in range(0, len(chunks), batch):
        coll.add(
            ids=ids[i : i + batch],
            embeddings=emb_list[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metas[i : i + batch],
        )

    summary = {
        "documents": len(docs),
        "chunks": len(chunks),
        "collection_name": cfg.collection_name,
        "embedding_model": cfg.embedding_model,
        "index_dir": str(cfg.index_dir),
    }
    return summary


def ensure_sqlite(sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                llm_model TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                retrieval_time REAL NOT NULL,
                generation_time REAL NOT NULL,
                total_time REAL NOT NULL,
                sources_json TEXT NOT NULL,
                prompt_chars INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context_parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        context_parts.append(f"[Source {i}: {c['source']}]\n{c['text']}")
    context = "\n\n".join(context_parts)
    return (
        "You are a healthcare document assistant.\n"
        "Answer ONLY from the provided context.\n"
        "If the answer is not in context, say exactly:\n"
        "\"I cannot find this information in the provided documents.\"\n"
        "Keep answer factual, concise, and cite [Source X].\n\n"
        "=== CONTEXT ===\n"
        f"{context}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def retrieve_chunks(cfg: Config, question: str) -> List[Dict[str, Any]]:
    embedder = SentenceTransformer(cfg.embedding_model)
    q_emb = embedder.encode(question, normalize_embeddings=True).astype(np.float32).tolist()

    client = chromadb.PersistentClient(path=str(cfg.index_dir))
    coll = client.get_collection(cfg.collection_name)
    out = coll.query(query_embeddings=[q_emb], n_results=cfg.top_k, include=["documents", "metadatas", "distances"])

    results: List[Dict[str, Any]] = []
    for i, (doc, meta, dist) in enumerate(zip(out["documents"][0], out["metadatas"][0], out["distances"][0])):
        results.append(
            {
                "chunk_id": out["ids"][0][i],
                "source": meta.get("source", "unknown"),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": float(round(1 - float(dist), 6)),
                "text": doc,
            }
        )
    return results


def call_groq(cfg: Config, prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


class LocalFineTunedRunner:
    """
    Loads base model + LoRA adapter for local inference.
    Adapter path should be what you saved after QLoRA training.
    """
    def __init__(self, cfg: Config):
        if not cfg.adapter_dir:
            raise ValueError("adapter_dir is required for llm_backend=local_peft")
        if not Path(cfg.adapter_dir).exists():
            raise FileNotFoundError(f"Adapter directory not found: {cfg.adapter_dir}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.cfg = cfg

        tok_src = cfg.adapter_dir if Path(cfg.adapter_dir).exists() else cfg.base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_id,
                trust_remote_code=True,
            )

        self.model = PeftModel.from_pretrained(base_model, cfg.adapter_dir)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.cfg.temperature > 0
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


_LOCAL_RUNNER: LocalFineTunedRunner | None = None


def llm_label(cfg: Config) -> str:
    if cfg.llm_backend == "groq":
        return f"groq:{cfg.llm_model}"
    return f"local_peft:{cfg.base_model_id}+{cfg.adapter_dir}"


def call_llm(cfg: Config, prompt: str) -> str:
    global _LOCAL_RUNNER

    if cfg.llm_backend == "groq":
        return call_groq(cfg, prompt)

    if cfg.llm_backend == "local_peft":
        if _LOCAL_RUNNER is None:
            _LOCAL_RUNNER = LocalFineTunedRunner(cfg)
        return _LOCAL_RUNNER.generate(prompt)

    raise ValueError(f"Unsupported llm_backend: {cfg.llm_backend}")


def save_run(cfg: Config, question: str, answer: str, sources: List[Dict[str, Any]], retrieval_time: float, generation_time: float, prompt_chars: int) -> int:
    total = retrieval_time + generation_time
    ensure_sqlite(cfg.sqlite_path)
    conn = sqlite3.connect(cfg.sqlite_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO rag_runs (
                ts_utc, question, answer, llm_model, embedding_model, collection_name, top_k,
                retrieval_time, generation_time, total_time, sources_json, prompt_chars
            )
            VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                answer,
                llm_label(cfg),
                cfg.embedding_model,
                cfg.collection_name,
                cfg.top_k,
                retrieval_time,
                generation_time,
                total,
                json.dumps(sources, ensure_ascii=False),
                prompt_chars,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def ask(cfg: Config, question: str) -> Dict[str, Any]:
    t0 = time.time()
    chunks = retrieve_chunks(cfg, question)
    retrieval_time = time.time() - t0

    prompt = build_prompt(question, chunks)

    t1 = time.time()
    answer = call_llm(cfg, prompt)
    generation_time = time.time() - t1

    row_id = save_run(
        cfg=cfg,
        question=question,
        answer=answer,
        sources=[
            {
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "score": c["score"],
            }
            for c in chunks
        ],
        retrieval_time=round(retrieval_time, 4),
        generation_time=round(generation_time, 4),
        prompt_chars=len(prompt),
    )

    return {
        "row_id": row_id,
        "question": question,
        "answer": answer,
        "retrieved_chunks": chunks,
        "retrieval_time": round(retrieval_time, 4),
        "generation_time": round(generation_time, 4),
        "total_time": round(retrieval_time + generation_time, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored RAG + SQLite pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build/update Chroma index from raw docs")
    p_build.add_argument("--raw-dir", default=None, help="Folder with .pdf/.txt/.md")
    p_build.add_argument("--index-dir", default=None, help="Persistent Chroma folder")
    p_build.add_argument("--collection", default=None, help="Chroma collection name")
    p_build.add_argument("--embedding-model", default=None, help="SentenceTransformer model id")
    p_build.add_argument("--chunk-size", type=int, default=None, help="Chunk size in approx tokens")
    p_build.add_argument("--overlap", type=int, default=None, help="Chunk overlap in approx tokens")

    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("--question", required=True, help="Question to ask")
    p_ask.add_argument("--index-dir", default=None)
    p_ask.add_argument("--collection", default=None)
    p_ask.add_argument("--embedding-model", default=None)
    p_ask.add_argument("--llm-backend", choices=["groq", "local_peft"], default=None)
    p_ask.add_argument("--llm-model", default=None)
    p_ask.add_argument("--base-model-id", default=None)
    p_ask.add_argument("--adapter-dir", default=None)
    p_ask.add_argument("--sqlite-path", default=None)
    p_ask.add_argument("--top-k", type=int, default=None)

    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if getattr(args, "raw_dir", None):
        cfg.raw_dir = Path(args.raw_dir)
    if getattr(args, "index_dir", None):
        cfg.index_dir = Path(args.index_dir)
    if getattr(args, "collection", None):
        cfg.collection_name = args.collection
    if getattr(args, "embedding_model", None):
        cfg.embedding_model = args.embedding_model
    if getattr(args, "chunk_size", None):
        cfg.chunk_size_tokens = int(args.chunk_size)
    if getattr(args, "overlap", None):
        cfg.overlap_tokens = int(args.overlap)
    if getattr(args, "llm_backend", None):
        cfg.llm_backend = args.llm_backend
    if getattr(args, "llm_model", None):
        cfg.llm_model = args.llm_model
    if getattr(args, "base_model_id", None):
        cfg.base_model_id = args.base_model_id
    if getattr(args, "adapter_dir", None):
        cfg.adapter_dir = args.adapter_dir
    if getattr(args, "sqlite_path", None):
        cfg.sqlite_path = Path(args.sqlite_path)
    if getattr(args, "top_k", None):
        cfg.top_k = int(args.top_k)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)

    if args.cmd == "build":
        summary = build_index(cfg)
        print(json.dumps({"status": "ok", "action": "build", **summary}, indent=2))
        return

    if args.cmd == "ask":
        result = ask(cfg, args.question)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
