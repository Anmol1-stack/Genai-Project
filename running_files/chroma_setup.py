import chromadb
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

# Point to the real ChromaDB populated by hospital_multillm_rag.py
CHROMA_DB_PATH = os.getenv("CHROMA_PATH", "/Users/Genai Project/vectordb/chromadb")
SOP_COLLECTION  = os.getenv("CHROMA_COLLECTION", "hospital_sops")

_client    = None
_embedder  = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def initialize_chromadb():
    """
    Connect to the existing ChromaDB populated by the RAG pipeline.
    No re-indexing is done — the SOP collection is already fully populated.
    """
    client = _get_client()
    try:
        collection = client.get_collection(SOP_COLLECTION)
        print(f"ChromaDB ready: '{SOP_COLLECTION}' has {collection.count()} documents.")
    except Exception as e:
        print(f"WARNING: Could not connect to ChromaDB collection '{SOP_COLLECTION}': {e}")
        print("Run 'python hospital_multillm_rag.py prepare-sop --sop-dir sop_docs' to populate it.")


def get_sop_retriever():
    return _get_client().get_collection(SOP_COLLECTION)


def query_relevant_sop(query_text: str, n_results: int = 1) -> str:
    """Return the top-matching SOP snippet for display in the dashboard."""
    try:
        collection    = get_sop_retriever()
        embedder      = _get_embedder()
        query_embedding = embedder.encode([query_text], normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )
        if results and results["documents"] and results["documents"][0]:
            return results["documents"][0][0]
    except Exception as e:
        print(f"WARNING: SOP retrieval failed: {e}")
    return "No relevant SOP found."
