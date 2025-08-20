# app/core/rag.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import os, numpy as np, pandas as pd, faiss
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
from app.core.llm import call_llm

# Configure multiprocessing to avoid issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes the warning
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp"  # Use temp directory
os.environ["HF_HOME"] = "/tmp"  # Use temp directory for HuggingFace cache
# Try to set multiprocessing start method, but don't fail if already set
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

PATH_PAR = Path("docs/entries.parquet")
PATH_IDX = Path("docs/index.faiss")

@lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    if not PATH_PAR.exists():
        raise FileNotFoundError(f"Missing {PATH_PAR}")
    return pd.read_parquet(PATH_PAR)

@lru_cache(maxsize=1)
def _load_index() -> faiss.Index:
    if not PATH_IDX.exists():
        raise FileNotFoundError(f"Missing {PATH_IDX}")
    return faiss.read_index(str(PATH_IDX))

@lru_cache(maxsize=1)
def _load_embedder() -> SentenceTransformer:
    # Force CPU to avoid irritating MPS/Metal on 8 GB
    # Also disable multiprocessing to avoid segmentation faults
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
        device="cpu",
        cache_folder=None  # Disable caching to avoid multiprocessing issues
    )

def retrieve(query: str, k: int = 4):
    """Top-k: first exact match (term==query), then FAISS."""
    df = _load_df()
    index = _load_index()
    emb = _load_embedder()

    # 1) exact match
    exact_idx = df.index[df["term"].str.casefold() == query.strip().casefold()].tolist()

    # 2) semantic search
    qv = emb.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, max(k, 4))
    faiss_idx = [i for i in I[0].tolist() if i not in exact_idx]

    # 3) mix: exact first
    idxs = (exact_idx + faiss_idx)[:k]
    hits = df.iloc[idxs].copy()
    # "high" scores for exact
    scores = [1.0]*len(exact_idx) + D[0].tolist()
    hits["__score"] = scores[:len(hits)]
    return hits, hits["__score"].tolist()

def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit("\n",1)[0] + "…"

def build_prompt_def(term: str, context: str) -> str:
    # Prompt for "reference", without translation
    return f"""
You are a lexicographer. Use ONLY the CONTEXT (dictionary snippets) to answer about the exact word "{term}".
If multiple senses exist, pick the main/common sense. Do NOT translate; do not switch to synonyms.

Return concise markdown with:
- **Definition** (1–2 sentences)
- **Part of speech** (if obvious)
- **2 common collocations**
- **2 short example sentences** (you may reuse from context or write minimal natural ones)

CONTEXT:
{context}
""".strip()

def ask_with_rag_def(term: str, k: int = 4, model: str = "qwen2.5:3b-instruct",
                     max_context_chars: int = 1200, llm_options: dict | None = None):
    try:
        hits, _ = retrieve(term, k=k)
        parts = [f"TERM: {row['term']}\nTEXT:\n{row['text']}" for _, row in hits.iterrows()]
        ctx = _clip("\n\n---\n\n".join(parts), max_context_chars)
        prompt = build_prompt_def(term, ctx)
        return call_llm(prompt, model=model, options=llm_options)
    except Exception as e:
        return f"❌ Error retrieving RAG information: {e}\n\nFalling back to simple LLM response..."
