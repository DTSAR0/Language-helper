import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

try:
    import faiss
    faiss.omp_set_num_threads(4)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(4)
except Exception:
    pass

import numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

CSV = Path("docs/entries.csv")
IDX = Path("docs/index.faiss")
PAR = Path("docs/entries.parquet")

assert CSV.exists(), f"File not found: {CSV}"

print("→ Reading CSV...")
df = pd.read_csv(CSV)
assert {"term","text"} <= set(df.columns), "CSV must have columns 'term' and 'text'"
texts = df["text"].astype(str).tolist()

print("→ Loading embedder...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print(f"→ Building embeddings for {len(texts)} rows...")
emb = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
emb = np.asarray(emb, dtype="float32")

print("→ Creating FAISS index (cosine via inner product)...")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

IDX.parent.mkdir(parents=True, exist_ok=True)
PAR.parent.mkdir(parents=True, exist_ok=True)

print(f"→ Saving index: {IDX}")
faiss.write_index(index, str(IDX))

print(f"→ Saving data: {PAR}")
df.to_parquet(PAR, index=False)

print("✅ Done:", IDX, "and", PAR)
