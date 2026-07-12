"""
embed_store.py
--------------
Embeds chunks with a local sentence-transformers model and stores them in
a FAISS index for fast top-k similarity search.

We use `all-MiniLM-L6-v2` (384-dim): it's small, fast on CPU, free (no API
key / cost per embedding call), and good enough for short policy-style
passages. Vectors are L2-normalized so that FAISS's inner-product index
(`IndexFlatIP`) behaves like cosine similarity, which gives us a similarity
score in a predictable [-1, 1] range that we can threshold for the
"I don't know" / grounded-confidence logic in rag_pipeline.py.
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self._model = None  # lazy-loaded
        self.index = None
        self.chunks = []  # parallel list of Chunk objects (or dicts)

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts):
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # -> cosine similarity via inner product
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    def build(self, chunks):
        """chunks: list of ingest.Chunk"""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vecs = self._embed(texts)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)
        return self

    def add(self, chunks):
        """Add more chunks to an already-built index (used for the
        live PDF-upload bonus feature)."""
        if self.index is None:
            return self.build(chunks)
        texts = [c.text for c in chunks]
        vecs = self._embed(texts)
        self.index.add(vecs)
        self.chunks.extend(chunks)
        return self

    def search(self, query: str, top_k: int = 4):
        """Returns a list of (chunk, similarity_score) sorted best-first."""
        if self.index is None or self.index.ntotal == 0:
            return []
        qvec = self._embed([query])
        scores, idxs = self.index.search(qvec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # ---- persistence -----------------------------------------------
    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"model_name": self.model_name, "n_chunks": len(self.chunks)}, f)

    @classmethod
    def load(cls, out_dir: str):
        with open(os.path.join(out_dir, "meta.json")) as f:
            meta = json.load(f)
        store = cls(model_name=meta["model_name"])
        store.index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "chunks.pkl"), "rb") as f:
            store.chunks = pickle.load(f)
        return store
