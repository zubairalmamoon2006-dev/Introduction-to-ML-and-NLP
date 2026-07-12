"""
build_index.py
---------------
One-time (or re-run-when-docs-change) script: loads all documents from
`data/`, chunks them, embeds them, and persists a FAISS index + chunk
metadata to `index_store/` so the Streamlit app can load it instantly
without re-embedding on every launch.

Usage:
    python build_index.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingest import load_and_chunk
from embed_store import VectorStore

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
INDEX_DIR = os.path.join(HERE, "index_store")


def main():
    print(f"Loading & chunking documents from {DATA_DIR} ...")
    chunks = load_and_chunk(DATA_DIR)
    print(f"  -> {len(chunks)} chunks from "
          f"{len(set(c.doc_id for c in chunks))} documents")

    print("Embedding chunks with sentence-transformers (all-MiniLM-L6-v2) ...")
    store = VectorStore()
    store.build(chunks)

    print(f"Saving FAISS index + metadata to {INDEX_DIR} ...")
    store.save(INDEX_DIR)
    print("Done. Run `streamlit run src/app.py` to launch the assistant.")


if __name__ == "__main__":
    main()
