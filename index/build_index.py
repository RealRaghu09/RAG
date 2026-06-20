import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import faiss
import numpy as np

from Embeddings.embedder import Embedder
from chunking.Chunks import sentence_aware_chunking
from Models.models import Chunk

DEFAULT_CHUNK_SIZE = 512


def load_documents(directory_of_docs: Path) -> list[tuple[str, str]]:
    """Return list of (doc_id, text) for .txt and .md files."""
    documents: list[tuple[str, str]] = []

    for file_name in sorted(os.listdir(directory_of_docs)):
        if not (file_name.endswith(".md") or file_name.endswith(".txt")):
            continue

        entire_filepath = os.path.join(directory_of_docs, file_name)
        with open(entire_filepath, "r", encoding="utf-8") as file:
            text = file.read().strip()

        if text:
            documents.append((file_name, text))

    return documents


def build_chunks(documents: list[tuple[str, str]], chunk_size: int = DEFAULT_CHUNK_SIZE):
    """Return parallel lists of chunk strings and FAISS row metadata (order matches index rows)."""
    chunk_texts: list[str] = []
    metadata: list[dict] = []
    chunk_index = 0

    for doc_id, text in documents:
        state = Chunk(text=text, chunk_size=chunk_size)
        chunks = sentence_aware_chunking(state)

        for chunk_id, ch in enumerate(chunks):
            chunk_texts.append(ch)
            metadata.append(
                {
                    "index_id": chunk_index,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": ch,
                }
            )
            chunk_index += 1

    return chunk_texts, metadata


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def main():
    repo_root = Path(__file__).resolve().parent.parent
    corpus_dir = repo_root / "corpus"
    out_dir = repo_root / "index_store"
    out_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(corpus_dir)
    if not documents:
        raise SystemExit(f"No .txt/.md documents found under {corpus_dir}")

    chunk_texts, metadata = build_chunks(documents)
    embedder = Embedder()
    embeddings = embedder.embed(chunk_texts).numpy().astype("float32")

    index = build_faiss_index(embeddings)
    index_path = out_dir / "corpus.faiss"
    meta_path = out_dir / "metadata.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(metadata)} vectors to {index_path}")
    print(f"Wrote metadata to {meta_path}")


# build index first later query and rerank it later 
