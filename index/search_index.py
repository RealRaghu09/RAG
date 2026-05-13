import faiss
import json
import numpy as np
from collections import defaultdict
from typing import List , Dict , Any 

from Embeddings.embedder import Embedder

class FaissRetriever:

    def __init__(self, index_path: str , meta_path: str):
        self.index = load_index(index_path)
        self.metadata = load_metadata(meta_path)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index vectors ({self.index.ntotal}) != metadata rows ({len(self.metadata)}). "
                "Your mapping is inconsistent."
            )

        self.embedder = Embedder()

    def retrieve(self, query: str, top_k: int = 5):
        results = search(
            query=query,
            embedder=self.embedder,
            index=self.index,
            metadata=self.metadata,
            top_k=top_k
        )

        return [
            {
                "doc_id": r["doc_id"],
                "chunk_id": r["chunk_id"],
                "text": r["text"]
            }
            for r in results
        ]





def load_index(index_path: str) -> faiss.Index:
    """Load a FAISS index from the specified path."""
    return faiss.read_index(index_path)

def load_metadata(meta_path: str) -> List[Dict[str, Any]]:
    """Load metadata from a JSON file."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def search(
        query:str,
        embedder: Embedder,
        index: faiss.Index,
        metadata: List[Dict[str, Any]],
        top_k: int = 5,
) -> List[Dict[str,Any]]:

    q_vec = embedder.embed([query]).numpy().astype('float32') 


    scores, indices = index.search(q_vec, top_k) 


    results = []
    for rank, (score,idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue  
        item = metadata[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "doc_id": item["doc_id"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
            }
        )
    return results

def rank_documents(results):
    """
    Aggregate chunk-level results into document-level ranking.
    """
    doc_scores = defaultdict(list)

    for r in results:
        doc_scores[r["doc_id"]].append(r["score"])
    
    ranked_docs =[]
    for doc_id, scores in doc_scores.items():
        ranked_docs.append({
            "doc_id": doc_id,
            "score": max(scores), 
            "num_chunks": len(scores),
        })

    ranked_docs.sort(key=lambda x: x["score"], reverse=True)
    return ranked_docs
