import faiss
import pickle
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any


DATA_DIR = r"D:\REA"
DEFAULT_VECTOR_DB = os.path.join(DATA_DIR, "vector_store.faiss")
DEFAULT_METADATA = os.path.join(DATA_DIR, "metadata.pkl")

RETRIEVER_MODEL_NAME = "all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class IntelligentSearcher:
    def __init__(self):
        self.vector_db_path = DEFAULT_VECTOR_DB
        self.metadata_path = DEFAULT_METADATA

        if not os.path.exists(self.vector_db_path) or not os.path.exists(self.metadata_path):
            alt_vector = os.path.join(DATA_DIR, "vector_store.faiss")
            alt_meta = os.path.join(DATA_DIR, "metadata.pkl")

            if os.path.exists(alt_vector) and os.path.exists(alt_meta):
                self.vector_db_path = alt_vector
                self.metadata_path = alt_meta
            else:
                raise FileNotFoundError

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)["metadata"]

        self.index = faiss.read_index(self.vector_db_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.retriever = SentenceTransformer(RETRIEVER_MODEL_NAME, device=self.device)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME, device=self.device)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_vec = self.retriever.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        _, indices = self.index.search(query_vec, 30)

        candidates = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                item = self.metadata[idx]

                t_type = item.get("test_type", [])
                if isinstance(t_type, list):
                    t_type = ", ".join(t_type)

                job_lvl = item.get("job_levels", "All Levels")

                rich_text = (
                    f"Assessment Title: {item['name']}\n"
                    f"Category: {t_type}\n"
                    f"Target Levels: {job_lvl}\n"
                    f"Description: {item['description']}"
                )

                candidates.append({"doc": item, "text": rich_text})

        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["score"] = float(score)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:top_k]

        results = []
        for cand in selected:
            doc = cand["doc"]

            duration = doc.get("duration")
            if not isinstance(duration, int):
                try:
                    duration = int(str(duration).replace(" mins", ""))
                except:
                    duration = 45

            results.append({
                "url": doc.get("url", ""),
                "name": doc.get("name", ""),
                "adaptive_support": doc.get("adaptive_support", "No"),
                "description": doc.get("description", "")[:600],
                "duration": duration,
                "remote_support": doc.get("remote_support", "Yes"),
                "test_type": doc.get("test_type", [])
            })

        return results
