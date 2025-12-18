import faiss
import pickle
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any
import logging

# ============================================================
# DYNAMIC PATH SETUP (Works on Windows & Render)
# ============================================================
# Get the absolute path of the directory where retriever.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Move up one level to the project root where the data files are stored
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DEFAULT_VECTOR_DB = os.path.join(PROJECT_ROOT, "vector_store.faiss")
DEFAULT_METADATA = os.path.join(PROJECT_ROOT, "metadata.pkl")

RETRIEVER_MODEL_NAME = "all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class IntelligentSearcher:
    def __init__(self):
        self.vector_db_path = DEFAULT_VECTOR_DB
        self.metadata_path = DEFAULT_METADATA

        # Robust check for data files in the root
        if not os.path.exists(self.vector_db_path) or not os.path.exists(self.metadata_path):
            logging.error(f"Data files not found in project root: {PROJECT_ROOT}")
            raise FileNotFoundError("vector_store.faiss or metadata.pkl missing from root.")

        logging.info(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)["metadata"]

        logging.info(f"Loading FAISS index from {self.vector_db_path}")
        self.index = faiss.read_index(self.vector_db_path)

        # Set device (Render Free Tier will use CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.retriever = SentenceTransformer(RETRIEVER_MODEL_NAME, device=self.device)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME, device=self.device)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_vec = self.retriever.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Initial broad retrieval (Top 30 candidates)
        _, indices = self.index.search(query_vec, 30)

        candidates = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                item = self.metadata[idx]
                t_type = item.get("test_type", [])
                if isinstance(t_type, list):
                    t_type = ", ".join(t_type)

                job_lvl = item.get("job_levels", "All Levels")

                # Rich context for re-ranking
                rich_text = (
                    f"Assessment Title: {item['name']}\n"
                    f"Category: {t_type}\n"
                    f"Target Levels: {job_lvl}\n"
                    f"Description: {item['description']}"
                )
                candidates.append({"doc": item, "text": rich_text})

        if not candidates:
            return []

        # Cross-Encoder Re-ranking for Precision
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["score"] = float(score)

        # Sort and return Top K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:top_k]

        results = []
        for cand in selected:
            doc = cand["doc"]
            # Ensure duration is a clean integer
            duration = doc.get("duration", 45)
            if not isinstance(duration, int):
                try:
                    duration = int(str(duration).replace(" mins", "").strip())
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