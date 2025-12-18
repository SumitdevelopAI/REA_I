import faiss
import pickle
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any
import logging

# ============================================================
# DYNAMIC PATH SETUP (Critical for Cloud Deployment)
# ============================================================
# This calculates the path regardless of the OS (Windows or Linux)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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
            logging.error(f"❌ Data files missing at: {PROJECT_ROOT}")
            logging.info(f"Searched for: {self.vector_db_path}")
            raise FileNotFoundError("vector_store.faiss or metadata.pkl missing from root.")

        logging.info(f"✅ Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            # Assumes your pickle file structure is {'metadata': [...]}
            data = pickle.load(f)
            self.metadata = data["metadata"] if isinstance(data, dict) else data

        logging.info(f"✅ Loading FAISS index from {self.vector_db_path}")
        self.index = faiss.read_index(self.vector_db_path)

        # Set device (Render Free Tier will use CPU automatically)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"🖥️ Using device: {self.device}")
        
        self.retriever = SentenceTransformer(RETRIEVER_MODEL_NAME, device=self.device)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME, device=self.device)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # 1. Retriever: Vector Search (FAISS)
        query_vec = self.retriever.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Retrieve 30 candidates for better re-ranking precision
        _, indices = self.index.search(query_vec, 30)

        candidates = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                item = self.metadata[idx]
                
                # Format strings for Cross-Encoder context
                t_type = item.get("test_type", [])
                t_type_str = ", ".join(t_type) if isinstance(t_type, list) else str(t_type)
                job_lvl = item.get("job_levels", "All Levels")

                rich_text = (
                    f"Title: {item['name']}\n"
                    f"Type: {t_type_str}\n"
                    f"Level: {job_lvl}\n"
                    f"Description: {item['description']}"
                )
                candidates.append({"doc": item, "text": rich_text})

        if not candidates:
            return []

        # 2. Re-ranker: Cross-Encoder (MS-MARCO) for Recall@K optimization
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["score"] = float(score)

        # Sort by re-ranker score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:top_k]

        # 3. Final Output Formatting
        results = []
        for cand in selected:
            doc = cand["doc"]
            
            # Clean duration field
            duration = doc.get("duration", 45)
            if not isinstance(duration, int):
                try:
                    duration = int(str(duration).lower().replace("mins", "").strip())
                except:
                    duration = 45

            results.append({
                "url": doc.get("url", ""),
                "name": doc.get("name", "Unknown Assessment"),
                "description": doc.get("description", "")[:600], # Trimmed for UI
                "duration": duration,
                "job_levels": doc.get("job_levels", "All Levels"),
                "test_type": doc.get("test_type", []),
                "remote_support": doc.get("remote_support", "Yes"),
                "adaptive_support": doc.get("adaptive_support", "No")
            })

        return results