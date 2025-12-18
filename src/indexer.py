import json
import pickle
import os
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict

DATA_DIR = r"D:\REA\data"
INPUT_FILE = os.path.join(DATA_DIR, "test_catalog.json")
VECTOR_DB_FILE = os.path.join(DATA_DIR, "vector_store.faiss")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.pkl")

MODEL_NAME = "all-mpnet-base-v2"


def create_rich_context(item: Dict) -> str:
    name = item.get("name", "Unknown Assessment")
    description = item.get("description", "")
    job_levels = item.get("job_levels", "All Levels")

    test_types = item.get("test_type", [])
    if isinstance(test_types, list):
        test_types = ", ".join(test_types)

    return (
        f"Assessment Name: {name}. "
        f"Category: {test_types}. "
        f"Target Level: {job_levels}. "
        f"Description: {description}"
    )


def main():
    if not os.path.exists(INPUT_FILE):
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = []
    metadata = []

    for item in raw_data:
        documents.append(create_rich_context(item))
        metadata.append({
            "name": item.get("name"),
            "url": item.get("url"),
            "description": item.get("description"),
            "test_type": item.get("test_type", []),
            "job_levels": item.get("job_levels"),
            "adaptive_support": item.get("adaptive_support", "No"),
            "remote_support": item.get("remote_support", "Yes"),
            "duration": item.get("duration", "N/A")
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    embeddings = model.encode(
        documents,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, VECTOR_DB_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump({"metadata": metadata}, f)


if __name__ == "__main__":
    main()
