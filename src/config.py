import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data Paths
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "shl_individual_test_catalog.json")
DATA_ARTIFACTS = os.path.join(PROJECT_ROOT, "data", "artifacts")
VECTOR_DB = os.path.join(DATA_ARTIFACTS, "D:\REA\data\shl_vector_store.faiss")
METADATA = os.path.join(DATA_ARTIFACTS, "D:\REA\data\shl_metadata.pkl")

# Model Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Retriever (Finds the documents)
RETRIEVER_MODEL = 'all-mpnet-base-v2'

# 2. Reranker (Sorts them by relevance)
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# 3. Generator (The AI that answers)
# Use "google/flan-t5-large" for good answers on 8GB VRAM
# Use "google/flan-t5-base" if you have <4GB VRAM
LLM_MODEL = 'google/flan-t5-large'