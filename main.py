from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import sys
import logging
import re
import time

# ============================================================
# PATH SETUP
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure data file path is absolute for Render stability
DATA_PATH = os.path.join(BASE_DIR, "test_catalog.json")

SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

try:
    from retriever import IntelligentSearcher
except Exception as e:
    print("❌ Failed to import IntelligentSearcher:", e)
    sys.exit(1)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="SHL Recommender API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine: Optional[IntelligentSearcher] = None

@app.on_event("startup")
async def startup_event():
    global search_engine
    logging.info("Initializing IntelligentSearcher...")
    
    # Critical Fix: Check if data file exists before loading
    if not os.path.exists(DATA_PATH):
        logging.error(f"FATAL: {DATA_PATH} not found. Ensure it is pushed to GitHub root.")
        return

    try:
        search_engine = IntelligentSearcher()
        logging.info("IntelligentSearcher initialized successfully.")
    except Exception:
        logging.exception("Failed to initialize IntelligentSearcher")
        search_engine = None

# ============================================================
# MODELS (Strictly matching Appendix 2 & 3)
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)

class AssessmentItem(BaseModel):
    url: str
    name: str
    description: str
    duration: int
    test_type: List[str]
    remote_support: str  # Requirement: "Yes" or "No"
    adaptive_support: str # Requirement: "Yes" or "No"

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]

# ============================================================
# DATA CLEANING UTIL
# ============================================================
def clean_assessment_data(item: dict) -> dict:
    """Normalize and clean raw assessment metadata to match SHL requirements."""
    raw_desc = item.get("description", "") or ""

    # Extraction logic for SHL metadata strings
    duration_match = re.search(r"Time in minutes = (\d+)", raw_desc)
    duration = int(duration_match.group(1)) if duration_match else int(item.get("duration", 0))

    # Requirement: Standardized Yes/No strings
    remote = "Yes" if "remote" in raw_desc.lower() or item.get("remote_support") else "No"
    adaptive = "Yes" if "adaptive" in raw_desc.lower() or item.get("adaptive_support") else "No"

    return {
        "url": item.get("url", ""),
        "name": item.get("name", "Unknown Assessment"),
        "description": raw_desc.strip(),
        "duration": duration,
        "test_type": item.get("test_type", ["General"]),
        "remote_support": remote,
        "adaptive_support": adaptive
    }

# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
async def health_check():
    """Health endpoint: Returns 200 OK and healthy status."""
    if search_engine:
        return {"status": "healthy"}
    raise HTTPException(status_code=503, detail={"status": "initializing"})

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    """Returns Top 10 ranked SHL assessments."""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine initializing")

    try:
        results = search_engine.search(request.query, top_k=10)
        cleaned_results = [clean_assessment_data(item) for item in results]
        return {"recommended_assessments": cleaned_results}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")