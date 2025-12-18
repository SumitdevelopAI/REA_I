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
# Critical: Ensure this file is in your GitHub root folder
DATA_PATH = os.path.join(BASE_DIR, "test_catalog.json")

SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

try:
    from retriever import IntelligentSearcher
except Exception as e:
    logging.error(f"❌ Failed to import IntelligentSearcher: {e}")
    sys.exit(1)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="SHL Assessment Recommender API", 
    version="2.2.0",
    description="Semantic search and ranking service for SHL assessments"
)

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
    
    # Check if data file exists to prevent 503 error
    if not os.path.exists(DATA_PATH):
        logging.error(f"FATAL: {DATA_PATH} not found. Ensure test_catalog.json is in root.")
        return

    try:
        search_engine = IntelligentSearcher()
        logging.info("IntelligentSearcher initialized successfully.")
    except Exception:
        logging.exception("Failed to initialize IntelligentSearcher")
        search_engine = None

# ============================================================
# MODELS (Matching OAS 3.1 & Screenshots)
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Job description or role query")

class AssessmentItem(BaseModel):
    url: str
    name: str
    description: str
    duration: int
    job_levels: str  # Added to match your OAS schema requirements
    test_type: List[str]
    remote_support: str  # "Yes" or "No"
    adaptive_support: str # "Yes" or "No"

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]

# ============================================================
# DATA CLEANING UTIL
# ============================================================
def clean_assessment_data(item: dict) -> dict:
    """Standardizes response fields to match evaluation requirements."""
    raw_desc = item.get("description", "") or ""

    # Extraction logic for metadata hidden in text
    duration_match = re.search(r"Time in minutes\s*=\s*(\d+)", raw_desc, re.I)
    duration = int(duration_match.group(1)) if duration_match else int(item.get("duration", 0))

    # Job levels extraction
    level_match = re.search(r"Job levels\s+(.*?)(?:,|$)", raw_desc, re.I)
    job_levels = level_match.group(1).strip() if level_match else item.get("job_levels", "All Levels")

    return {
        "url": item.get("url", ""),
        "name": item.get("name", "Assessment"),
        "description": raw_desc.strip(),
        "duration": duration,
        "job_levels": job_levels,
        "test_type": item.get("test_type", []),
        "remote_support": "Yes" if "remote" in raw_desc.lower() or item.get("remote_support") == "Yes" else "No",
        "adaptive_support": "Yes" if "adaptive" in raw_desc.lower() or item.get("adaptive_support") == "Yes" else "No"
    }

# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
async def health_check():
    """Returns 200 OK for automated health monitoring."""
    if search_engine:
        return {"status": "healthy"}
    # Returns 503 if engine isn't ready
    raise HTTPException(status_code=503, detail="Search engine not ready")

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    """Returns top relevant assessments based on query."""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    try:
        results = search_engine.search(request.query, top_k=10)
        return {"recommended_assessments": [clean_assessment_data(i) for i in results]}
    except Exception:
        logging.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail="Internal server error")