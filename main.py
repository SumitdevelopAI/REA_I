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
# PATH SETUP (Resilient for Render/Linux)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "test_catalog.json")

SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SHL-API")

try:
    from retriever import IntelligentSearcher
except Exception as e:
    logger.error(f"❌ Failed to import IntelligentSearcher: {e}")
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
    logger.info("🚀 Starting up IntelligentSearcher...")
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"FATAL: {DATA_PATH} not found. Ensure test_catalog.json is in GitHub root.")
        return

    try:
        search_engine = IntelligentSearcher()
        logger.info("✅ IntelligentSearcher initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {str(e)}")
        search_engine = None

# ============================================================
# MODELS (Strictly matching Appendix 2 & OAS 3.1) 
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Job description or role query")

class AssessmentItem(BaseModel):
    url: str               # 
    name: str              # 
    adaptive_support: str  #  - Either "Yes" or "No"
    description: str       # 
    duration: int          #  - Integer
    remote_support: str    #  - Either "Yes" or "No"
    test_type: List[str]   #  - Array of Strings

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]

# ============================================================
# DATA CLEANING UTIL
# ============================================================
def clean_assessment_data(item: dict) -> dict:
    """Standardizes response fields to match SHL requirements exactly."""
    raw_desc = item.get("description", "") or ""

    # Extract duration as Integer 
    duration_match = re.search(r"Time in minutes\s*=\s*(\d+)", raw_desc, re.I)
    duration = int(duration_match.group(1)) if duration_match else int(item.get("duration", 45))

    # Standardize Yes/No strings 
    def to_yes_no(val):
        if str(val).lower() in ['yes', 'true', '1']: return "Yes"
        return "No"

    return {
        "url": item.get("url", ""),
        "name": item.get("name", "Assessment"),
        "adaptive_support": to_yes_no(item.get("adaptive_support", "No")),
        "description": raw_desc.strip(),
        "duration": duration,
        "remote_support": to_yes_no(item.get("remote_support", "Yes")),
        "test_type": item.get("test_type", ["Knowledge & Skills"])
    }

# ============================================================
# ROUTES [cite: 154]
# ============================================================
@app.get("/health") # [cite: 155]
async def health_check():
    """Health check returns status: healthy """
    if search_engine:
        return {"status": "healthy"}
    raise HTTPException(status_code=503, detail="Search engine not ready")

@app.post("/recommend", response_model=RecommendationResponse) # [cite: 163]
async def recommend_assessments(request: QueryRequest):
    """Returns ranked list of 1 to 10 assessments [cite: 163]"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine initializing")

    try:
        # Requesting top_k results between 5 and 10 [cite: 45]
        results = search_engine.search(request.query, top_k=10)
        return {"recommended_assessments": [clean_assessment_data(i) for i in results]}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")