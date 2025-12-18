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
# These files MUST be in your GitHub root folder
DATA_PATH = os.path.join(BASE_DIR, "test_catalog.json")

# Add 'src' to path so we can import retriever
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Configure logging to show in Render console
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
    
    # Check if data file exists to prevent 503 error
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
# MODELS
# ============================================================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Job description or role query")

class AssessmentItem(BaseModel):
    url: str
    name: str
    description: str
    duration: int
    job_levels: str
    test_type: List[str]
    remote_support: str
    adaptive_support: str

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]

# ============================================================
# DATA CLEANING UTIL
# ============================================================
def clean_assessment_data(item: dict) -> dict:
    raw_desc = item.get("description", "") or ""

    # Metadata extraction
    duration_match = re.search(r"Time in minutes\s*=\s*(\d+)", raw_desc, re.I)
    duration = int(duration_match.group(1)) if duration_match else int(item.get("duration", 45))

    level_match = re.search(r"Job levels\s+(.*?)(?:,|$)", raw_desc, re.I)
    job_levels = level_match.group(1).strip() if level_match else item.get("job_levels", "All Levels")

    return {
        "url": item.get("url", ""),
        "name": item.get("name", "Assessment"),
        "description": raw_desc.strip(),
        "duration": duration,
        "job_levels": job_levels,
        "test_type": item.get("test_type", ["General"]),
        "remote_support": "Yes" if "remote" in raw_desc.lower() or item.get("remote_support") == "Yes" else "No",
        "adaptive_support": "Yes" if "adaptive" in raw_desc.lower() or item.get("adaptive_support") == "Yes" else "No"
    }

# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
async def health_check():
    if search_engine:
        return {"status": "healthy"}
    raise HTTPException(status_code=503, detail="Search engine not ready")

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine initializing")

    try:
        results = search_engine.search(request.query, top_k=10)
        return {"recommended_assessments": [clean_assessment_data(i) for i in results]}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")