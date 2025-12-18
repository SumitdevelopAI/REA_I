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
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

try:
    from retriever import IntelligentSearcher
except Exception as e:
    print("❌ Failed to import IntelligentSearcher:", e)
    sys.exit(1)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("SHL-API")

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
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine: Optional[IntelligentSearcher] = None

# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
async def startup_event():
    global search_engine
    logger.info("Initializing IntelligentSearcher...")
    try:
        search_engine = IntelligentSearcher()
        logger.info("IntelligentSearcher initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize IntelligentSearcher")
        search_engine = None

# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Job description or role query")


class AssessmentItem(BaseModel):
    url: str
    name: str
    description: str
    duration: Optional[int] = None
    job_levels: str
    test_type: List[str]
    remote_support: Optional[str] = None
    adaptive_support: Optional[str] = None


class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]

# ============================================================
# DATA CLEANING UTIL
# ============================================================

def clean_assessment_data(item: dict) -> dict:
    """
    Normalize and clean raw assessment metadata
    """
    raw_desc = item.get("description", "") or ""

    # ---- Duration extraction ----
    duration = item.get("duration")
    if not duration or duration in (0, 45):
        match = re.search(r"(\\d+)\\s*mins", raw_desc, re.IGNORECASE)
        if match:
            duration = int(match.group(1))

    # ---- Job level extraction ----
    job_levels = item.get("job_levels", "All Levels")
    if job_levels.lower() in ("all levels", "not specified"):
        match = re.search(
            r"Job levels\\s+(.*?)(?:,|$)",
            raw_desc,
            re.IGNORECASE
        )
        if match:
            job_levels = match.group(1).strip()

    # ---- Description cleanup ----
    clean_desc = raw_desc
    clean_desc = re.sub(r"Product Fact Sheet.*", "", clean_desc, flags=re.IGNORECASE)
    clean_desc = re.sub(r"Test Type:.*", "", clean_desc, flags=re.IGNORECASE)

    return {
        **item,
        "description": clean_desc.strip(),
        "duration": duration,
        "job_levels": job_levels
    }

# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
async def health_check():
    """
    Health endpoint for frontend & monitoring
    """
    return {
        "status": "healthy" if search_engine else "initializing"
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    """
    Return ranked SHL assessments for a job query
    """
    if not search_engine:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready"
        )

    start_time = time.time()

    try:
        results = search_engine.search(
            request.query,
            top_k=10
        )

        cleaned_results = [
            clean_assessment_data(item)
            for item in results
        ]

        logger.info(
            "Query='%s' | Results=%d | Time=%.2fs",
            request.query,
            len(cleaned_results),
            time.time() - start_time
        )

        return {
            "recommended_assessments": cleaned_results
        }

    except Exception:
        logger.exception("Recommendation failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

# ============================================================
# LOCAL RUN (DEV ONLY)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
