from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import sys
import os
import logging
import re


current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

try:
    from retriever import IntelligentSearcher
except ImportError:
    sys.exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="SHL Assessment Recommender", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = None


@app.on_event("startup")
async def initialize_system():
    global search_engine
    search_engine = IntelligentSearcher()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)


class AssessmentItem(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]
    job_levels: str


class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]


def clean_assessment_data(item: dict) -> dict:
    raw_desc = item.get("description", "")

    duration = item.get("duration")
    if not duration or duration == 45 or duration == 0:
        match = re.search(r"Approximate Completion Time.*?=\s*(\d+)", raw_desc, re.IGNORECASE)
        if match:
            duration = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s*mins", raw_desc, re.IGNORECASE)
            if match:
                duration = int(match.group(1))

    job_levels = item.get("job_levels", "All Levels")
    if job_levels == "All Levels" or "Not Specified" in job_levels:
        match = re.search(
            r"Job levels\s+(.*?)(?:, Languages|, Assessment|Test Type|$)",
            raw_desc,
            re.IGNORECASE,
        )
        if match:
            job_levels = match.group(1).strip()

    clean_desc = raw_desc
    clean_desc = re.sub(
        r"^Home\s+Products\s+Product\s+Catalog.*?(?=\bDescription\b|\bMulti-choice\b|\bSimulation\b)",
        "",
        clean_desc,
        flags=re.IGNORECASE,
    )
    clean_desc = re.sub(r"^Description\s+", "", clean_desc, flags=re.IGNORECASE)
    clean_desc = re.sub(r"Product Fact Sheet.*", "", clean_desc, flags=re.IGNORECASE)
    clean_desc = re.sub(r"Test Type:.*", "", clean_desc, flags=re.IGNORECASE)

    return {
        **item,
        "description": clean_desc.strip(),
        "duration": duration,
        "job_levels": job_levels,
    }


@app.get("/health")
async def health_check():
    if search_engine is None:
        return {"status": "initializing"}
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    if not search_engine:
        raise HTTPException(status_code=503, detail="System loading")

    try:
        results = search_engine.search(request.query, top_k=10)
        cleaned = [clean_assessment_data(item) for item in results]
        return {"recommended_assessments": cleaned}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
