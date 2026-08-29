"""
Step 12: FastAPI backend.

Wraps the LangGraph agent from agent.py in a REST API. This is what
turns a command-line script into something a real frontend (or a
recruiter clicking a live demo link) can actually use.

Run locally:
    uvicorn src.api:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/match \
      -H "Content-Type: application/json" \
      -d '{"query": "62 year old female with type 2 diabetes and heart disease, based in Manchester"}'

Or just open http://localhost:8000/docs for interactive API docs (FastAPI
generates this automatically — good to mention in your README/demo).
"""

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import build_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medtrial-api")

app = FastAPI(
    title="MedTrial-Match AI",
    description="Agentic RAG system matching patients to relevant clinical trials.",
    version="1.0.0",
)

# Allows a Streamlit app (or any frontend) running on a different port/domain
# to call this API from the browser. Fine to leave open for a portfolio demo;
# you'd restrict this to specific origins in a real production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the agent graph once at startup, not per-request — it's cheap to
# build but there's no reason to redo it on every call.
_agent = build_agent()


class MatchRequest(BaseModel):
    query: str


class MatchResultItem(BaseModel):
    nct_id: str
    title: str
    eligible: str
    confidence: float
    reasoning: str
    url: str


class MatchResponse(BaseModel):
    query: str
    patient_profile: dict
    results: list[MatchResultItem]
    result_count: int
    latency_seconds: float


@app.get("/")
def root():
    return {"status": "ok", "message": "MedTrial-Match AI is running. See /docs for usage."}


@app.get("/health")
def health():
    """Simple endpoint for deployment platforms (Render/Railway) to check the service is alive."""
    return {"status": "healthy"}


@app.post("/match", response_model=MatchResponse)
def match_trials(request: MatchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start = time.time()
    logger.info(f"Received query: {request.query}")

    initial_state = {
        "raw_query": request.query,
        "patient": None,
        "candidates": [],
        "filtered": [],
        "matches": [],
        "final_results": [],
    }

    try:
        result = _agent.invoke(initial_state)
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    latency = time.time() - start
    logger.info(f"Completed in {latency:.2f}s, {len(result['final_results'])} results")

    return MatchResponse(
        query=request.query,
        patient_profile=result["patient"] or {},
        results=result["final_results"],
        result_count=len(result["final_results"]),
        latency_seconds=round(latency, 2),
    )
