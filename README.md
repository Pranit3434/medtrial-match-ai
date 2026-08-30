![Tests](https://github.com/Pranit3434/medtrial-match-ai/actions/workflows/test.yml/badge.svg)

# MedTrial-Match AI

An agentic RAG system that matches patient profiles to relevant clinical
trials using multi-step reasoning, real-time API retrieval, and
explainable LLM-based ranking.

**Status:** 🚧 In progress — Phase 1 (data pipeline) complete.

## Architecture

```
User Query (patient profile)
        │
        ▼
[1] Query Parser (LLM) — extract structured fields
        │
        ▼
[2] Retrieval Node — ClinicalTrials.gov API + vector search
        │
        ▼
[3] Filter Node — hard filters (age, location, status)
        │
        ▼
[4] Reasoning Node (LLM) — reads eligibility criteria, reasons about fit
        │
        ▼
[5] Ranking + Explanation Node — top-N matches with plain-English reasoning
        │
        ▼
Response (Streamlit UI / API JSON)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in your ANTHROPIC_API_KEY
```

## Usage so far

**1. Fetch trial data from ClinicalTrials.gov:**

```bash
python src/fetch_trials.py --condition "type 2 diabetes" --max-results 100
```

This saves flattened trial records to `data/raw_trials.json`.

**2. Ingest into the local vector store:**

```bash
python src/ingest.py --input data/raw_trials.json --query "62 year old with type 2 diabetes and heart disease"
```

This embeds each trial and stores it in ChromaDB (`data/chroma_db/`),
then runs a sample semantic search so you can sanity-check retrieval
quality before building agent logic on top of it.

## Roadmap

- [x] Explore ClinicalTrials.gov API, build fetch script
- [x] Build ingestion pipeline (embed + store in ChromaDB)
- [ ] Build LangGraph agent (parser → retrieval → filter → reasoning → ranking nodes)
- [ ] FastAPI backend (`/match` endpoint)
- [ ] Streamlit demo UI
- [ ] Evaluation harness (precision@5, LLM-as-judge)
- [ ] Dockerize + deploy (Render/Railway)
- [ ] Demo video + portfolio writeup

## Tech Stack

LangGraph · Anthropic Claude API · FastAPI · ChromaDB · Streamlit · Docker
🔗 Live demo: https://medtrial-match-ai-1.onrender.com
🔗 API: https://medtrial-match-ai.onrender.com

git add README.md
git commit -m "Add CI status badge to README"
git push
