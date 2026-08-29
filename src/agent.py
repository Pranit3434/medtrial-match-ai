"""
Step 5-11: The LangGraph agent itself.

This wires together 5 nodes into a stateful graph:
  1. parse_query      — LLM extracts structured PatientProfile from free text
  2. retrieve_trials   — vector search over ChromaDB for candidate trials
  3. filter_trials      — rule-based filtering (age, status) — NO LLM here
  4. reason_eligibility — LLM reads each candidate's eligibility criteria
  5. rank_results        — sort by confidence, format final output

Run a quick manual test:
    python src/agent.py --query "62 year old female with type 2 diabetes and heart disease, based in Manchester"
"""

import argparse
import json
import os
from typing import TypedDict, Optional

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from models import PatientProfile, TrialCandidate, TrialMatch

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = "clinical_trials"
# Groq's free tier — llama-3.3-70b is strong for this kind of structured
# reasoning task and comfortably fast enough for a portfolio demo.
LLM_MODEL = "openai/gpt-oss-120b"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(prompt: str, max_tokens: int) -> str:
    """
    Thin wrapper so the rest of the file doesn't need to know which
    provider is behind it — swap this one function if you ever want to
    try Anthropic, OpenAI, or a local model instead.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Agent state — this is what flows through every node in the graph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    raw_query: str
    patient: Optional[dict]          # PatientProfile as dict
    candidates: list                  # list of TrialCandidate dicts
    filtered: list                    # list of TrialCandidate dicts, post-filter
    matches: list                     # list of TrialMatch dicts, post-reasoning
    final_results: list                # ranked, final output


# ---------------------------------------------------------------------------
# Node 1: Query Parser
# ---------------------------------------------------------------------------
def parse_query(state: AgentState) -> AgentState:
    prompt = f"""Extract structured patient information from this query.
Return ONLY valid JSON matching this schema, nothing else:
{{"condition": "...", "age": <int or null>, "sex": "MALE"/"FEMALE"/null, "location": "..." or null, "additional_notes": "..." or null}}

Query: "{state['raw_query']}"
"""
    text = call_llm(prompt, max_tokens=300)
    # strip markdown code fences if the model adds them
    text = text.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(text)
    patient = PatientProfile(**parsed)
    print(f"[parse_query] Extracted profile: {patient.model_dump()}")

    state["patient"] = patient.model_dump()
    return state


# ---------------------------------------------------------------------------
# Node 2: Retrieval
# ---------------------------------------------------------------------------
def retrieve_trials(state: AgentState) -> AgentState:
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    patient = state["patient"]
    search_text = f"{patient['condition']} {patient.get('additional_notes') or ''}"

    results = collection.query(query_texts=[search_text], n_results=15)

    candidates = []
    for doc_id, doc_text, meta in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0]
    ):
        candidates.append(
            TrialCandidate(
                nct_id=doc_id,
                title=meta.get("title", ""),
                status=meta.get("status", ""),
                min_age=meta.get("min_age"),
                max_age=meta.get("max_age"),
                sex=meta.get("sex"),
                locations=meta.get("locations"),
                document_text=doc_text,
            ).model_dump()
        )

    print(f"[retrieve_trials] Found {len(candidates)} candidates")
    state["candidates"] = candidates
    return state


# ---------------------------------------------------------------------------
# Node 3: Hard Filters (deliberately NOT using the LLM — fast, reliable rules)
# ---------------------------------------------------------------------------
def _parse_age_years(age_str: Optional[str]) -> Optional[int]:
    if not age_str:
        return None
    digits = "".join(c for c in age_str if c.isdigit())
    return int(digits) if digits else None


def filter_trials(state: AgentState) -> AgentState:
    patient_age = state["patient"].get("age")
    filtered = []

    for candidate in state["candidates"]:
        # only recruiting/active trials are useful to a real patient
        if candidate["status"] not in ("RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"):
            continue

        if patient_age is not None:
            min_age = _parse_age_years(candidate.get("min_age"))
            max_age = _parse_age_years(candidate.get("max_age"))
            if min_age is not None and patient_age < min_age:
                continue
            if max_age is not None and patient_age > max_age:
                continue

        filtered.append(candidate)

    print(f"[filter_trials] {len(filtered)} candidates passed hard filters")
    state["filtered"] = filtered
    return state


# ---------------------------------------------------------------------------
# Node 4: Reasoning — LLM reads eligibility criteria against patient profile
# ---------------------------------------------------------------------------
def reason_eligibility(state: AgentState) -> AgentState:
    patient = state["patient"]
    matches = []

    for candidate in state["filtered"]:
        prompt = f"""A patient has this profile:
- Condition: {patient['condition']}
- Age: {patient.get('age', 'unknown')}
- Sex: {patient.get('sex', 'unknown')}
- Location: {patient.get('location', 'unknown')}
- Notes: {patient.get('additional_notes', 'none')}

Here is a clinical trial's information:
{candidate['document_text'][:2000]}

Based ONLY on the eligibility criteria and summary above, assess whether this
patient is likely eligible. Return ONLY valid JSON, nothing else:
{{"eligible": "yes"/"no"/"maybe", "confidence": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
"""
        text = call_llm(prompt, max_tokens=200)
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # if the model didn't return clean JSON, skip this candidate rather than crash
            print(f"[reason_eligibility] Could not parse response for {candidate['nct_id']}, skipping")
            continue

        if result.get("eligible") == "no":
            continue  # don't show trials the agent thinks the patient doesn't qualify for

        matches.append(
            TrialMatch(
                nct_id=candidate["nct_id"],
                title=candidate["title"],
                eligible=result["eligible"],
                confidence=float(result["confidence"]),
                reasoning=result["reasoning"],
                url=f"https://clinicaltrials.gov/study/{candidate['nct_id']}",
            ).model_dump()
        )

    print(f"[reason_eligibility] {len(matches)} trials assessed as yes/maybe")
    state["matches"] = matches
    return state


# ---------------------------------------------------------------------------
# Node 5: Ranking
# ---------------------------------------------------------------------------
def rank_results(state: AgentState) -> AgentState:
    ranked = sorted(state["matches"], key=lambda m: m["confidence"], reverse=True)
    state["final_results"] = ranked[:5]  # top 5 for the user
    return state


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("parse_query", parse_query)
    graph.add_node("retrieve_trials", retrieve_trials)
    graph.add_node("filter_trials", filter_trials)
    graph.add_node("reason_eligibility", reason_eligibility)
    graph.add_node("rank_results", rank_results)

    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query", "retrieve_trials")
    graph.add_edge("retrieve_trials", "filter_trials")
    graph.add_edge("filter_trials", "reason_eligibility")
    graph.add_edge("reason_eligibility", "rank_results")
    graph.add_edge("rank_results", END)

    return graph.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    agent = build_agent()
    initial_state: AgentState = {
        "raw_query": args.query,
        "patient": None,
        "candidates": [],
        "filtered": [],
        "matches": [],
        "final_results": [],
    }

    result = agent.invoke(initial_state)

    print("\n=== FINAL RESULTS ===")
    if not result["final_results"]:
        print("No matching trials found.")
    for i, match in enumerate(result["final_results"], 1):
        print(f"\n{i}. {match['title']}")
        print(f"   NCT ID: {match['nct_id']}  |  Confidence: {match['confidence']:.2f}  |  Eligible: {match['eligible']}")
        print(f"   Reasoning: {match['reasoning']}")
        print(f"   URL: {match['url']}")


if __name__ == "__main__":
    main()