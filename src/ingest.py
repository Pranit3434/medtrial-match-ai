"""
Step 3: Ingest flattened trial records into a local ChromaDB vector store.

Takes the JSON produced by fetch_trials.py, builds a searchable text
chunk per trial (title + summary + eligibility criteria), embeds it,
and stores it with metadata so we can later filter by status/age/etc.

Run:
    python src/ingest.py --input data/raw_trials.json
"""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = "clinical_trials"


def build_document_text(trial: dict) -> str:
    """
    Combine the fields that matter for semantic search into one text
    blob per trial. This is what gets embedded — keep it focused on
    content a patient's query would semantically match against.
    """
    parts = [
        f"Title: {trial.get('title', '')}",
        f"Conditions: {', '.join(trial.get('conditions', []))}",
        f"Summary: {trial.get('summary', '')}",
        f"Eligibility Criteria: {trial.get('eligibility_criteria', '')}",
    ]
    return "\n".join(p for p in parts if p.strip())


def build_metadata(trial: dict) -> dict:
    """
    Fields we want to filter/display on later, WITHOUT re-parsing the
    document text. Chroma metadata values must be str/int/float/bool,
    so lists get joined into strings.
    """
    return {
        "nct_id": trial.get("nct_id") or "",
        "title": trial.get("title") or "",
        "status": trial.get("status") or "",
        "min_age": trial.get("min_age") or "",
        "max_age": trial.get("max_age") or "",
        "sex": trial.get("sex") or "",
        "phases": ", ".join(trial.get("phases", [])),
        "locations": ", ".join(trial.get("locations", [])[:5]),  # cap to keep it small
    }


def ingest(input_path: str):
    trials = json.loads(Path(input_path).read_text())
    print(f"Loaded {len(trials)} trials from {input_path}")

    # Default embedding function (all-MiniLM-L6-v2, runs locally, no API key needed).
    # Swap for OpenAIEmbeddingFunction or a Voyage embedding function later
    # if you want higher-quality embeddings — this default is fine to start.
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    ids, documents, metadatas = [], [], []
    for trial in trials:
        nct_id = trial.get("nct_id")
        if not nct_id:
            continue  # skip malformed records
        ids.append(nct_id)
        documents.append(build_document_text(trial))
        metadatas.append(build_metadata(trial))

    if not ids:
        print("No valid trials to ingest — check your input file.")
        return

    # upsert so re-running this script is safe and idempotent
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Upserted {len(ids)} trials into collection '{COLLECTION_NAME}' at {CHROMA_DB_PATH}")


def sanity_check_query(query: str, n_results: int = 3):
    """
    Quick manual check: run a semantic search and print what comes back.
    This is Step 4 from the build plan — confirming retrieval works
    before building any agent logic on top of it.
    """
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    results = collection.query(query_texts=[query], n_results=n_results)
    print(f"\nTop {n_results} results for query: '{query}'")
    for i, (doc_id, meta, dist) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"\n{i+1}. {meta['title']}  (nct_id={doc_id}, distance={dist:.4f})")
        print(f"   status={meta['status']}, age={meta['min_age']}-{meta['max_age']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw_trials.json")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional: run a sanity-check semantic search after ingesting",
    )
    args = parser.parse_args()

    ingest(args.input)

    if args.query:
        sanity_check_query(args.query)
