"""
Step 2: Explore the ClinicalTrials.gov API (v2) and pull raw trial records.

This version filters for actively recruiting trials at the API level,
since a real patient matching tool has no use for completed/terminated
trials — pulling only relevant records here also keeps the downstream
vector database focused and the reasoning node's job realistic.

Docs: https://clinicaltrials.gov/data-api/api

Run:
    python src/fetch_trials.py --condition "type 2 diabetes" --max-results 100
"""

import argparse
import json
import time
from pathlib import Path

import requests

API_BASE = "https://clinicaltrials.gov/api/v2/studies"

FIELDS = [
    "NCTId",
    "BriefTitle",
    "Condition",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "OverallStatus",
    "Phase",
    "LocationCity",
    "LocationCountry",
    "BriefSummary",
]


def fetch_trials(condition: str, max_results: int = 100, page_size: int = 50) -> list[dict]:
    all_studies = []
    next_page_token = None

    while len(all_studies) < max_results:
        params = {
            "query.cond": condition,
            "fields": ",".join(FIELDS),
            "pageSize": min(page_size, max_results - len(all_studies)),
            "format": "json",
            # Only pull trials that are actually open to new patients —
            # this is the fix: without this filter the API returns a mix
            # of recruiting, completed, and terminated trials by default.
            "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ENROLLING_BY_INVITATION",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        studies = payload.get("studies", [])
        if not studies:
            break

        all_studies.extend(studies)
        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.3)

    return all_studies[:max_results]


def flatten_study(study: dict) -> dict:
    protocol = study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    elig = protocol.get("eligibilityModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    conditions = protocol.get("conditionsModule", {})
    description = protocol.get("descriptionModule", {})
    locations = protocol.get("contactsLocationsModule", {}).get("locations", [])

    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle"),
        "conditions": conditions.get("conditions", []),
        "eligibility_criteria": elig.get("eligibilityCriteria"),
        "min_age": elig.get("minimumAge"),
        "max_age": elig.get("maximumAge"),
        "sex": elig.get("sex"),
        "status": status.get("overallStatus"),
        "phases": design.get("phases", []),
        "summary": description.get("briefSummary"),
        "locations": [
            f"{loc.get('city', '')}, {loc.get('country', '')}"
            for loc in locations
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch trials from ClinicalTrials.gov")
    parser.add_argument("--condition", type=str, default="type 2 diabetes")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/raw_trials.json")
    args = parser.parse_args()

    print(f"Fetching up to {args.max_results} RECRUITING trials for condition: '{args.condition}'...")
    raw_studies = fetch_trials(args.condition, args.max_results)
    print(f"Retrieved {len(raw_studies)} raw records.")

    flattened = [flatten_study(s) for s in raw_studies]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(flattened, indent=2))
    print(f"Saved flattened records to {out_path.resolve()}")

    if flattened:
        statuses = {}
        for t in flattened:
            statuses[t["status"]] = statuses.get(t["status"], 0) + 1
        print(f"\nStatus breakdown: {statuses}")
        print("\n--- Sample record ---")
        print(json.dumps(flattened[0], indent=2)[:1000])


if __name__ == "__main__":
    main()
