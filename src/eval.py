"""
Step 14-16: Evaluation harness.

Runs a curated set of test patient queries through the full agent and
scores the results two ways:
  1. Objective metrics — did it return anything, how long did it take
  2. LLM-as-judge — a SEPARATE LLM call rates whether each match's
     reasoning is actually sound, given the patient profile and the
     trial's eligibility text

This version adds retry logic and pacing between calls, since free-tier
API rate limits can cause a judge call to return a truncated or non-JSON
response under heavy back-to-back load.

Run:
    python src/eval.py
"""

import json
import os
import re
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from agent import build_agent

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"
judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Small delay between API calls to stay well under free-tier rate limits.
# Increase this if you still see parse failures in data/eval_results.json.
CALL_DELAY_SECONDS = 2

TEST_QUERIES = [
    "62 year old female with type 2 diabetes and heart disease, based in Manchester",
    "45 year old male, newly diagnosed type 2 diabetes, no complications",
    "58 year old with obesity and type 2 diabetes, on metformin",
    "70 year old female with long-standing type 2 diabetes and kidney problems",
    "35 year old male with type 2 diabetes, otherwise healthy",
    "50 year old with type 2 diabetes and high blood pressure",
    "28 year old female recently diagnosed with type 2 diabetes during pregnancy screening",
    "65 year old male, type 2 diabetes, history of stroke",
    "40 year old with poorly controlled type 2 diabetes despite medication",
    "55 year old female with type 2 diabetes, interested in lifestyle intervention trials",
]


def _extract_json(text: str) -> str:
    """
    Pulls the first {...} block out of a response, even if the model
    added stray text before/after it, or wrapped its answer in a
    <think>...</think> reasoning block (common with "thinking" models
    on Groq's free tier).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def judge_match(patient: dict, match: dict, retries: int = 2) -> dict:
    """
    A second, independent LLM call that critiques the agent's own output.
    Retries on parse failure since a rate-limited or truncated response
    is common under free-tier limits — a real failure after retries is
    recorded as None rather than silently guessed at.
    """
    prompt = f"""You are auditing an AI clinical trial matching system.

Patient profile: {json.dumps(patient)}

The system matched this trial and gave this reasoning:
Trial: {match['title']}
Eligibility verdict: {match['eligible']}
Confidence: {match['confidence']}
Reasoning given: {match['reasoning']}

Rate the QUALITY of this reasoning on a scale of 1-5, where:
1 = reasoning is wrong or contradicts the patient profile
3 = reasoning is plausible but vague or incomplete
5 = reasoning is specific, correct, and clearly justified

Respond with ONLY this JSON object and nothing else before or after it:
{{"score": <int 1-5>, "critique": "<max 10 words>"}}
"""
    last_error = None
    text = ""
    for attempt in range(retries + 1):
        try:
            response = judge_client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            json_text = _extract_json(text)
            parsed = json.loads(json_text)
            if "score" in parsed:
                return parsed
            last_error = f"No 'score' key in response: {text[:100]}"
        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed: {e}. Raw: {text[:100]}"
        except Exception as e:
            last_error = f"API call failed: {e}"

        if attempt < retries:
            time.sleep(CALL_DELAY_SECONDS * (attempt + 1))  # backoff

    return {"score": None, "critique": f"Failed after {retries + 1} attempts: {last_error}"}


def run_evaluation():
    agent = build_agent()
    results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Running: {query}")
        start = time.time()

        initial_state = {
            "raw_query": query, "patient": None, "candidates": [],
            "filtered": [], "matches": [], "final_results": [],
        }

        try:
            output = agent.invoke(initial_state)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "query": query, "error": str(e), "latency": time.time() - start,
                "result_count": 0, "judge_scores": [],
            })
            time.sleep(CALL_DELAY_SECONDS)
            continue

        latency = time.time() - start
        final_results = output["final_results"]
        patient = output["patient"]

        judge_scores = []
        judge_critiques = []
        for match in final_results:
            time.sleep(CALL_DELAY_SECONDS)  # pace judge calls to avoid rate limits
            judged = judge_match(patient, match)
            judge_scores.append(judged.get("score"))
            judge_critiques.append(judged.get("critique"))
            print(f"  - {match['title'][:50]}... | agent confidence: {match['confidence']:.2f} | judge score: {judged.get('score')}/5")

        results.append({
            "query": query,
            "patient": patient,
            "result_count": len(final_results),
            "latency": round(latency, 2),
            "judge_scores": judge_scores,
            "judge_critiques": judge_critiques,
        })

        time.sleep(CALL_DELAY_SECONDS)  # pace between queries too

    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    with_results = sum(1 for r in results if r.get("result_count", 0) > 0)
    latencies = [r["latency"] for r in results if "latency" in r]
    all_scores = [s for r in results for s in r.get("judge_scores", []) if s is not None]
    failed_judgments = sum(1 for r in results for s in r.get("judge_scores", []) if s is None)

    return {
        "total_queries": total,
        "queries_with_at_least_one_match": with_results,
        "match_rate": round(with_results / total, 2) if total else 0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "avg_judge_score": round(sum(all_scores) / len(all_scores), 2) if all_scores else None,
        "total_matches_judged": len(all_scores),
        "failed_judgments": failed_judgments,
    }


def main():
    print(f"Running evaluation on {len(TEST_QUERIES)} test queries...")
    print(f"(pacing calls with a {CALL_DELAY_SECONDS}s delay to respect free-tier rate limits — this will take a few minutes)\n")
    results = run_evaluation()
    summary = summarize(results)

    out_path = Path("data/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nFull results saved to {out_path.resolve()}")
    print("\nCopy the summary above into your README's evaluation section.")


if __name__ == "__main__":
    main()
