"""
Step 13: Streamlit frontend.

A simple demo UI on top of the FastAPI backend — this is what you'll
screen-record for your portfolio, and what a recruiter can actually
click around in on the live deployed link.

Run locally (with the API already running separately):
    streamlit run src/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MedTrial-Match AI", page_icon="🩺", layout="centered")

st.title("🩺 MedTrial-Match AI")
st.caption("An agentic RAG system matching patients to relevant clinical trials.")

with st.expander("ℹ️ How this works"):
    st.markdown(
        """
        This isn't a single AI prompt — it's a 5-step agent:
        1. **Parse** your description into structured patient data
        2. **Retrieve** candidate trials via semantic search + the ClinicalTrials.gov API
        3. **Filter** out trials you clearly don't qualify for (age, recruiting status) — no AI needed here
        4. **Reason** — an LLM reads each trial's real eligibility criteria against your profile
        5. **Rank** the results by confidence, with plain-English explanations

        This project is for demonstration purposes only and is **not medical advice**.
        Always consult a healthcare professional and verify trial details directly on
        [ClinicalTrials.gov](https://clinicaltrials.gov).
        """
    )

query = st.text_area(
    "Describe the patient",
    placeholder="e.g. 62 year old female with type 2 diabetes and heart disease, based in Manchester",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    search_clicked = st.button("Find Trials", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Please describe the patient first.")
    else:
        with st.spinner("Parsing → Retrieving → Filtering → Reasoning → Ranking..."):
            try:
                response = requests.post(f"{API_URL}/match", json={"query": query}, timeout=60)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Couldn't reach the API at {API_URL}. "
                    "Make sure the FastAPI backend is running (`uvicorn src.api:app --reload`)."
                )
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}")
                st.stop()

        st.divider()

        profile = data.get("patient_profile", {})
        if profile:
            st.subheader("Extracted patient profile")
            cols = st.columns(4)
            cols[0].metric("Condition", profile.get("condition") or "—")
            cols[1].metric("Age", profile.get("age") or "—")
            cols[2].metric("Sex", profile.get("sex") or "—")
            cols[3].metric("Location", profile.get("location") or "—")

        st.subheader(f"Matched trials ({data['result_count']} found in {data['latency_seconds']}s)")

        if data["result_count"] == 0:
            st.info("No matching trials found for this profile. Try adding more detail, or a different condition.")
        else:
            for result in data["results"]:
                eligibility_color = {"yes": "🟢", "maybe": "🟡", "no": "🔴"}.get(result["eligible"], "⚪")
                with st.container(border=True):
                    st.markdown(f"### {eligibility_color} {result['title']}")
                    st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.0%}")
                    st.write(result["reasoning"])
                    st.markdown(f"**NCT ID:** `{result['nct_id']}` · [View on ClinicalTrials.gov]({result['url']})")

st.divider()
st.caption("Built by Pranit Salvi · [GitHub](https://github.com/Pranit3434) · Not medical advice.")
