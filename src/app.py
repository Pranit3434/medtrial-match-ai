"""
Step 13 (v2): Streamlit frontend with a proper visual design pass.

Run locally (with the API already running separately):
    streamlit run src/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MedTrial-Match AI",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — gives this a distinct, "medical-tech" look instead of the
# default Streamlit theme. Keeps everything readable; only touches visual
# polish (spacing, gradients, card styling, custom badges).
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2DD4BF 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.6rem;
    }
    .pipeline-strip {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-bottom: 1.8rem;
    }
    .pipeline-step {
        background: rgba(45, 212, 191, 0.08);
        border: 1px solid rgba(45, 212, 191, 0.25);
        color: #2DD4BF;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .result-card {
        background: linear-gradient(145deg, #141B2D 0%, #101728 100%);
        border: 1px solid #22304A;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
        transition: border-color 0.2s ease;
    }
    .result-card:hover {
        border-color: #2DD4BF66;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .badge-yes { background: rgba(52, 211, 153, 0.15); color: #34D399; }
    .badge-maybe { background: rgba(251, 191, 36, 0.15); color: #FBBF24; }
    .badge-no { background: rgba(248, 113, 113, 0.15); color: #F87171; }

    .footer-note {
        color: #64748B;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — example queries so a recruiter can try it in one click,
# plus a short "how it works" explainer.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 MedTrial-Match AI")
    st.caption("Agentic RAG for clinical trial matching")

    st.markdown("---")
    st.markdown("**Try an example**")

    example_queries = [
        "62 year old female with type 2 diabetes and heart disease, based in Manchester",
        "45 year old male, newly diagnosed type 2 diabetes, no complications",
        "58 year old with obesity and type 2 diabetes, on metformin",
    ]

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""

    for i, ex in enumerate(example_queries):
        if st.button(ex, key=f"example_{i}", use_container_width=True):
            st.session_state.query_text = ex

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        """
        <div class="pipeline-strip">
            <span class="pipeline-step">1. Parse</span>
            <span class="pipeline-step">2. Retrieve</span>
            <span class="pipeline-step">3. Filter</span>
            <span class="pipeline-step">4. Reason</span>
            <span class="pipeline-step">5. Rank</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Every match is produced by an LLM reading the trial's real "
        "eligibility criteria against the patient — not just keyword "
        "or vector similarity."
    )

    st.markdown("---")
    st.caption("⚠️ For demonstration purposes only. Not medical advice.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">MedTrial-Match AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Describe a patient. An agent reads real '
    "ClinicalTrials.gov eligibility criteria and explains what actually matches.</div>",
    unsafe_allow_html=True,
)

query = st.text_area(
    "Describe the patient",
    value=st.session_state.get("query_text", ""),
    placeholder="e.g. 62 year old female with type 2 diabetes and heart disease, based in Manchester",
    height=100,
    label_visibility="collapsed",
)

search_clicked = st.button("🔍  Find Trials", type="primary", use_container_width=False)

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
                    "Make sure the FastAPI backend is running (`uvicorn api:app --reload`)."
                )
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}")
                st.stop()

        st.markdown("---")

        profile = data.get("patient_profile", {})
        if profile:
            st.markdown("#### Extracted patient profile")
            cols = st.columns(4)
            cols[0].metric("Condition", (profile.get("condition") or "—")[:16])
            cols[1].metric("Age", profile.get("age") or "—")
            cols[2].metric("Sex", profile.get("sex") or "—")
            cols[3].metric("Location", (profile.get("location") or "—")[:14])

        st.markdown(f"#### Matched trials · {data['result_count']} found in {data['latency_seconds']}s")

        if data["result_count"] == 0:
            st.info("No matching trials found for this profile. Try adding more detail, or a different condition.")
        else:
            badge_map = {
                "yes": ("badge-yes", "✅ LIKELY ELIGIBLE"),
                "maybe": ("badge-maybe", "🟡 POSSIBLY ELIGIBLE"),
                "no": ("badge-no", "🔴 NOT ELIGIBLE"),
            }
            for result in data["results"]:
                badge_class, badge_text = badge_map.get(result["eligible"], ("badge-maybe", "UNKNOWN"))
                st.markdown(
                    f"""
                    <div class="result-card">
                        <span class="badge {badge_class}">{badge_text}</span>
                        <h4 style="margin-top:0.2rem; margin-bottom:0.6rem;">{result['title']}</h4>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.0%}")
                st.write(result["reasoning"])
                st.markdown(
                    f"**NCT ID:** `{result['nct_id']}` &nbsp;·&nbsp; "
                    f"[View on ClinicalTrials.gov]({result['url']})",
                )
                st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Built by Pranit Salvi · '
    '<a href="https://github.com/Pranit3434" style="color:#2DD4BF;">GitHub</a> · '
    "Not medical advice.</div>",
    unsafe_allow_html=True,
)
