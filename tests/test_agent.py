"""
tests/test_agent.py

WHAT THIS FILE DOES:
Tests the parts of our agent that DON'T need a live LLM call. This matters
because GitHub Actions runs in a clean environment with no Groq API key —
so any test that tries to actually call the LLM would fail every time,
not because the code is broken, but because there's no key to use.

Instead, we test two things that ARE fully deterministic and don't need
any external service:
  1. The Pydantic models (PatientProfile, TrialMatch) — do they accept
     good data and reject bad data the way we expect?
  2. filter_trials() — the rule-based age/status filtering logic, which
     is plain Python with no AI involved at all.

The LLM-dependent parts (parse_query, reason_eligibility) get tested
separately with "mocks" — fake stand-ins for the real API — which is
the standard way to test AI code without spending money or needing
network access every time tests run.
"""

import sys
import os

# This adds the src/ folder to Python's search path, so "import models"
# and "import agent" work the same way whether pytest is run from the
# project root or from inside tests/. Without this, pytest often can't
# find our source files.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock

# agent.py creates a real Groq client the moment it's imported (it needs
# an API key to do that). Since this test environment has no key, we
# fake out the whole groq module BEFORE importing agent — this is a
# common pattern called "mocking a dependency."
sys.modules["groq"] = MagicMock()

from models import PatientProfile, TrialMatch
import agent


# ---------------------------------------------------------------------------
# Tests for the Pydantic models
# ---------------------------------------------------------------------------

def test_patient_profile_accepts_full_data():
    """A PatientProfile should build correctly when given all fields."""
    patient = PatientProfile(
        condition="type 2 diabetes",
        age=62,
        sex="FEMALE",
        location="Manchester",
        additional_notes="heart disease",
    )
    assert patient.condition == "type 2 diabetes"
    assert patient.age == 62


def test_patient_profile_allows_missing_optional_fields():
    """
    Age, sex, location, and notes are all optional — a patient query
    might not mention them. Only 'condition' is required. This test
    makes sure the model doesn't crash when those are left out.
    """
    patient = PatientProfile(condition="type 2 diabetes")
    assert patient.age is None
    assert patient.sex is None


def test_trial_match_holds_expected_fields():
    """A TrialMatch (the agent's final output for one trial) should
    store all the fields the UI and API rely on."""
    match = TrialMatch(
        nct_id="NCT12345",
        title="Test Trial",
        eligible="yes",
        confidence=0.9,
        reasoning="Meets all inclusion criteria.",
        url="https://clinicaltrials.gov/study/NCT12345",
    )
    assert match.eligible == "yes"
    assert 0.0 <= match.confidence <= 1.0


# ---------------------------------------------------------------------------
# Tests for filter_trials() — pure rule-based logic, no LLM needed
# ---------------------------------------------------------------------------

def _make_state(patient_age, candidates):
    """
    Small helper to build the 'state' dict that filter_trials() expects.
    Keeping this in one place means if the state shape ever changes,
    we only need to update it here instead of in every test.
    """
    return {
        "patient": {"age": patient_age},
        "candidates": candidates,
    }


def test_filter_trials_excludes_completed_trials():
    """
    A patient should never be shown a trial that's already finished
    recruiting — that's the whole point of this filter existing.
    """
    state = _make_state(
        patient_age=62,
        candidates=[
            {"nct_id": "NCT1", "status": "COMPLETED", "min_age": "18 Years", "max_age": "80 Years", "title": "Old trial"},
            {"nct_id": "NCT2", "status": "RECRUITING", "min_age": "18 Years", "max_age": "80 Years", "title": "Active trial"},
        ],
    )
    result = agent.filter_trials(state)
    remaining_ids = [c["nct_id"] for c in result["filtered"]]
    assert "NCT1" not in remaining_ids
    assert "NCT2" in remaining_ids


def test_filter_trials_excludes_patient_outside_age_range():
    """
    If a trial only accepts ages 18-40 and our patient is 62, that
    trial should be filtered out — no point showing it if they're
    not eligible on age alone.
    """
    state = _make_state(
        patient_age=62,
        candidates=[
            {"nct_id": "NCT1", "status": "RECRUITING", "min_age": "18 Years", "max_age": "40 Years", "title": "Too young a cutoff"},
        ],
    )
    result = agent.filter_trials(state)
    assert len(result["filtered"]) == 0


def test_filter_trials_includes_patient_inside_age_range():
    """The mirror image of the test above — a patient who DOES fit
    the age range should make it through the filter."""
    state = _make_state(
        patient_age=62,
        candidates=[
            {"nct_id": "NCT1", "status": "RECRUITING", "min_age": "18 Years", "max_age": "80 Years", "title": "Fits fine"},
        ],
    )
    result = agent.filter_trials(state)
    assert len(result["filtered"]) == 1


def test_filter_trials_handles_missing_age_gracefully():
    """
    Some trials don't specify a maximum age (e.g. "18 Years" and up,
    no cap). The filter should treat a missing bound as "no limit"
    rather than crashing or wrongly excluding the patient.
    """
    state = _make_state(
        patient_age=90,
        candidates=[
            {"nct_id": "NCT1", "status": "RECRUITING", "min_age": "18 Years", "max_age": None, "title": "No upper limit"},
        ],
    )
    result = agent.filter_trials(state)
    assert len(result["filtered"]) == 1
