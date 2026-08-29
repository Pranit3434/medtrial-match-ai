"""
Shared data models for the agent.

Keeping these in one file means every node (parser, retrieval, filter,
reasoning, ranking) speaks the same "language" for what a patient
profile and a trial match look like — this is what makes the LangGraph
state machine reliable instead of passing loose dicts around.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    """Structured fields extracted from the user's free-text query."""
    condition: str = Field(description="The medical condition, e.g. 'type 2 diabetes'")
    age: Optional[int] = Field(default=None, description="Patient age in years")
    sex: Optional[str] = Field(default=None, description="MALE, FEMALE, or None if unspecified")
    location: Optional[str] = Field(default=None, description="City or region, e.g. 'Manchester'")
    additional_notes: Optional[str] = Field(
        default=None,
        description="Anything else relevant: prior treatments, disease stage, comorbidities",
    )


class TrialCandidate(BaseModel):
    """A trial pulled from retrieval, before reasoning is applied."""
    nct_id: str
    title: str
    status: str
    min_age: Optional[str] = None
    max_age: Optional[str] = None
    sex: Optional[str] = None
    locations: Optional[str] = None
    document_text: str  # the full text used for reasoning (summary + eligibility criteria)


class TrialMatch(BaseModel):
    """A trial after the reasoning node has evaluated it against the patient."""
    nct_id: str
    title: str
    eligible: str  # "yes", "no", or "maybe"
    confidence: float  # 0.0 - 1.0
    reasoning: str  # one or two sentence plain-English explanation
    url: str
