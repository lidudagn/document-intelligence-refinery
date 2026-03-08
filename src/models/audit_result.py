from pydantic import BaseModel, Field
from typing import List
from enum import Enum

from src.models.provenance import Citation


class AuditVerdict(str, Enum):
    """Four-tier verdict for claim verification."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIABLE = "unverifiable"
    CONTRADICTED = "contradicted"


class AuditResult(BaseModel):
    """
    Result of a claim verification against the document corpus.
    Produced by the AuditAgent when explicitly requested.
    """
    claim: str = Field(description="The original claim to verify.")
    verdict: AuditVerdict = Field(description="Overall verification verdict.")
    confidence: float = Field(
        default=0.0,
        description="Verification confidence: 3/3 sources agree >= 0.9, 2/3 >= 0.7, 1/3 >= 0.4, 0/3 = 0.0"
    )
    supporting_citations: List[Citation] = Field(
        default_factory=list,
        description="Citations that support the claim."
    )
    contradicting_evidence: List[Citation] = Field(
        default_factory=list,
        description="Citations that contradict the claim."
    )
    reasoning: str = Field(
        default="",
        description="LLM chain-of-thought explanation of the verdict."
    )
