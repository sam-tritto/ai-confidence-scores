"""
Pydantic Schemas for AI Confidence Scores evaluation and calibration engine.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DomainRole(str, Enum):
    SOFTWARE_ENGINEERING = "Software Engineering"
    DATA_SCIENCE = "Data Science"
    FINANCE = "Finance"
    HUMAN_RESOURCES = "HR"
    BUSINESS_DEVELOPMENT = "Business Development"
    OTHER = "Other"

    @classmethod
    def _missing_(cls, value: object) -> "DomainRole":
        if isinstance(value, str):
            val_lower = value.lower()
            if "data" in val_lower or "machine learning" in val_lower or "ai" in val_lower or "analytics" in val_lower:
                return cls.DATA_SCIENCE
            if "software" in val_lower or "engineer" in val_lower or "developer" in val_lower:
                return cls.SOFTWARE_ENGINEERING
            if "finance" in val_lower or "accounting" in val_lower or "financial" in val_lower:
                return cls.FINANCE
            if "hr" in val_lower or "human" in val_lower or "recruitment" in val_lower or "talent" in val_lower or "acquisition" in val_lower:
                return cls.HUMAN_RESOURCES
            if "business" in val_lower or "sales" in val_lower or "marketing" in val_lower:
                return cls.BUSINESS_DEVELOPMENT
        return cls.OTHER


class SeniorityLevel(str, Enum):
    INTERN = "Intern"
    JUNIOR = "Junior"
    MID_LEVEL = "Mid-Level"
    SENIOR = "Senior"
    LEAD_MANAGEMENT = "Lead / Management"
    EXECUTIVE = "Executive"
    OTHER = "Other"

    @classmethod
    def _missing_(cls, value: object) -> "SeniorityLevel":
        if isinstance(value, str):
            val_lower = value.lower()
            if "intern" in val_lower:
                return cls.INTERN
            if "junior" in val_lower or "entry" in val_lower or "associate" in val_lower:
                return cls.JUNIOR
            if "mid" in val_lower:
                return cls.MID_LEVEL
            if "senior" in val_lower or "sr" in val_lower:
                return cls.SENIOR
            if "lead" in val_lower or "manager" in val_lower or "head" in val_lower or "director" in val_lower:
                return cls.LEAD_MANAGEMENT
            if "vp" in val_lower or "chief" in val_lower or "c-level" in val_lower or "executive" in val_lower:
                return cls.EXECUTIVE
        return cls.OTHER


class AuditDecision(str, Enum):
    AUTOMATE = "AUTOMATE"
    FLAG_FOR_HUMAN_REVIEW = "FLAG_FOR_HUMAN_REVIEW"


class ResumeExtraction(BaseModel):
    candidate_name: str = Field(default="Unknown Candidate", description="Full name of the candidate")
    domain_role: DomainRole = Field(default=DomainRole.OTHER, description="Classified professional domain role")
    seniority_level: SeniorityLevel = Field(default=SeniorityLevel.OTHER, description="Extracted candidate seniority level")
    years_of_experience: float = Field(default=0.0, description="Estimated total years of professional experience")
    key_skills: List[str] = Field(default_factory=list, description="Top technical and professional skills extracted")
    key_claims: List[str] = Field(default_factory=list, description="Atomic factual claims extracted from the resume")
    raw_text_summary: Optional[str] = Field(default=None, description="Brief overall summary of the candidate profile")


class VerbalizedConfidenceOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the extraction and confidence score")
    candidate_name: str = Field(default="Unknown Candidate")
    domain_role: DomainRole = Field(default=DomainRole.OTHER)
    seniority_level: SeniorityLevel = Field(default=SeniorityLevel.OTHER)
    years_of_experience: float = Field(default=0.0)
    key_skills: List[str] = Field(default_factory=list)
    key_claims: List[str] = Field(default_factory=list)
    verbalized_confidence_score: float = Field(
        description="Self-assessed confidence score between 0.0 (completely uncertain) and 1.0 (certain)",
        ge=0.0,
        le=1.0,
    )


class ContinuousPromptingOutput(BaseModel):
    rationale: str = Field(description="Detailed rationale explaining certainty level")
    extracted_domain_role: DomainRole = Field(default=DomainRole.OTHER)
    extracted_seniority: SeniorityLevel = Field(default=SeniorityLevel.OTHER)
    candidate_name: str = Field(default="Unknown Candidate")
    years_of_experience: float = Field(default=0.0)
    key_skills: List[str] = Field(default_factory=list)
    key_claims: List[str] = Field(default_factory=list)
    confidence_rating_0_to_100: float = Field(
        description="Continuous integer rating from 0 to 100 representing confidence",
        ge=0.0,
        le=100.0,
    )


class ClaimVerificationResult(BaseModel):
    claim: str = Field(description="Atomic factual claim extracted from LLM completion")
    verdict: str = Field(description="Verdict: SUPPORTED, CONTRADICTED, or UNVERIFIABLE")
    evidence_snippet: Optional[str] = Field(default=None, description="Direct matching snippet from PDF text if supported")


class GroundingOutput(BaseModel):
    claims: List[ClaimVerificationResult] = Field(default_factory=list)
    grounding_score: float = Field(description="Fraction of claims supported by source PDF text", ge=0.0, le=1.0)


class JudgeEvaluation(BaseModel):
    precision_score: float = Field(description="Precision score from 1.0 to 5.0", ge=1.0, le=5.0)
    hallucination_score: float = Field(description="Hallucination risk score from 1.0 (severe) to 5.0 (none)", ge=1.0, le=5.0)
    completeness_score: float = Field(description="Completeness score from 1.0 to 5.0", ge=1.0, le=5.0)
    justification: str = Field(description="Evaluator rationale for rubric scores")
    normalized_score: float = Field(description="Overall normalized quality/confidence score between 0.0 and 1.0", ge=0.0, le=1.0)


class CalibrationResult(BaseModel):
    engine_name: str = Field(description="Name of the confidence evaluation engine")
    extraction: ResumeExtraction = Field(description="Structured resume extraction output")
    raw_confidence: float = Field(description="Uncalibrated confidence score (0.0 to 1.0)")
    calibrated_confidence: float = Field(description="Calibrated confidence score (0.0 to 1.0)")
    audit_decision: AuditDecision = Field(description="Human-in-the-loop audit routing flag (AUTOMATE vs FLAG_FOR_HUMAN_REVIEW)")
    latency_ms: float = Field(description="Execution latency in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Engine specific intermediate artifacts and metrics")


class EvaluationMetrics(BaseModel):
    engine_name: str
    accuracy: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    mean_confidence: float
    mean_latency_ms: float
    automate_ratio: float
