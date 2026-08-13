"""
Pydantic Schemas for AI Confidence Scores evaluation and calibration method.
Supports configurable domain-agnostic schemas and specialized extraction models.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field


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


class GenericExtraction(BaseModel):
    """Domain-agnostic generic extraction schema for any text classification task."""
    primary_category: str = Field(default="Unspecified", description="Primary category or label classified from text")
    confidence_estimate: Optional[float] = Field(default=None, description="Optional raw model confidence estimate")
    extracted_fields: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs of extracted attributes")
    key_claims: List[str] = Field(default_factory=list, description="List of key factual claims extracted from input text")
    summary: Optional[str] = Field(default=None, description="Brief summary of input text")


class ResumeExtraction(BaseModel):
    """Specialized extraction schema for PDF Resume classification benchmark."""
    candidate_name: str = Field(default="Unknown Candidate", description="Full name of the candidate")
    domain_role: DomainRole = Field(default=DomainRole.OTHER, description="Classified professional domain role")
    seniority_level: SeniorityLevel = Field(default=SeniorityLevel.OTHER, description="Extracted candidate seniority level")
    years_of_experience: float = Field(default=0.0, description="Estimated total years of professional experience")
    key_skills: List[str] = Field(default_factory=list, description="Top technical and professional skills extracted")
    key_claims: List[str] = Field(default_factory=list, description="Atomic factual claims extracted from the resume")
    raw_text_summary: Optional[str] = Field(default=None, description="Brief overall summary of candidate profile")


class VerbalizedConfidenceOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for extraction and confidence score")
    extraction_data: Dict[str, Any] = Field(default_factory=dict, description="Structured extracted data matching schema")
    verbalized_confidence_score: float = Field(
        description="Self-assessed confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )


class CustomerSupportTicket(BaseModel):
    """Example custom user-defined schema for customer support classification."""
    ticket_id: str = Field(default="TICK-000")
    issue_category: str = Field(default="General Inquiry")
    urgency_level: str = Field(default="Low")
    affected_product: str = Field(default="Core Platform")
    key_claims: List[str] = Field(default_factory=list)


class ContinuousPromptingOutput(BaseModel):
    rationale: str = Field(description="Detailed rationale explaining certainty level")
    extraction_data: Dict[str, Any] = Field(default_factory=dict, description="Structured extracted data matching schema")
    confidence_rating_0_to_100: float = Field(
        description="Continuous rating from 0 to 100 representing confidence",
        ge=0.0,
        le=100.0,
    )


class ConfidenceLevel(str, Enum):
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

    @classmethod
    def _missing_(cls, value: object) -> "ConfidenceLevel":
        if isinstance(value, str):
            val_lower = value.lower()
            if "very low" in val_lower or "very_low" in val_lower:
                return cls.VERY_LOW
            if "very high" in val_lower or "very_high" in val_lower:
                return cls.VERY_HIGH
            if "low" in val_lower:
                return cls.LOW
            if "high" in val_lower:
                return cls.HIGH
            if "mid" in val_lower or "medium" in val_lower:
                return cls.MEDIUM
        return cls.MEDIUM

    @property
    def numeric_score(self) -> float:
        mapping = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9,
        }
        return mapping.get(self, 0.5)


class StructuredSelfAssessmentOutput(BaseModel):
    rationale: str = Field(description="Detailed step-by-step chain-of-thought rationale explaining certainty")
    extraction_data: Dict[str, Any] = Field(default_factory=dict, description="Structured extracted fields")
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Categorical confidence rating: [Very Low, Low, Medium, High, Very High]"
    )
    numerical_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Continuous self-assessed confidence score between 0.0 and 1.0"
    )



class ClaimVerificationResult(BaseModel):
    claim: str = Field(description="Atomic factual claim extracted from LLM completion")
    verdict: str = Field(description="Verdict: SUPPORTED, CONTRADICTED, or UNVERIFIABLE")
    evidence_snippet: Optional[str] = Field(default=None, description="Direct matching snippet from source text if supported")


class GroundingOutput(BaseModel):
    claims: List[ClaimVerificationResult] = Field(default_factory=list)
    grounding_score: float = Field(description="Fraction of claims supported by source text", ge=0.0, le=1.0)


class JudgeEvaluation(BaseModel):
    precision_score: float = Field(description="Precision score from 1.0 to 5.0", ge=1.0, le=5.0)
    hallucination_score: float = Field(description="Hallucination risk score from 1.0 (severe) to 5.0 (none)", ge=1.0, le=5.0)
    completeness_score: float = Field(description="Completeness score from 1.0 to 5.0", ge=1.0, le=5.0)
    justification: str = Field(description="Evaluator rationale for rubric scores")
    normalized_score: float = Field(description="Overall normalized quality/confidence score between 0.0 and 1.0", ge=0.0, le=1.0)


class CalibrationResult(BaseModel):
    method_name: str = Field(description="Name of confidence evaluation method")
    extraction: Any = Field(description="Structured extraction output object (Pydantic model instance)")
    raw_confidence: float = Field(description="Uncalibrated confidence score (0.0 to 1.0)")
    calibrated_confidence: float = Field(description="Calibrated confidence score (0.0 to 1.0)")
    audit_decision: AuditDecision = Field(description="Human-in-the-loop audit routing flag")
    latency_ms: float = Field(description="Execution latency in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Method specific intermediate metrics")


class EvaluationMetrics(BaseModel):
    method_name: str
    accuracy: float
    ece: float
    mce: float
    brier_score: float
    mean_confidence: float
    mean_latency_ms: float
    automate_ratio: float
