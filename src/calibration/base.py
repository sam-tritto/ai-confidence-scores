"""
Abstract Base Class for Confidence Evaluation Engines.
"""

from abc import ABC, abstractmethod
import os
import time
from typing import Any, Dict, Optional
from google import genai
from google.genai import types

from src.schema import (
    AuditDecision,
    CalibrationResult,
    DomainRole,
    ResumeExtraction,
    SeniorityLevel,
)


class BaseConfidenceEngine(ABC):
    """Abstract base class for LLM confidence calibration engines."""

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: str = "gemini-2.5-flash",
        audit_threshold: float = 0.75,
    ):
        self.model_name = os.getenv("GEMINI_MODEL", model_name)
        self.audit_threshold = float(os.getenv("AUDIT_CONFIDENCE_THRESHOLD", str(audit_threshold)))
        
        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = None

    @abstractmethod
    def evaluate(
        self,
        resume_text: str,
        pdf_bytes: Optional[bytes] = None,
        ground_truth: Optional[ResumeExtraction] = None,
    ) -> CalibrationResult:
        """Evaluate resume text and return structured extraction with calibrated confidence score."""
        pass

    def determine_audit_decision(self, confidence: float) -> AuditDecision:
        """Route to automated processing vs human review based on threshold."""
        if confidence >= self.audit_threshold:
            return AuditDecision.AUTOMATE
        return AuditDecision.FLAG_FOR_HUMAN_REVIEW

    def _create_fallback_extraction(self, resume_text: str) -> ResumeExtraction:
        """Rule-based heuristic fallback extraction when API key is missing or offline mode is used."""
        text_lower = resume_text.lower()
        
        # Domain Role Heuristic
        if "data scientist" in text_lower or "machine learning" in text_lower or "python" in text_lower and "model" in text_lower:
            role = DomainRole.DATA_SCIENCE
        elif "software" in text_lower or "developer" in text_lower or "full stack" in text_lower or "backend" in text_lower:
            role = DomainRole.SOFTWARE_ENGINEERING
        elif "finance" in text_lower or "accounting" in text_lower or "investment" in text_lower:
            role = DomainRole.FINANCE
        elif "recruiter" in text_lower or "hr" in text_lower or "talent" in text_lower:
            role = DomainRole.HUMAN_RESOURCES
        elif "sales" in text_lower or "business development" in text_lower or "marketing" in text_lower:
            role = DomainRole.BUSINESS_DEVELOPMENT
        else:
            role = DomainRole.OTHER

        # Seniority Heuristic
        if "senior" in text_lower or "sr." in text_lower:
            seniority = SeniorityLevel.SENIOR
        elif "lead" in text_lower or "manager" in text_lower or "head" in text_lower:
            seniority = SeniorityLevel.LEAD_MANAGEMENT
        elif "junior" in text_lower or "associate" in text_lower:
            seniority = SeniorityLevel.JUNIOR
        elif "intern" in text_lower:
            seniority = SeniorityLevel.INTERN
        elif "vp" in text_lower or "chief" in text_lower:
            seniority = SeniorityLevel.EXECUTIVE
        else:
            seniority = SeniorityLevel.MID_LEVEL

        # Extract lines as claims
        lines = [line.strip() for line in resume_text.splitlines() if len(line.strip()) > 15]
        key_claims = lines[:4] if lines else ["Candidate has relevant experience."]
        
        # Extract skills keyword search
        skills_candidates = ["Python", "SQL", "Machine Learning", "Java", "Docker", "AWS", "PyTorch", "Finance", "Communication"]
        key_skills = [s for s in skills_candidates if s.lower() in text_lower]

        return ResumeExtraction(
            candidate_name="Extracted Candidate",
            domain_role=role,
            seniority_level=seniority,
            years_of_experience=5.0,
            key_skills=key_skills or ["General"],
            key_claims=key_claims,
            raw_text_summary=f"Resume for candidate in {role.value} with {seniority.value} experience.",
        )
