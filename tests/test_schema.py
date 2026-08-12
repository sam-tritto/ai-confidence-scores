"""
Unit tests for schema validation and enum parsing.
"""

from src.schema import (
    AuditDecision,
    CalibrationResult,
    DomainRole,
    ResumeExtraction,
    SeniorityLevel,
)


def test_domain_role_fuzzy_parsing():
    assert DomainRole("Software Development") == DomainRole.SOFTWARE_ENGINEERING
    assert DomainRole("Machine Learning Engineer") == DomainRole.DATA_SCIENCE
    assert DomainRole("Financial Analyst") == DomainRole.FINANCE
    assert DomainRole("Talent Acquisition") == DomainRole.HUMAN_RESOURCES
    assert DomainRole("Sales Manager") == DomainRole.BUSINESS_DEVELOPMENT
    assert DomainRole("Astronaut") == DomainRole.OTHER


def test_seniority_level_fuzzy_parsing():
    assert SeniorityLevel("Sr. Engineer") == SeniorityLevel.SENIOR
    assert SeniorityLevel("Associate Developer") == SeniorityLevel.JUNIOR
    assert SeniorityLevel("Engineering Lead") == SeniorityLevel.LEAD_MANAGEMENT
    assert SeniorityLevel("VP of Product") == SeniorityLevel.EXECUTIVE
    assert SeniorityLevel("Internship") == SeniorityLevel.INTERN
    assert SeniorityLevel("Unspecified") == SeniorityLevel.OTHER


def test_resume_extraction_creation():
    ext = ResumeExtraction(
        candidate_name="Alice Smith",
        domain_role=DomainRole.SOFTWARE_ENGINEERING,
        seniority_level=SeniorityLevel.SENIOR,
        years_of_experience=8.5,
        key_skills=["Python", "C++", "Docker"],
        key_claims=["8.5 years of C++ experience"],
    )
    assert ext.candidate_name == "Alice Smith"
    assert ext.domain_role == DomainRole.SOFTWARE_ENGINEERING
    assert len(ext.key_skills) == 3


def test_calibration_result_schema(sample_ground_truth):
    res = CalibrationResult(
        method_name="Test Method",
        extraction=sample_ground_truth,
        raw_confidence=0.85,
        calibrated_confidence=0.82,
        audit_decision=AuditDecision.AUTOMATE,
        latency_ms=120.5,
        metadata={"test": "data"},
    )
    assert res.method_name == "Test Method"
    assert res.audit_decision == AuditDecision.AUTOMATE
    assert res.calibrated_confidence == 0.82
