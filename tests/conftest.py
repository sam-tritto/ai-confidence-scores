"""
Pytest configuration and shared fixtures for test suite.
"""

from pathlib import Path
import tempfile
import pymupdf as fitz  # PyMuPDF
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel

from src.schema import (
    ClaimVerificationResult,
    ConfidenceLevel,
    ContinuousPromptingOutput,
    DomainRole,
    GenericExtraction,
    GroundingOutput,
    JudgeEvaluation,
    ResumeExtraction,
    SeniorityLevel,
    StructuredSelfAssessmentOutput,
    VerbalizedConfidenceOutput,
)


def create_mock_gemini_client() -> MagicMock:
    """Create a mock Google GenAI Client dynamically adapting to any requested response_schema."""
    client = MagicMock()

    def mock_generate_content(model, contents, config=None, **kwargs):
        resp = MagicMock()
        schema = getattr(config, "response_schema", None) if config else None
        schema_name = getattr(schema, "__name__", "") if schema else ""

        if schema_name == "VerbalizedConfidenceOutput":
            obj = VerbalizedConfidenceOutput(
                reasoning="Extracted accurately based on text analysis.",
                extraction_data={"candidate_name": "Jane Doe", "domain_role": "Data Science"},
                verbalized_confidence_score=0.88,
            )
        elif schema_name == "StructuredSelfAssessmentOutput":
            obj = StructuredSelfAssessmentOutput(
                rationale="Chain-of-thought analysis demonstrates high accuracy.",
                extraction_data={"candidate_name": "Jane Doe", "domain_role": "Data Science"},
                confidence_level=ConfidenceLevel.HIGH,
                numerical_confidence=0.85,
            )
        elif schema_name == "ContinuousPromptingOutput":

            obj = ContinuousPromptingOutput(
                rationale="Detailed chain of thought evaluation shows high certainty.",
                extraction_data={"candidate_name": "Jane Doe", "domain_role": "Data Science"},
                confidence_rating_0_to_100=85.0,
            )
        elif schema_name == "GroundingOutput":
            obj = GroundingOutput(
                claims=[
                    ClaimVerificationResult(
                        claim="Senior Data Scientist with 6 years experience",
                        verdict="SUPPORTED",
                        evidence_snippet="Senior Data Scientist with 6 years of experience",
                    )
                ],
                grounding_score=1.0,
            )
        elif schema_name == "JudgeEvaluation":
            obj = JudgeEvaluation(
                precision_score=4.5,
                hallucination_score=4.8,
                completeness_score=4.2,
                justification="Extraction is precise, accurate, and complete.",
                normalized_score=0.90,
            )
        elif schema_name == "GenericExtraction":
            obj = GenericExtraction(
                primary_category="Data Science",
                confidence_estimate=0.9,
                extracted_fields={"candidate_name": "Jane Doe"},
                key_claims=["Senior Data Scientist with 6 years experience"],
                summary="Sample summary",
            )
        elif schema is not None and issubclass(schema, BaseModel):
            try:
                obj = schema()
            except Exception:
                obj = ResumeExtraction(
                    candidate_name="Jane Doe",
                    domain_role=DomainRole.DATA_SCIENCE,
                    seniority_level=SeniorityLevel.SENIOR,
                    years_of_experience=6.0,
                    key_skills=["Python", "PyTorch"],
                    key_claims=["Senior Data Scientist with 6 years experience"],
                )
        else:
            obj = ResumeExtraction(
                candidate_name="Jane Doe",
                domain_role=DomainRole.DATA_SCIENCE,
                seniority_level=SeniorityLevel.SENIOR,
                years_of_experience=6.0,
                key_skills=["Python", "PyTorch"],
                key_claims=["Senior Data Scientist with 6 years experience"],
            )

        resp.text = obj.model_dump_json()

        # Logprobs mock
        logprob1 = MagicMock()
        logprob1.log_probability = -0.10
        top_cand1 = MagicMock()
        top_cand1.log_probability = -0.10
        top_cand2 = MagicMock()
        top_cand2.log_probability = -2.50
        logprob1.top_candidates = [top_cand1, top_cand2]

        logprob2 = MagicMock()
        logprob2.log_probability = -0.15
        logprob2.top_candidates = [top_cand1, top_cand2]

        mock_candidate = MagicMock()
        mock_candidate.logprobs_result.chosen_candidates = [logprob1, logprob2]
        resp.candidates = [mock_candidate]
        return resp

    client.models.generate_content.side_effect = mock_generate_content
    return client


@pytest.fixture
def mock_gemini_client() -> MagicMock:
    """Pytest fixture wrapper for mock Gemini Client."""
    return create_mock_gemini_client()


@pytest.fixture
def sample_resume_text() -> str:
    """Sample resume text for tests."""
    return """
    Jane Doe
    Email: jane.doe@example.com | Phone: (555) 123-4567

    SUMMARY
    Senior Data Scientist with 6 years of experience building machine learning models in Python, PyTorch, and SQL.
    Led predictive analytics and AI deployment pipelines on AWS cloud infrastructure.

    EXPERIENCE
    Lead AI Engineer - Tech Corp (2021 - Present)
    - Developed deep learning models achieving 94% precision.
    - Managed team of 4 data scientists and machine learning engineers.

    EDUCATION
    M.S. in Computer Science - Tech University (2019)

    SKILLS
    Python, SQL, PyTorch, Machine Learning, AWS, Docker, Scikit-Learn
    """


@pytest.fixture
def sample_pdf_file(tmp_path: Path) -> Path:
    """Create a temporary PDF file with sample text using PyMuPDF."""
    pdf_path = tmp_path / "sample_resume.pdf"
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "John Smith - Software Engineer\n"
        "5 Years experience in Python, Django, PostgreSQL, and React.\n"
        "Built microservices and REST APIs."
    )
    page.insert_text((50, 50), text)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_ground_truth() -> ResumeExtraction:
    """Sample ground truth ResumeExtraction fixture."""
    return ResumeExtraction(
        candidate_name="Jane Doe",
        domain_role=DomainRole.DATA_SCIENCE,
        seniority_level=SeniorityLevel.SENIOR,
        years_of_experience=6.0,
        key_skills=["Python", "PyTorch", "SQL", "Machine Learning", "AWS"],
        key_claims=[
            "Senior Data Scientist with 6 years of experience",
            "Managed team of 4 data scientists",
            "M.S. in Computer Science from Tech University",
        ],
        raw_text_summary="Experienced Senior Data Scientist.",
    )
