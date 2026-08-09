"""
Pytest configuration and shared fixtures for test suite.
"""

from pathlib import Path
import tempfile
import pymupdf as fitz  # PyMuPDF
import pytest
from unittest.mock import MagicMock

from src.schema import DomainRole, ResumeExtraction, SeniorityLevel


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


@pytest.fixture
def mock_gemini_client() -> MagicMock:
    """Mock Google GenAI Client for offline deterministic testing."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ResumeExtraction(
        candidate_name="Jane Doe",
        domain_role=DomainRole.DATA_SCIENCE,
        seniority_level=SeniorityLevel.SENIOR,
        years_of_experience=6.0,
        key_skills=["Python", "PyTorch"],
        key_claims=["Senior Data Scientist with 6 years experience"],
    ).model_dump_json()

    # Logprob mock
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
    mock_response.candidates = [mock_candidate]

    client.models.generate_content.return_value = mock_response
    return client
