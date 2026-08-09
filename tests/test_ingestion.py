"""
Unit tests for ResumeIngestor PDF processing.
"""

from pathlib import Path
from src.ingestion import ResumeIngestor


def test_scan_resumes(tmp_path: Path, sample_pdf_file: Path):
    ingestor = ResumeIngestor(data_directory=tmp_path)
    found_pdfs = ingestor.scan_resumes()
    assert len(found_pdfs) == 1
    assert found_pdfs[0] == sample_pdf_file


def test_extract_text_pymupdf(sample_pdf_file: Path):
    text = ResumeIngestor.extract_text_pymupdf(sample_pdf_file)
    assert "John Smith" in text
    assert "Software Engineer" in text
    assert "Python" in text


def test_load_pdf_bytes(sample_pdf_file: Path):
    pdf_bytes = ResumeIngestor.load_pdf_bytes(sample_pdf_file)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")
