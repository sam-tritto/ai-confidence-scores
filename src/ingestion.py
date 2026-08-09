"""
PDF Resume Ingestion Pipeline using PyMuPDF (fitz) and Gemini native multimodal ingestion.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import pymupdf as fitz  # PyMuPDF


class ResumeIngestor:
    """Handles scanning, loading, and text layout extraction for PDF resumes."""

    def __init__(self, data_directory: Union[str, Path] = "./data"):
        self.data_directory = Path(data_directory)

    def scan_resumes(self) -> List[Path]:
        """Scan data directory recursively for all PDF resume files."""
        if not self.data_directory.exists():
            return []
        
        pdf_files = sorted(list(self.data_directory.glob("*.pdf")) + list(self.data_directory.glob("**/*.pdf")))
        # Remove duplicates while preserving order
        seen = set()
        unique_pdfs = []
        for pdf in pdf_files:
            if pdf.resolve() not in seen:
                seen.add(pdf.resolve())
                unique_pdfs.append(pdf)
        return unique_pdfs

    @staticmethod
    def extract_text_pymupdf(pdf_path: Union[str, Path]) -> str:
        """Extract layout-preserved text from PDF using PyMuPDF (fitz)."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        extracted_text = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{text}")

        full_text = "\n\n".join(extracted_text).strip()
        if not full_text:
            return f"[EMPTY PDF: {pdf_path.name}]"
        return full_text

    @staticmethod
    def load_pdf_bytes(pdf_path: Union[str, Path]) -> bytes:
        """Load raw PDF bytes for native Gemini multimodal ingestion."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        return pdf_path.read_bytes()

    def process_pdf(self, pdf_path: Union[str, Path]) -> Tuple[str, bytes]:
        """Extract both PyMuPDF text and raw bytes for a PDF resume."""
        text = self.extract_text_pymupdf(pdf_path)
        pdf_bytes = self.load_pdf_bytes(pdf_path)
        return text, pdf_bytes
