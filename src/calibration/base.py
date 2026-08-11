"""
Abstract Base Class for Confidence Evaluation Engines.
Domain-agnostic with configurable Pydantic response schemas and explicit error handling.
"""

from abc import ABC, abstractmethod
import logging
import os
import time
from typing import Any, Dict, Optional, Type
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

from src.exceptions import ClientNotConfiguredError
from src.schema import (
    AuditDecision,
    CalibrationResult,
    GenericExtraction,
)

# Load environment variables automatically from .env file
load_dotenv()


class BaseConfidenceEngine(ABC):
    """Abstract base class for LLM confidence calibration engines."""

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: Optional[str] = None,
        audit_threshold: Optional[float] = None,
        use_vertex: Optional[bool] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        load_dotenv()

        is_vertex = (
            use_vertex
            if use_vertex is not None
            else os.getenv("USE_VERTEX_AI", "false").lower() in ("true", "1")
        )
        self.use_vertex = is_vertex

        # Select provider-specific default model unless explicitly overridden
        if is_vertex:
            default_model = os.getenv("VERTEX_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
        else:
            default_model = os.getenv("API_KEY_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))

        self.model_name = model_name or default_model

        env_threshold = os.getenv("AUDIT_CONFIDENCE_THRESHOLD", "0.75")
        self.audit_threshold = float(audit_threshold if audit_threshold is not None else env_threshold)
        self.logger = logging.getLogger(self.__class__.__name__)

        if client is not None:
            self.client = client
        else:
            gcp_project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
            gcp_location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

            if self.use_vertex and gcp_project:
                self.client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
            elif key:
                self.client = genai.Client(api_key=key)
            else:
                self.client = None

    def _ensure_client(self) -> None:
        """Verify client configuration or raise explicit ClientNotConfiguredError."""
        if self.client is None:
            err = ClientNotConfiguredError(self.__class__.__name__)
            self.logger.error(str(err))
            raise err

    def determine_audit_decision(self, confidence_score: float) -> AuditDecision:
        """Map confidence score to AUTOMATE or FLAG_FOR_HUMAN_REVIEW."""
        if confidence_score >= self.audit_threshold:
            return AuditDecision.AUTOMATE
        return AuditDecision.FLAG_FOR_HUMAN_REVIEW

    @abstractmethod
    def evaluate(
        self,
        input_text: str,
        schema: Type[BaseModel] = GenericExtraction,
        prompt: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        ground_truth: Optional[Any] = None,
    ) -> CalibrationResult:
        """Evaluate input text against target schema and return calibrated confidence score."""
        pass
