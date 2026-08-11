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
            default_model = os.getenv("VERTEX_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        else:
            default_model = os.getenv("API_KEY_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

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

    def _generate_content_with_retry(
        self,
        contents: Any,
        config: Any,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        override_model: Optional[str] = None,
    ) -> Any:
        """Call client.models.generate_content with exponential backoff on 429 RESOURCE_EXHAUSTED errors."""
        self._ensure_client()
        target_model = override_model or self.model_name
        delay = 2.0
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()) and attempt < max_retries - 1:
                    self.logger.warning(
                        "Rate limit (429 RESOURCE_EXHAUSTED) hit on attempt %d/%d. Retrying in %.1fs...",
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    raise e

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
