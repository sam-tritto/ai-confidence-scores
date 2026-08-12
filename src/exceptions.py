"""
Custom Exception Classes for AI Confidence Scores Calibration Methods.
"""

from typing import Optional


class ConfidenceMethodError(Exception):
    """Base exception class for all confidence calibration method errors."""
    pass


class ClientNotConfiguredError(ConfidenceMethodError):
    """Raised when a method is executed without an active Gemini client or API key."""
    def __init__(self, method_name: str):
        super().__init__(
            f"[{method_name}] Gemini API client is not configured. "
            "Pass an active genai.Client instance or set the GEMINI_API_KEY environment variable."
        )


class LogProbsUnavailableError(ConfidenceMethodError):
    """Raised when token log probabilities are missing or unsupported for the model."""
    def __init__(self, method_name: str, message: Optional[str] = None):
        msg = message or (
            f"Token log probabilities are missing from the model response. "
            "Ensure response_logprobs=True is supported and enabled in GenerateContentConfig."
        )
        if not msg.startswith(f"[{method_name}]"):
            msg = f"[{method_name}] {msg}"
        super().__init__(msg)


class ExtractionValidationError(ConfidenceMethodError):
    """Raised when LLM output text fails validation against the target Pydantic schema."""
    def __init__(self, method_name: str, raw_text: str, details: str):
        super().__init__(
            f"[{method_name}] Failed to validate model response against target schema. "
            f"Error: {details}\nRaw Output: {raw_text[:200]}"
        )


class JudgeEvaluationError(ConfidenceMethodError):
    """Raised when the secondary judge model fails to evaluate quality or rubric scores."""
    def __init__(self, method_name: str, details: str):
        super().__init__(
            f"[{method_name}] Secondary LLM Judge evaluation failed: {details}"
        )
