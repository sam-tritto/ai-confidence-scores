"""
4. Continuous Numerical Prompting Method.
Prompts the model for a continuous integer rating (0-100) alongside chain-of-thought rationale.
Domain-agnostic with configurable Pydantic response schemas.
"""

import time
from typing import Any, Optional, Type
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.calibration.base import BaseConfidenceMethod
from src.exceptions import ExtractionValidationError
from src.schema import (
    CalibrationResult,
    ContinuousPromptingOutput,
    GenericExtraction,
)


class ContinuousPromptingMethod(BaseConfidenceMethod):
    """Method 4: Continuous Numerical Prompting Method."""

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
        super().__init__(
            client=client,
            model_name=model_name,
            audit_threshold=audit_threshold,
            use_vertex=use_vertex,
            project=project,
            location=location,
            api_key=api_key,
        )

    def evaluate(
        self,
        input_text: str,
        schema: Type[BaseModel] = GenericExtraction,
        prompt: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        ground_truth: Optional[Any] = None,
    ) -> CalibrationResult:
        self._ensure_client()
        start_time = time.perf_counter()

        eval_prompt = prompt or (
            f"Analyze the text below.\n"
            f"1. Write chain-of-thought rationale explaining your certainty.\n"
            f"2. Output a continuous confidence rating from 0 to 100 (where 0 is zero confidence and 100 is absolute certainty).\n"
            f"Input Text:\n{input_text}"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ContinuousPromptingOutput,
            temperature=0.0,
        )
        contents = [eval_prompt]
        if pdf_bytes:
            contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        self.logger.info("Executing %s with target schema %s", self.__class__.__name__, schema.__name__)
        response = self._generate_content_with_retry(
            contents=contents,
            config=config,
        )

        if not response.text:
            raise ExtractionValidationError(self.__class__.__name__, "", "Empty API response text")

        try:
            prompt_output = ContinuousPromptingOutput.model_validate_json(response.text)
        except Exception as e:
            raise ExtractionValidationError(self.__class__.__name__, response.text, str(e))

        # Attempt mapping extraction_data to target schema
        try:
            extraction = schema.model_validate(prompt_output.extraction_data)
        except Exception:
            try:
                extraction = schema.model_validate_json(response.text)
            except Exception:
                extraction = GenericExtraction(
                    primary_category="Extracted",
                    extracted_fields=prompt_output.extraction_data,
                    summary=prompt_output.rationale,
                )

        raw_confidence = float(max(0.0, min(1.0, prompt_output.confidence_rating_0_to_100 / 100.0)))
        calibrated_confidence = raw_confidence

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            method_name="Continuous Numerical Prompting",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "chain_of_thought_rationale": prompt_output.rationale,
                "rating_0_to_100": prompt_output.confidence_rating_0_to_100,
                "target_schema": schema.__name__,
            },
        )
