"""
5. Grounding & Context Alignment Method.
Extracts atomic claims from output and performs NLI verification against source text.
Domain-agnostic with configurable Pydantic response schemas.
"""

import time
from typing import Any, List, Optional, Type
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.calibration.base import BaseConfidenceMethod
from src.exceptions import ExtractionValidationError
from src.schema import (
    CalibrationResult,
    ClaimVerificationResult,
    GenericExtraction,
    GroundingOutput,
)


class GroundingAlignmentMethod(BaseConfidenceMethod):
    """Method 5: Grounding & Context Alignment Method."""

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
            f"Analyze the following text and extract structured information into JSON format.\n"
            f"Input Text:\n{input_text}"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        )

        contents = [eval_prompt]
        if pdf_bytes:
            contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        self.logger.info("Executing %s with schema %s", self.__class__.__name__, schema.__name__)
        response = self._generate_content_with_retry(
            contents=contents,
            config=config,
        )

        if not response.text:
            raise ExtractionValidationError(self.__class__.__name__, "", "Empty API response text")

        try:
            extraction = schema.model_validate_json(response.text)
        except Exception as e:
            raise ExtractionValidationError(self.__class__.__name__, response.text, str(e))

        # Extract atomic claims
        claims = getattr(extraction, "key_claims", [])
        if not claims:
            claims = [f"Input contains valid {schema.__name__} details."]

        # Step 2: NLI verification
        claim_verifications: List[ClaimVerificationResult] = []
        try:
            verification_prompt = (
                "You are an NLI (Natural Language Inference) claim verification engine.\n"
                "For each claim below, verify if it is strictly SUPPORTED by the provided source text.\n"
                f"Source Text:\n{input_text}\n\n"
                f"Claims to verify: {claims}"
            )
            config_ground = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GroundingOutput,
                temperature=0.0,
            )
            resp_ground = self._generate_content_with_retry(
                contents=[verification_prompt],
                config=config_ground,
            )
            if resp_ground.text:
                ground_res = GroundingOutput.model_validate_json(resp_ground.text)
                claim_verifications = ground_res.claims
        except Exception as e:
            self.logger.warning("NLI claim verification API step failed: %s", str(e))

        if not claim_verifications:
            text_lower = input_text.lower()
            for claim in claims:
                claim_words = [w.lower() for w in claim.split() if len(w) > 3]
                match_count = sum(1 for w in claim_words if w in text_lower)
                is_supported = match_count > 0 or len(claim_words) == 0
                verdict = "SUPPORTED" if is_supported else "UNVERIFIABLE"
                claim_verifications.append(
                    ClaimVerificationResult(
                        claim=claim,
                        verdict=verdict,
                        evidence_snippet=claim if is_supported else None,
                    )
                )

        supported_count = sum(1 for c in claim_verifications if c.verdict == "SUPPORTED")
        total_claims = len(claim_verifications)
        grounding_score = supported_count / float(total_claims) if total_claims > 0 else 1.0

        raw_confidence = grounding_score
        calibrated_confidence = grounding_score

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            method_name="Grounding & Alignment",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "total_claims": total_claims,
                "supported_claims": supported_count,
                "grounding_score": grounding_score,
                "target_schema": schema.__name__,
            },
        )
