"""
6. Structured Self-Assessment Engine (Verbalized Confidence).
Enforces a JSON schema via Pydantic using response_schema and response_mime_type="application/json".
Requests explicit reasoning, extracted_class, and verbalized_confidence_score (0.0-1.0).
"""

import time
from typing import Optional
from google.genai import types

from src.calibration.base import BaseConfidenceEngine
from src.schema import (
    CalibrationResult,
    ResumeExtraction,
    VerbalizedConfidenceOutput,
)


class VerbalizedConfidenceEngine(BaseConfidenceEngine):
    """Engine 6: Structured Self-Assessment Engine (Verbalized Confidence).
    
    Prompts the LLM to output its structured extraction along with step-by-step reasoning
    and an explicit self-assessed verbalized confidence score (0.0 to 1.0).
    """

    def evaluate(
        self,
        resume_text: str,
        pdf_bytes: Optional[bytes] = None,
        ground_truth: Optional[ResumeExtraction] = None,
    ) -> CalibrationResult:
        start_time = time.perf_counter()

        prompt = (
            "Analyze the following resume. Extract candidate information and provide step-by-step reasoning.\n"
            "Assess your overall confidence in this extraction on a continuous scale from 0.0 (completely uncertain) to 1.0 (completely certain).\n"
            f"Resume Text:\n{resume_text}"
        )

        verbalized_output: Optional[VerbalizedConfidenceOutput] = None

        if self.client is not None:
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VerbalizedConfidenceOutput,
                    temperature=0.0,
                )
                contents = [prompt]
                if pdf_bytes:
                    contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                if response.text:
                    verbalized_output = VerbalizedConfidenceOutput.model_validate_json(response.text)
            except Exception:
                pass

        if verbalized_output is None:
            fallback = self._create_fallback_extraction(resume_text)
            verbalized_output = VerbalizedConfidenceOutput(
                reasoning="Extracted using rule-based heuristic parser fallback.",
                candidate_name=fallback.candidate_name,
                domain_role=fallback.domain_role,
                seniority_level=fallback.seniority_level,
                years_of_experience=fallback.years_of_experience,
                key_skills=fallback.key_skills,
                key_claims=fallback.key_claims,
                verbalized_confidence_score=0.72,
            )

        extraction = ResumeExtraction(
            candidate_name=verbalized_output.candidate_name,
            domain_role=verbalized_output.domain_role,
            seniority_level=verbalized_output.seniority_level,
            years_of_experience=verbalized_output.years_of_experience,
            key_skills=verbalized_output.key_skills,
            key_claims=verbalized_output.key_claims,
            raw_text_summary=verbalized_output.reasoning,
        )

        raw_confidence = float(max(0.0, min(1.0, verbalized_output.verbalized_confidence_score)))
        calibrated_confidence = raw_confidence

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            engine_name="Structured Verbalized Confidence",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "reasoning": verbalized_output.reasoning,
                "verbalized_score": verbalized_output.verbalized_confidence_score,
            },
        )
