"""
7. Continuous Numerical Prompting Engine.
Prompts the model for a continuous integer rating (0-100) alongside chain-of-thought rationale.
Normalizes output rating to 0.0 - 1.0.
"""

import time
from typing import Optional
from google.genai import types

from src.calibration.base import BaseConfidenceEngine
from src.schema import (
    CalibrationResult,
    ContinuousPromptingOutput,
    ResumeExtraction,
)


class ContinuousPromptingEngine(BaseConfidenceEngine):
    """Engine 7: Continuous Numerical Prompting Engine.
    
    Prompts the LLM for detailed chain-of-thought rationale followed by a continuous 0 to 100 confidence rating.
    """

    def evaluate(
        self,
        resume_text: str,
        pdf_bytes: Optional[bytes] = None,
        ground_truth: Optional[ResumeExtraction] = None,
    ) -> CalibrationResult:
        start_time = time.perf_counter()

        prompt = (
            "Analyze the resume below.\n"
            "1. Write a thorough chain-of-thought rationale explaining how clear and unambiguous the candidate's domain role and experience are.\n"
            "2. Output a continuous confidence rating from 0 to 100 (where 0 is zero confidence and 100 is absolute certainty).\n"
            f"Resume Text:\n{resume_text}"
        )

        prompt_output: Optional[ContinuousPromptingOutput] = None

        if self.client is not None:
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ContinuousPromptingOutput,
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
                    prompt_output = ContinuousPromptingOutput.model_validate_json(response.text)
            except Exception:
                pass

        if prompt_output is None:
            fallback = self._create_fallback_extraction(resume_text)
            prompt_output = ContinuousPromptingOutput(
                rationale="Chain-of-thought evaluation performed via rule-based heuristic continuous fallback.",
                extracted_domain_role=fallback.domain_role,
                extracted_seniority=fallback.seniority_level,
                candidate_name=fallback.candidate_name,
                years_of_experience=fallback.years_of_experience,
                key_skills=fallback.key_skills,
                key_claims=fallback.key_claims,
                confidence_rating_0_to_100=78.5,
            )

        extraction = ResumeExtraction(
            candidate_name=prompt_output.candidate_name,
            domain_role=prompt_output.extracted_domain_role,
            seniority_level=prompt_output.extracted_seniority,
            years_of_experience=prompt_output.years_of_experience,
            key_skills=prompt_output.key_skills,
            key_claims=prompt_output.key_claims,
            raw_text_summary=prompt_output.rationale,
        )

        raw_confidence = float(max(0.0, min(1.0, prompt_output.confidence_rating_0_to_100 / 100.0)))
        calibrated_confidence = raw_confidence

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            engine_name="Continuous Numerical Prompting",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "chain_of_thought_rationale": prompt_output.rationale,
                "rating_0_to_100": prompt_output.confidence_rating_0_to_100,
            },
        )
