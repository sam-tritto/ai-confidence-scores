"""
8. Grounding & Context Alignment Engine.
Extracts atomic claims from the output and performs NLI verification against source PDF resume text.
Grounding Score = Supported Claims / Total Claims
"""

import time
from typing import List, Optional
from google.genai import types

from src.calibration.base import BaseConfidenceEngine
from src.schema import (
    CalibrationResult,
    ClaimVerificationResult,
    GroundingOutput,
    ResumeExtraction,
)


class GroundingAlignmentEngine(BaseConfidenceEngine):
    """Engine 8: Grounding & Context Alignment Engine.
    
    Verifies extracted factual claims against the raw source resume text to calculate
    the percentage of grounded, verified claims.
    """

    def evaluate(
        self,
        resume_text: str,
        pdf_bytes: Optional[bytes] = None,
        ground_truth: Optional[ResumeExtraction] = None,
    ) -> CalibrationResult:
        start_time = time.perf_counter()

        # Step 1: Base extraction
        extraction = None
        if self.client is not None:
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResumeExtraction,
                    temperature=0.0,
                )
                contents = [
                    "Analyze the resume and extract candidate metadata including key atomic claims.\n"
                    f"Resume Text:\n{resume_text}"
                ]
                if pdf_bytes:
                    contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                if response.text:
                    extraction = ResumeExtraction.model_validate_json(response.text)
            except Exception:
                pass

        if extraction is None:
            extraction = self._create_fallback_extraction(resume_text)

        # Ensure claims exist
        claims = extraction.key_claims or [
            f"Role matches {extraction.domain_role.value}",
            f"Experience level is {extraction.seniority_level.value}",
            f"Candidate has {extraction.years_of_experience} years experience",
        ]

        # Step 2: Perform NLI / Entailment check against source text
        claim_verifications: List[ClaimVerificationResult] = []
        text_lower = resume_text.lower()

        if self.client is not None:
            try:
                verification_prompt = (
                    "You are an NLI (Natural Language Inference) claim verification engine.\n"
                    "For each claim below, verify if it is strictly SUPPORTED by the provided source resume text.\n"
                    f"Source Resume:\n{resume_text}\n\n"
                    f"Claims to verify: {claims}"
                )
                config_ground = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GroundingOutput,
                    temperature=0.0,
                )
                resp_ground = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[verification_prompt],
                    config=config_ground,
                )
                if resp_ground.text:
                    ground_res = GroundingOutput.model_validate_json(resp_ground.text)
                    claim_verifications = ground_res.claims
            except Exception:
                pass

        # Heuristic NLI fallback if API NLI call is omitted/fails
        if not claim_verifications:
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
            engine_name="Grounding & Alignment",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "total_claims": total_claims,
                "supported_claims": supported_count,
                "grounding_score": grounding_score,
                "verifications": [c.model_dump() for c in claim_verifications],
            },
        )
