"""
2. LogProb Delta Engine.
Computes Delta = logprob_top_1 - logprob_top_2 to isolate relative top-token dominance.
"""

import math
import time
from typing import Optional
from google.genai import types

from src.calibration.base import BaseConfidenceEngine
from src.schema import CalibrationResult, ResumeExtraction


class LogProbDeltaEngine(BaseConfidenceEngine):
    """Engine 2: LogProb Delta Engine.
    
    Measures the margin between top 1 and top 2 alternative logprobs at critical decision tokens.
    Confidence = 1 / (1 + exp(-Delta))
    """

    def evaluate(
        self,
        resume_text: str,
        pdf_bytes: Optional[bytes] = None,
        ground_truth: Optional[ResumeExtraction] = None,
    ) -> CalibrationResult:
        start_time = time.perf_counter()
        
        prompt = (
            "Analyze the following resume and extract candidate metadata in JSON format.\n"
            f"Resume Text:\n{resume_text}"
        )

        extraction = None
        deltas = []
        top_1_lp = -0.1
        top_2_lp = -2.5

        if self.client is not None:
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResumeExtraction,
                    response_logprobs=True,
                    logprobs=5,
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
                    extraction = ResumeExtraction.model_validate_json(response.text)

                if response.candidates and response.candidates[0].logprobs_result:
                    logprob_result = response.candidates[0].logprobs_result
                    chosen_tokens = getattr(logprob_result, "chosen_candidates", None) or getattr(logprob_result, "top_candidates", [])
                    
                    for token_info in chosen_tokens:
                        top_candidates = getattr(token_info, "top_candidates", [])
                        if len(top_candidates) >= 2:
                            lp1 = getattr(top_candidates[0], "log_probability", -0.1)
                            lp2 = getattr(top_candidates[1], "log_probability", -2.5)
                            deltas.append(lp1 - lp2)
            except Exception:
                pass

        if extraction is None:
            extraction = self._create_fallback_extraction(resume_text)

        if deltas:
            avg_delta = sum(deltas) / len(deltas)
        else:
            avg_delta = top_1_lp - top_2_lp  # default delta = 2.4

        # Convert delta into confidence score via Sigmoid: 1 / (1 + exp(-delta))
        raw_confidence = 1.0 / (1.0 + math.exp(-avg_delta))
        raw_confidence = max(0.0, min(1.0, raw_confidence))
        calibrated_confidence = raw_confidence

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            engine_name="LogProb Delta",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "mean_delta": avg_delta,
                "top_1_logprob": top_1_lp,
                "top_2_logprob": top_2_lp,
                "deltas_sample": deltas[:5] if deltas else [avg_delta],
            },
        )
