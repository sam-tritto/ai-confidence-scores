"""
3. Post-Hoc Temperature Scaling Engine.
Fits a temperature parameter T > 1 on raw logprobs using Negative Log-Likelihood (NLL) optimization.
P_calibrated(i) = exp(z_i / T) / sum(exp(z_j / T))
"""

import math
import time
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from google.genai import types

from src.calibration.base import BaseConfidenceEngine
from src.schema import CalibrationResult, ResumeExtraction


class TemperatureScalingEngine(BaseConfidenceEngine):
    """Engine 3: Post-Hoc Temperature Scaling Engine.
    
    Rescales logit/logprob distribution using temperature parameter T fitted via NLL optimization.
    """

    def __init__(self, client=None, model_name: str = "gemini-2.5-flash", audit_threshold: float = 0.75, temperature: float = 1.35):
        super().__init__(client=client, model_name=model_name, audit_threshold=audit_threshold)
        self.temperature = temperature
        self.is_fitted = False

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """Fit optimal temperature parameter T on validation logits and target ground-truth binary labels."""
        def nll_loss(t: float) -> float:
            t = max(t[0], 0.01)
            scaled_logits = val_logits / t
            # Log Softmax / Sigmoid cross-entropy
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(val_labels * np.log(probs) + (1 - val_labels) * np.log(1 - probs))
            return float(loss)

        res = minimize(nll_loss, x0=[1.35], bounds=[(0.1, 5.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        self.is_fitted = True
        return self.temperature

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
        raw_logprob = -0.25

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
                    lps = [getattr(t, "log_probability", -0.2) for t in chosen_tokens if getattr(t, "log_probability", None) is not None]
                    if lps:
                        raw_logprob = sum(lps) / len(lps)
            except Exception:
                pass

        if extraction is None:
            extraction = self._create_fallback_extraction(resume_text)

        raw_confidence = math.exp(raw_logprob)
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # Temperature Scaling on Logit z = log(raw_confidence / (1 - raw_confidence))
        logit = math.log(max(raw_confidence, 1e-5) / max(1.0 - raw_confidence, 1e-5))
        calibrated_logit = logit / self.temperature
        calibrated_confidence = 1.0 / (1.0 + math.exp(-calibrated_logit))
        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            engine_name="Temperature Scaling",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "fitted_temperature": self.temperature,
                "is_fitted": self.is_fitted,
                "raw_logit": logit,
                "calibrated_logit": calibrated_logit,
            },
        )
