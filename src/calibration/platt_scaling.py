"""
4. Platt Scaling Logistic Engine.
Trains a 1D LogisticRegression model on LogProb Delta mapped against empirical correctness on a validation fold.
P(y=1|Delta) = 1 / (1 + exp(A * Delta + B))
"""

import math
import time
from typing import List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from google.genai import types

from src.calibration.base import BaseConfidenceEngine
from src.schema import CalibrationResult, ResumeExtraction


class PlattScalingEngine(BaseConfidenceEngine):
    """Engine 4: Platt Scaling Logistic Engine.
    
    Fits logistic regression model mapping logprob delta (or raw confidence) to empirical correctness.
    """

    def __init__(self, client=None, model_name: str = "gemini-2.5-flash", audit_threshold: float = 0.75):
        super().__init__(client=client, model_name=model_name, audit_threshold=audit_threshold)
        self.model = LogisticRegression()
        # Pre-fit default weights: A=1.5, B=-1.0 for uncalibrated baseline
        self.is_fitted = False
        self.slope = 1.5
        self.intercept = -1.0

    def fit(self, deltas: np.ndarray, y_true: np.ndarray) -> None:
        """Fit Platt scaling logistic model on validation deltas and empirical correctness ground truth."""
        X = np.array(deltas).reshape(-1, 1)
        y = np.array(y_true)
        self.model.fit(X, y)
        self.slope = float(self.model.coef_[0][0])
        self.intercept = float(self.model.intercept_[0])
        self.is_fitted = True

    def predict_calibrated_probability(self, delta: float) -> float:
        """Predict calibrated probability using fitted logistic parameters."""
        if self.is_fitted:
            prob = self.model.predict_proba([[delta]])[0][1]
        else:
            prob = 1.0 / (1.0 + math.exp(-(self.slope * delta + self.intercept)))
        return float(max(0.0, min(1.0, prob)))

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
        delta = 2.2
        raw_logprob = -0.2

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
                    deltas_list = []
                    for token_info in chosen_tokens:
                        top_candidates = getattr(token_info, "top_candidates", [])
                        if len(top_candidates) >= 2:
                            lp1 = getattr(top_candidates[0], "log_probability", -0.1)
                            lp2 = getattr(top_candidates[1], "log_probability", -2.5)
                            deltas_list.append(lp1 - lp2)
                    if deltas_list:
                        delta = sum(deltas_list) / len(deltas_list)
            except Exception:
                pass

        if extraction is None:
            extraction = self._create_fallback_extraction(resume_text)

        raw_confidence = 1.0 / (1.0 + math.exp(-delta))
        calibrated_confidence = self.predict_calibrated_probability(delta)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            engine_name="Platt Scaling Logistic",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "delta": delta,
                "platt_slope": self.slope,
                "platt_intercept": self.intercept,
                "is_fitted": self.is_fitted,
            },
        )
