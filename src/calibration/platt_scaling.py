"""
4. Platt Scaling Logistic Engine.
Trains a 1D logistic regression model mapping logprob delta to empirical correctness.
Domain-agnostic with configurable Pydantic response schemas.
"""

from enum import Enum
import math
import time
from typing import Any, Optional, Type
from google import genai
from google.genai import types
import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

from src.calibration.base import BaseConfidenceEngine
from src.exceptions import ExtractionValidationError, LogProbsUnavailableError
from src.schema import CalibrationResult, GenericExtraction


class PlattScalingEngine(BaseConfidenceEngine):
    """Engine 4: Platt Scaling Logistic Engine."""

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
        self.platt_model = LogisticRegression()
        self.is_fitted = False

    def fit(self, val_deltas: np.ndarray, val_correctness: np.ndarray) -> None:
        """Fit 1D Logistic Regression mapping deltas (or raw confidences) to empirical correctness."""
        X = val_deltas.reshape(-1, 1)
        y = val_correctness.astype(int)
        self.platt_model.fit(X, y)
        self.is_fitted = True
        self.logger.info("Fitted Platt Scaling Logistic Regression model.")

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

        is_enum = isinstance(schema, type) and issubclass(schema, Enum)
        mime_type = "text/x.enum" if is_enum else "application/json"

        config = types.GenerateContentConfig(
            response_mime_type=mime_type,
            response_schema=schema,
            response_logprobs=True,
            logprobs=5,
            temperature=0.0,
        )

        contents = [eval_prompt]
        if pdf_bytes:
            contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        self.logger.info("Executing %s with schema %s on model %s", self.__class__.__name__, schema.__name__, self.model_name)
        try:
            response = self._generate_content_with_retry(
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e)
            if "Logprobs is not enabled" in err_str or "INVALID_ARGUMENT" in err_str or "logprobs" in err_str.lower():
                raise LogProbsUnavailableError(
                    self.__class__.__name__,
                    f"Logprobs are not enabled for model '{self.model_name}'. Please set GEMINI_MODEL to 'gemini-2.5-flash'. Original error: {err_str}"
                )
            raise e

        if not response.text:
            raise ExtractionValidationError(self.__class__.__name__, "", "Empty API response text")

        try:
            extraction = schema.model_validate_json(response.text)
        except Exception as e:
            raise ExtractionValidationError(self.__class__.__name__, response.text, str(e))

        deltas = []
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "logprobs_result") and candidate.logprobs_result:
                    logprobs_res = candidate.logprobs_result

                    # Primary: check logprobs_result.top_candidates (google-genai SDK format)
                    top_cands_list = getattr(logprobs_res, "top_candidates", [])
                    if top_cands_list:
                        for top_cand_group in top_cands_list:
                            cands = getattr(top_cand_group, "candidates", getattr(top_cand_group, "top_candidates", []))
                            if isinstance(cands, list) and len(cands) >= 2:
                                p1 = getattr(cands[0], "log_probability", None)
                                p2 = getattr(cands[1], "log_probability", None)
                                if p1 is not None and p2 is not None:
                                    p1_val = max(p1, -100.0)
                                    p2_val = max(p2, -100.0)
                                    deltas.append(abs(p1_val - p2_val))

                    # Fallback: check chosen_candidates[i].top_candidates (legacy/mock format)
                    if not deltas:
                        chosen = getattr(logprobs_res, "chosen_candidates", [])
                        for chosen_cand in chosen:
                            top_candidates = getattr(chosen_cand, "top_candidates", [])
                            if isinstance(top_candidates, list) and len(top_candidates) >= 2:
                                p1 = getattr(top_candidates[0], "log_probability", None)
                                p2 = getattr(top_candidates[1], "log_probability", None)
                                if p1 is not None and p2 is not None:
                                    p1_val = max(p1, -100.0)
                                    p2_val = max(p2, -100.0)
                                    deltas.append(abs(p1_val - p2_val))
        except Exception as e:
            self.logger.warning("Failed to extract logprob top-2 deltas: %s", str(e))

        if not deltas:
            raise LogProbsUnavailableError(
                self.__class__.__name__,
                f"Model '{self.model_name}' did not return valid top-2 token alternative logprobs."
            )

        mean_delta = float(sum(deltas) / len(deltas))
        raw_confidence = float(1.0 / (1.0 + math.exp(-mean_delta)))

        if self.is_fitted:
            prob_true = self.platt_model.predict_proba(np.array([[mean_delta]]))[0, 1]
            calibrated_confidence = float(prob_true)
        else:
            # Default un-fitted Platt curve using standard default slope A=1.2, B=-0.5
            calibrated_confidence = float(1.0 / (1.0 + math.exp(-(1.2 * mean_delta - 0.5))))

        calibrated_confidence = float(max(0.0, min(1.0, calibrated_confidence)))

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        slope = float(self.platt_model.coef_[0][0]) if self.is_fitted else 1.2
        intercept = float(self.platt_model.intercept_[0]) if self.is_fitted else -0.5

        return CalibrationResult(
            engine_name="Platt Scaling Logistic",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "mean_delta": mean_delta,
                "is_fitted": self.is_fitted,
                "platt_slope": slope,
                "platt_intercept": intercept,
                "target_schema": schema.__name__,
            },
        )
