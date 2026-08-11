"""
3. Post-Hoc Temperature Scaling Engine.
Fits temperature parameter T > 1 on logit/logprob values via NLL optimization.
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
from scipy.optimize import minimize

from src.calibration.base import BaseConfidenceEngine
from src.exceptions import ExtractionValidationError, LogProbsUnavailableError
from src.schema import CalibrationResult, GenericExtraction


class TemperatureScalingEngine(BaseConfidenceEngine):
    """Engine 3: Post-Hoc Temperature Scaling Engine."""

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: Optional[str] = None,
        audit_threshold: Optional[float] = None,
        temperature: float = 1.2,
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
        self.temperature = temperature
        self.is_fitted = False

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """Fit optimal temperature parameter T > 0 on validation split using NLL loss."""
        def nll_loss(t: float) -> float:
            scaled = val_logits / max(t[0], 0.01)
            probs = 1.0 / (1.0 + np.exp(-scaled))
            eps = 1e-12
            probs = np.clip(probs, eps, 1.0 - eps)
            loss = -np.mean(val_labels * np.log(probs) + (1 - val_labels) * np.log(1 - probs))
            return float(loss)

        res = minimize(nll_loss, x0=[self.temperature], bounds=[(0.01, 10.0)])
        self.temperature = float(res.x[0])
        self.is_fitted = True
        self.logger.info("Fitted optimal Temperature parameter T = %.4f", self.temperature)
        return self.temperature

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

        logprob_values = []
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "logprobs_result") and candidate.logprobs_result:
                    chosen = getattr(candidate.logprobs_result, "chosen_candidates", [])
                    for top_cand in chosen:
                        if hasattr(top_cand, "log_probability") and top_cand.log_probability is not None:
                            logprob_values.append(top_cand.log_probability)
        except Exception as e:
            self.logger.warning("Failed to extract logprob values from candidates: %s", str(e))

        if not logprob_values:
            raise LogProbsUnavailableError(self.__class__.__name__, f"Model '{self.model_name}' did not return valid token logprobabilities.")

        mean_logprob = float(sum(logprob_values) / len(logprob_values))
        raw_confidence = float(math.exp(mean_logprob))

        # Temperature scaling on logit representation
        raw_logit = mean_logprob
        scaled_logit = raw_logit / max(self.temperature, 0.01)
        calibrated_confidence = float(math.exp(scaled_logit))
        calibrated_confidence = float(max(0.0, min(1.0, calibrated_confidence)))

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
                "temperature": self.temperature,
                "is_fitted": self.is_fitted,
                "mean_logprob": mean_logprob,
                "target_schema": schema.__name__,
            },
        )
