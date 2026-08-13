"""
10. Structured Self-Assessment + Platt Scaling Method.
Combines structured LLM self-assessment (categorical & continuous confidence ratings)
with post-hoc Platt Scaling (logistic regression calibration).
Fully logprob-independent, making it ideal for Gemini > 3.0 models where token logprobs are unavailable.
"""

import math
import time
from typing import Any, Optional, Tuple, Type
from google import genai
from google.genai import types
import numpy as np
from pydantic import BaseModel

from src.calibration.base import BaseConfidenceMethod
from src.exceptions import ExtractionValidationError
from src.schema import (
    CalibrationResult,
    ConfidenceLevel,
    GenericExtraction,
    StructuredSelfAssessmentOutput,
)


class StructuredSelfAssessmentPlattMethod(BaseConfidenceMethod):
    """Method 10: Structured Self-Assessment + Platt Scaling Method.

    First obtains a structured categorical and continuous confidence self-assessment from the LLM,
    converts the raw self-assessment probability into logit space (z = logit(p_raw)), and applies
    post-hoc Platt Scaling (p_calibrated = sigmoid(a * z + b)).
    """

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: Optional[str] = None,
        audit_threshold: Optional[float] = None,
        a: float = 1.0,
        b: float = 0.0,
        use_categorical: bool = True,
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
        self.a = a
        self.b = b
        self.use_categorical = use_categorical
        self.is_fitted = False

    @staticmethod
    def _prob_to_logit(prob: float, eps: float = 1e-6) -> float:
        """Convert probability p in (0, 1) to logit z = ln(p / (1-p))."""
        p_clipped = max(eps, min(1.0 - eps, prob))
        return float(math.log(p_clipped / (1.0 - p_clipped)))

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable logistic sigmoid function 1 / (1 + exp(-z))."""
        if z >= 0:
            return float(1.0 / (1.0 + math.exp(-z)))
        else:
            return float(math.exp(z) / (1.0 + math.exp(z)))

    def fit(self, val_scores: np.ndarray, val_labels: np.ndarray) -> Tuple[float, float]:
        """Fit 1D Platt Logistic parameters (a: slope, b: intercept) on validation dataset using NLL loss.

        Args:
            val_scores: Array of uncalibrated confidence probability scores in [0, 1].
            val_labels: Array of binary correctness labels (1 for correct extraction, 0 for incorrect).

        Returns:
            Tuple of fitted parameters (a, b).
        """
        from scipy.optimize import minimize

        eps = 1e-6
        clipped_scores = np.clip(val_scores, eps, 1.0 - eps)
        val_logits = np.log(clipped_scores / (1.0 - clipped_scores))

        def nll_loss(params: np.ndarray) -> float:
            a_param, b_param = params[0], params[1]
            scaled_logits = a_param * val_logits + b_param
            # Log-sum-exp trick for numerical stability
            log_p = np.where(scaled_logits >= 0, -np.log1p(np.exp(-scaled_logits)), scaled_logits - np.log1p(np.exp(scaled_logits)))
            log_1_minus_p = np.where(scaled_logits >= 0, -scaled_logits - np.log1p(np.exp(-scaled_logits)), -np.log1p(np.exp(scaled_logits)))

            loss = -np.mean(val_labels * log_p + (1.0 - val_labels) * log_1_minus_p)
            return float(loss)

        initial_params = np.array([self.a, self.b])
        res = minimize(nll_loss, x0=initial_params, method="L-BFGS-B")
        self.a = float(res.x[0])
        self.b = float(res.x[1])
        self.is_fitted = True
        self.logger.info("Fitted Platt Scaling parameters: a = %.4f, b = %.4f", self.a, self.b)
        return self.a, self.b

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
            f"1. Extract structured data according to schema fields.\n"
            f"2. Write step-by-step chain-of-thought rationale explaining your certainty.\n"
            f"3. Select a categorical confidence_level: [Very Low, Low, Medium, High, Very High].\n"
            f"4. Provide a numerical_confidence rating between 0.0 and 1.0.\n"
            f"Input Text:\n{input_text}"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StructuredSelfAssessmentOutput,
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
            assessment_output = StructuredSelfAssessmentOutput.model_validate_json(response.text)
        except Exception as e:
            raise ExtractionValidationError(self.__class__.__name__, response.text, str(e))

        # Attempt mapping extraction_data to target schema
        try:
            extraction = schema.model_validate(assessment_output.extraction_data)
        except Exception:
            try:
                extraction = schema.model_validate_json(response.text)
            except Exception:
                extraction = GenericExtraction(
                    primary_category="Extracted",
                    extracted_fields=assessment_output.extraction_data,
                    summary=assessment_output.rationale,
                )

        # Select raw confidence score based on preference (categorical vs continuous)
        if self.use_categorical and hasattr(assessment_output.confidence_level, "numeric_score"):
            raw_confidence = assessment_output.confidence_level.numeric_score
        else:
            raw_confidence = float(max(0.0, min(1.0, assessment_output.numerical_confidence)))

        # Convert raw probability to logit representation z = logit(raw_confidence)
        raw_logit = self._prob_to_logit(raw_confidence)

        # Apply Platt Scaling: calibrated_confidence = sigmoid(a * z + b)
        calibrated_logit = self.a * raw_logit + self.b
        calibrated_confidence = self._sigmoid(calibrated_logit)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        conf_level_str = (
            assessment_output.confidence_level.value
            if isinstance(assessment_output.confidence_level, ConfidenceLevel)
            else str(assessment_output.confidence_level)
        )

        return CalibrationResult(
            method_name="Structured Self-Assessment + Platt Scaling",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "chain_of_thought_rationale": assessment_output.rationale,
                "confidence_level": conf_level_str,
                "numerical_confidence": assessment_output.numerical_confidence,
                "raw_logit": raw_logit,
                "calibrated_logit": calibrated_logit,
                "platt_a": self.a,
                "platt_b": self.b,
                "is_fitted": self.is_fitted,
                "target_schema": schema.__name__,
            },
        )
