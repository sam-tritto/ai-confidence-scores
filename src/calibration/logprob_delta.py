"""
2. LogProb Delta Engine.
Isolates top-token dominance margin: Delta = top_1_logprob - top_2_logprob.
Domain-agnostic with configurable Pydantic response schemas.
"""

import math
import time
from typing import Any, Optional, Type
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.calibration.base import BaseConfidenceEngine
from src.exceptions import ExtractionValidationError, LogProbsUnavailableError
from src.schema import CalibrationResult, GenericExtraction


class LogProbDeltaEngine(BaseConfidenceEngine):
    """Engine 2: LogProb Delta Engine."""

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
            response_logprobs=True,
            logprobs=5,
            temperature=0.0,
        )

        contents = [eval_prompt]
        if pdf_bytes:
            contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        self.logger.info("Executing %s with schema %s on model %s", self.__class__.__name__, schema.__name__, self.model_name)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e)
            if "Logprobs is not enabled" in err_str or "INVALID_ARGUMENT" in err_str or "logprobs" in err_str.lower():
                raise LogProbsUnavailableError(
                    self.__class__.__name__,
                    f"Logprobs are not enabled for model '{self.model_name}'. Please switch GEMINI_MODEL to a supported model such as 'gemini-2.0-flash' or 'gemini-1.5-flash'. Original error: {err_str}"
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
                    chosen = getattr(candidate.logprobs_result, "chosen_candidates", [])
                    for chosen_cand in chosen:
                        top_candidates = getattr(chosen_cand, "top_candidates", [])
                        if len(top_candidates) >= 2:
                            p1 = top_candidates[0].log_probability
                            p2 = top_candidates[1].log_probability
                            if p1 is not None and p2 is not None:
                                deltas.append(abs(p1 - p2))
        except Exception as e:
            self.logger.warning("Failed to extract logprob top-2 deltas: %s", str(e))

        if not deltas:
            raise LogProbsUnavailableError(self.__class__.__name__, f"Model '{self.model_name}' did not return valid top-2 token alternative logprobs.")

        mean_delta = float(sum(deltas) / len(deltas))
        # Logistic sigmoid mapping of delta: 1 / (1 + exp(-delta))
        calibrated_confidence = float(1.0 / (1.0 + math.exp(-mean_delta)))
        raw_confidence = calibrated_confidence

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
                "mean_delta": mean_delta,
                "deltas_count": len(deltas),
                "target_schema": schema.__name__,
            },
        )
