"""
9. Two-Tier Evaluator Method (LLM-as-a-Judge).
Uses a secondary judge instance with an evaluation rubric to score output quality.
Domain-agnostic with configurable Pydantic response schemas.
"""

import os
import time
from typing import Any, Optional, Type
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.calibration.base import BaseConfidenceMethod
from src.exceptions import ExtractionValidationError, JudgeEvaluationError
from src.schema import (
    CalibrationResult,
    GenericExtraction,
    JudgeEvaluation,
)


class LLMAsAJudgeMethod(BaseConfidenceMethod):
    """Method 9: Two-Tier Evaluator Method (LLM-as-a-Judge)."""

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: Optional[str] = None,
        judge_model_name: Optional[str] = None,
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
        if self.use_vertex:
            default_judge = os.getenv("VERTEX_LLM_JUDGE_MODEL", os.getenv("LLM_JUDGE_MODEL", self.model_name))
        else:
            default_judge = os.getenv("API_KEY_LLM_JUDGE_MODEL", os.getenv("LLM_JUDGE_MODEL", self.model_name))

        self.judge_model_name = judge_model_name or default_judge

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

        config_primary = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        )

        contents = [eval_prompt]
        if pdf_bytes:
            contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        self.logger.info("Executing %s with primary model %s and judge model %s", self.__class__.__name__, self.model_name, self.judge_model_name)
        response_primary = self._generate_content_with_retry(
            contents=contents,
            config=config_primary,
        )

        if not response_primary.text:
            raise ExtractionValidationError(self.__class__.__name__, "", "Empty API response from primary model")

        try:
            extraction = schema.model_validate_json(response_primary.text)
        except Exception as e:
            raise ExtractionValidationError(self.__class__.__name__, response_primary.text, str(e))

        # Step 2: Pass extraction to secondary LLM Judge
        judge_prompt = f"""
You are an expert impartial AI Audit Judge.
Evaluate the primary extraction quality against the raw source text.

[SOURCE TEXT]
{input_text}

[CANDIDATE EXTRACTION TO AUDIT]
{extraction.model_dump_json(indent=2)}

Scoring Rubric (Scale 1.0 to 5.0 for each):
1. Precision Score: Are extracted fields accurate to source text?
2. Hallucination Score: Is the extraction free of fabricated information? (5.0 = Zero Hallucination, 1.0 = Severe Hallucination)
3. Completeness Score: Does the extraction cover key information?

Provide scores and an explicit normalized quality score (0.0 to 1.0).
"""

        try:
            config_judge = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JudgeEvaluation,
                temperature=0.0,
            )
            response_judge = self._generate_content_with_retry(
                contents=[judge_prompt],
                config=config_judge,
                override_model=self.judge_model_name,
            )
            if not response_judge.text:
                raise JudgeEvaluationError(self.__class__.__name__, "Empty judge response text")
            judge_eval = JudgeEvaluation.model_validate_json(response_judge.text)
        except Exception as e:
            raise JudgeEvaluationError(self.__class__.__name__, str(e))

        raw_confidence = judge_eval.normalized_score
        calibrated_confidence = judge_eval.normalized_score

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            method_name="LLM-as-a-Judge",
            extraction=extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "judge_model": self.judge_model_name,
                "precision_score": judge_eval.precision_score,
                "hallucination_score": judge_eval.hallucination_score,
                "completeness_score": judge_eval.completeness_score,
                "justification": judge_eval.justification,
                "target_schema": schema.__name__,
            },
        )
