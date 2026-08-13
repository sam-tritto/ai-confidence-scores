"""
5. Self-Consistency & Sampling Agreement Method.
Generates N=5 parallel completions at T=0.7, clusters outputs, and computes agreement ratio.
Domain-agnostic with configurable Pydantic response schemas.
"""

from collections import Counter
import time
from typing import Any, List, Optional, Type
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.calibration.base import BaseConfidenceMethod
from src.exceptions import ExtractionValidationError
from src.schema import CalibrationResult, GenericExtraction


class SelfConsistencyMethod(BaseConfidenceMethod):
    """Method 5: Self-Consistency & Sampling Agreement Method."""

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: Optional[str] = None,
        audit_threshold: Optional[float] = None,
        num_samples: int = 5,
        sample_temperature: float = 0.7,
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
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature

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
            temperature=self.sample_temperature,
        )
        contents = [eval_prompt]
        if pdf_bytes:
            contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        self.logger.info("Executing %s with %d samples on schema %s", self.__class__.__name__, self.num_samples, schema.__name__)
        sample_extractions: List[BaseModel] = []
        raw_keys: List[str] = []

        for _ in range(self.num_samples):
            try:
                response = self._generate_content_with_retry(
                    contents=contents,
                    config=config,
                )
                if response.text:
                    ext = schema.model_validate_json(response.text)
                    sample_extractions.append(ext)
                    # Key representation for clustering
                    key = getattr(ext, "domain_role", None) or getattr(ext, "primary_category", None) or ext.model_dump_json()
                    raw_keys.append(str(key))
            except Exception as e:
                self.logger.warning("Sampling completion failed: %s", str(e))

        if not sample_extractions:
            raise ExtractionValidationError(self.__class__.__name__, "", f"All {self.num_samples} sampling completions failed.")

        # Cluster by key representation
        clusters: Counter[str] = Counter(raw_keys)
        most_common_key, majority_count = clusters.most_common(1)[0]

        # Find consensus extraction matching majority cluster
        consensus_index = raw_keys.index(most_common_key)
        consensus_extraction = sample_extractions[consensus_index]

        agreement_ratio = majority_count / float(len(sample_extractions))
        raw_confidence = agreement_ratio
        calibrated_confidence = agreement_ratio

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            method_name="Self-Consistency & Agreement",
            extraction=consensus_extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "num_samples_requested": self.num_samples,
                "num_samples_successful": len(sample_extractions),
                "majority_count": majority_count,
                "agreement_ratio": agreement_ratio,
                "target_schema": schema.__name__,
            },
        )
