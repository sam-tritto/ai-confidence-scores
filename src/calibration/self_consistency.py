"""
5. Self-Consistency & Sampling Agreement Engine.
Generates N=5 parallel completions at T=0.7, clusters outputs, and computes agreement ratio.
Confidence = Majority Cluster Size / N
"""

from collections import Counter
import time
from typing import List, Optional, Tuple
from google.genai import types

from src.calibration.base import BaseConfidenceEngine
from src.schema import CalibrationResult, DomainRole, ResumeExtraction, SeniorityLevel


class SelfConsistencyEngine(BaseConfidenceEngine):
    """Engine 5: Self-Consistency & Sampling Agreement Engine.
    
    Generates N=5 sampled completions at temperature T=0.7, clusters extracted outputs,
    and returns majority cluster agreement ratio as the confidence score.
    """

    def __init__(
        self,
        client=None,
        model_name: str = "gemini-2.5-flash",
        audit_threshold: float = 0.75,
        num_samples: int = 5,
        sample_temperature: float = 0.7,
    ):
        super().__init__(client=client, model_name=model_name, audit_threshold=audit_threshold)
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature

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

        sample_extractions: List[ResumeExtraction] = []

        if self.client is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeExtraction,
                temperature=self.sample_temperature,
            )
            contents = [prompt]
            if pdf_bytes:
                contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

            for _ in range(self.num_samples):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config,
                    )
                    if response.text:
                        ext = ResumeExtraction.model_validate_json(response.text)
                        sample_extractions.append(ext)
                except Exception:
                    pass

        # Fallback if less than N samples returned
        fallback = self._create_fallback_extraction(resume_text)
        while len(sample_extractions) < self.num_samples:
            sample_extractions.append(fallback)

        # Cluster by (domain_role, seniority_level)
        clusters: Counter[Tuple[DomainRole, SeniorityLevel]] = Counter()
        for ext in sample_extractions:
            clusters[(ext.domain_role, ext.seniority_level)] += 1

        most_common_cluster, majority_count = clusters.most_common(1)[0]
        majority_role, majority_seniority = most_common_cluster

        # Find representative extraction matching majority
        consensus_extraction = next(
            (e for e in sample_extractions if e.domain_role == majority_role and e.seniority_level == majority_seniority),
            fallback,
        )

        agreement_ratio = majority_count / float(self.num_samples)
        raw_confidence = agreement_ratio
        calibrated_confidence = agreement_ratio

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        decision = self.determine_audit_decision(calibrated_confidence)

        return CalibrationResult(
            engine_name="Self-Consistency & Agreement",
            extraction=consensus_extraction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            audit_decision=decision,
            latency_ms=latency_ms,
            metadata={
                "num_samples": self.num_samples,
                "majority_count": majority_count,
                "agreement_ratio": agreement_ratio,
                "cluster_breakdown": {f"{r.value} | {s.value}": count for (r, s), count in clusters.items()},
            },
        )
