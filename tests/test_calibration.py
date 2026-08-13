"""
Unit tests for all 9 Confidence Calibration Methods with configurable schemas and exception handling.
"""

import numpy as np
import pytest

from src.calibration import (
    ContinuousPromptingMethod,
    GroundingAlignmentMethod,
    LLMAsAJudgeMethod,
    LogProbDeltaMethod,
    NativeLogProbMethod,
    PlattScalingMethod,
    SelfConsistencyMethod,
    StructuredSelfAssessmentPlattMethod,
    TemperatureScalingMethod,
    VerbalizedConfidenceMethod,
)
from src.exceptions import ClientNotConfiguredError, LogProbsUnavailableError
from src.schema import CalibrationResult, CustomerSupportTicket, GenericExtraction, ResumeExtraction


def test_client_not_configured_raises_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("USE_VERTEX_AI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    method = NativeLogProbMethod(client=None)
    method.client = None
    with pytest.raises(ClientNotConfiguredError):
        method.evaluate("Sample text input")


def test_native_logprob_method_with_custom_schema(sample_resume_text, mock_gemini_client):
    method = NativeLogProbMethod(client=mock_gemini_client)
    res = method.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert isinstance(res, CalibrationResult)
    assert res.method_name == "Native Token LogProb"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "mean_logprob" in res.metadata


def test_logprob_delta_method(sample_resume_text, mock_gemini_client):
    method = LogProbDeltaMethod(client=mock_gemini_client)
    res = method.evaluate(sample_resume_text, schema=GenericExtraction)
    assert res.method_name == "LogProb Delta"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "mean_delta" in res.metadata


def test_temperature_scaling_method(sample_resume_text, mock_gemini_client):
    method = TemperatureScalingMethod(client=mock_gemini_client, temperature=1.2)
    logits = np.array([2.5, 1.2, -0.5, 3.1, 0.8])
    labels = np.array([1, 1, 0, 1, 0])
    fitted_t = method.fit(logits, labels)
    assert fitted_t > 0.0

    res = method.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.method_name == "Temperature Scaling"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_platt_scaling_method(sample_resume_text, mock_gemini_client):
    method = PlattScalingMethod(client=mock_gemini_client)
    deltas = np.array([2.4, 0.5, 1.8, 3.1, 0.2])
    y_true = np.array([1, 0, 1, 1, 0])
    method.fit(deltas, y_true)
    assert method.is_fitted is True

    res = method.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.method_name == "Platt Scaling Logistic"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_self_consistency_method(sample_resume_text, mock_gemini_client):
    method = SelfConsistencyMethod(client=mock_gemini_client, num_samples=3)
    res = method.evaluate(sample_resume_text, schema=CustomerSupportTicket)
    assert res.method_name == "Self-Consistency & Agreement"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_verbalized_confidence_method(sample_resume_text, mock_gemini_client):
    method = VerbalizedConfidenceMethod(client=mock_gemini_client)
    res = method.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.method_name == "Structured Verbalized Confidence"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_continuous_prompting_method(sample_resume_text, mock_gemini_client):
    method = ContinuousPromptingMethod(client=mock_gemini_client)
    res = method.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.method_name == "Continuous Numerical Prompting"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_grounding_alignment_method(sample_resume_text, mock_gemini_client):
    method = GroundingAlignmentMethod(client=mock_gemini_client)
    res = method.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.method_name == "Grounding & Alignment"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_llm_as_a_judge_method(sample_resume_text, mock_gemini_client):
    method = LLMAsAJudgeMethod(client=mock_gemini_client)
    res = method.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.method_name == "LLM-as-a-Judge"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_structured_self_assessment_platt_method(sample_resume_text, mock_gemini_client):
    method = StructuredSelfAssessmentPlattMethod(client=mock_gemini_client, a=1.2, b=-0.1)
    res = method.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.method_name == "Structured Self-Assessment + Platt Scaling"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert 0.0 <= res.raw_confidence <= 1.0
    assert "platt_a" in res.metadata


def test_structured_self_assessment_platt_fit():
    method = StructuredSelfAssessmentPlattMethod(client=None)
    val_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    val_labels = np.array([1, 1, 1, 0, 0, 0])
    a, b = method.fit(val_scores, val_labels)
    assert method.is_fitted
    assert isinstance(a, float)
    assert isinstance(b, float)
