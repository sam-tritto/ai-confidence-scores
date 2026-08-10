"""
Unit tests for all 9 Confidence Calibration Engines with configurable schemas and exception handling.
"""

import numpy as np
import pytest

from src.calibration import (
    ContinuousPromptingEngine,
    GroundingAlignmentEngine,
    LLMAsAJudgeEngine,
    LogProbDeltaEngine,
    NativeLogProbEngine,
    PlattScalingEngine,
    SelfConsistencyEngine,
    TemperatureScalingEngine,
    VerbalizedConfidenceEngine,
)
from src.exceptions import ClientNotConfiguredError, LogProbsUnavailableError
from src.schema import CalibrationResult, CustomerSupportTicket, GenericExtraction, ResumeExtraction


def test_client_not_configured_raises_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("USE_VERTEX_AI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    engine = NativeLogProbEngine(client=None)
    engine.client = None
    with pytest.raises(ClientNotConfiguredError):
        engine.evaluate("Sample text input")


def test_native_logprob_engine_with_custom_schema(sample_resume_text, mock_gemini_client):
    engine = NativeLogProbEngine(client=mock_gemini_client)
    res = engine.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert isinstance(res, CalibrationResult)
    assert res.engine_name == "Native Token LogProb"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "mean_logprob" in res.metadata


def test_logprob_delta_engine(sample_resume_text, mock_gemini_client):
    engine = LogProbDeltaEngine(client=mock_gemini_client)
    res = engine.evaluate(sample_resume_text, schema=GenericExtraction)
    assert res.engine_name == "LogProb Delta"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "mean_delta" in res.metadata


def test_temperature_scaling_engine(sample_resume_text, mock_gemini_client):
    engine = TemperatureScalingEngine(client=mock_gemini_client, temperature=1.2)
    logits = np.array([2.5, 1.2, -0.5, 3.1, 0.8])
    labels = np.array([1, 1, 0, 1, 0])
    fitted_t = engine.fit(logits, labels)
    assert fitted_t > 0.0

    res = engine.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.engine_name == "Temperature Scaling"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_platt_scaling_engine(sample_resume_text, mock_gemini_client):
    engine = PlattScalingEngine(client=mock_gemini_client)
    deltas = np.array([2.4, 0.5, 1.8, 3.1, 0.2])
    y_true = np.array([1, 0, 1, 1, 0])
    engine.fit(deltas, y_true)
    assert engine.is_fitted is True

    res = engine.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.engine_name == "Platt Scaling Logistic"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_self_consistency_engine(sample_resume_text, mock_gemini_client):
    engine = SelfConsistencyEngine(client=mock_gemini_client, num_samples=3)
    res = engine.evaluate(sample_resume_text, schema=CustomerSupportTicket)
    assert res.engine_name == "Self-Consistency & Agreement"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_verbalized_confidence_engine(sample_resume_text, mock_gemini_client):
    engine = VerbalizedConfidenceEngine(client=mock_gemini_client)
    res = engine.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.engine_name == "Structured Verbalized Confidence"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_continuous_prompting_engine(sample_resume_text, mock_gemini_client):
    engine = ContinuousPromptingEngine(client=mock_gemini_client)
    res = engine.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.engine_name == "Continuous Numerical Prompting"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_grounding_alignment_engine(sample_resume_text, mock_gemini_client):
    engine = GroundingAlignmentEngine(client=mock_gemini_client)
    res = engine.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.engine_name == "Grounding & Alignment"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_llm_as_a_judge_engine(sample_resume_text, mock_gemini_client):
    engine = LLMAsAJudgeEngine(client=mock_gemini_client)
    res = engine.evaluate(sample_resume_text, schema=ResumeExtraction)
    assert res.engine_name == "LLM-as-a-Judge"
    assert 0.0 <= res.calibrated_confidence <= 1.0
