"""
Unit tests for all 9 Confidence Calibration Engines.
"""

import numpy as np
import pytest

from src.calibration.native_logprob import NativeLogProbEngine
from src.calibration.logprob_delta import LogProbDeltaEngine
from src.calibration.temperature_scaling import TemperatureScalingEngine
from src.calibration.platt_scaling import PlattScalingEngine
from src.calibration.self_consistency import SelfConsistencyEngine
from src.calibration.verbalized_confidence import VerbalizedConfidenceEngine
from src.calibration.continuous_prompting import ContinuousPromptingEngine
from src.calibration.grounding_alignment import GroundingAlignmentEngine
from src.calibration.llm_as_a_judge import LLMAsAJudgeEngine
from src.schema import AuditDecision, CalibrationResult


def test_native_logprob_engine(sample_resume_text, mock_gemini_client):
    engine = NativeLogProbEngine(client=mock_gemini_client)
    res = engine.evaluate(sample_resume_text)
    assert isinstance(res, CalibrationResult)
    assert res.engine_name == "Native Token LogProb"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "mean_logprob" in res.metadata


def test_logprob_delta_engine(sample_resume_text, mock_gemini_client):
    engine = LogProbDeltaEngine(client=mock_gemini_client)
    res = engine.evaluate(sample_resume_text)
    assert res.engine_name == "LogProb Delta"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "mean_delta" in res.metadata


def test_temperature_scaling_engine(sample_resume_text, mock_gemini_client):
    engine = TemperatureScalingEngine(client=mock_gemini_client, temperature=1.2)
    # Test fitting
    logits = np.array([2.5, 1.2, -0.5, 3.1, 0.8])
    labels = np.array([1, 1, 0, 1, 0])
    fitted_t = engine.fit(logits, labels)
    assert fitted_t > 0.0

    res = engine.evaluate(sample_resume_text)
    assert res.engine_name == "Temperature Scaling"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert res.metadata["is_fitted"] is True


def test_platt_scaling_engine(sample_resume_text, mock_gemini_client):
    engine = PlattScalingEngine(client=mock_gemini_client)
    deltas = np.array([2.4, 0.5, 1.8, 3.1, 0.2])
    y_true = np.array([1, 0, 1, 1, 0])
    engine.fit(deltas, y_true)
    assert engine.is_fitted is True

    res = engine.evaluate(sample_resume_text)
    assert res.engine_name == "Platt Scaling Logistic"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_self_consistency_engine(sample_resume_text, mock_gemini_client):
    engine = SelfConsistencyEngine(client=mock_gemini_client, num_samples=3)
    res = engine.evaluate(sample_resume_text)
    assert res.engine_name == "Self-Consistency & Agreement"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert res.metadata["num_samples"] == 3


def test_verbalized_confidence_engine(sample_resume_text):
    engine = VerbalizedConfidenceEngine(client=None)  # Test fallback mode
    res = engine.evaluate(sample_resume_text)
    assert res.engine_name == "Structured Verbalized Confidence"
    assert 0.0 <= res.calibrated_confidence <= 1.0


def test_continuous_prompting_engine(sample_resume_text):
    engine = ContinuousPromptingEngine(client=None)
    res = engine.evaluate(sample_resume_text)
    assert res.engine_name == "Continuous Numerical Prompting"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "rating_0_to_100" in res.metadata


def test_grounding_alignment_engine(sample_resume_text):
    engine = GroundingAlignmentEngine(client=None)
    res = engine.evaluate(sample_resume_text)
    assert res.engine_name == "Grounding & Alignment"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "grounding_score" in res.metadata


def test_llm_as_a_judge_engine(sample_resume_text):
    engine = LLMAsAJudgeEngine(client=None)
    res = engine.evaluate(sample_resume_text)
    assert res.engine_name == "LLM-as-a-Judge"
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert "precision_score" in res.metadata
