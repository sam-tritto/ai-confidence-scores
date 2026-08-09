"""
Calibration Engine Module exporting BaseConfidenceEngine and all 9 specialized frameworks.
"""

from src.calibration.base import BaseConfidenceEngine
from src.calibration.native_logprob import NativeLogProbEngine
from src.calibration.logprob_delta import LogProbDeltaEngine
from src.calibration.temperature_scaling import TemperatureScalingEngine
from src.calibration.platt_scaling import PlattScalingEngine
from src.calibration.self_consistency import SelfConsistencyEngine
from src.calibration.verbalized_confidence import VerbalizedConfidenceEngine
from src.calibration.continuous_prompting import ContinuousPromptingEngine
from src.calibration.grounding_alignment import GroundingAlignmentEngine
from src.calibration.llm_as_a_judge import LLMAsAJudgeEngine

__all__ = [
    "BaseConfidenceEngine",
    "NativeLogProbEngine",
    "LogProbDeltaEngine",
    "TemperatureScalingEngine",
    "PlattScalingEngine",
    "SelfConsistencyEngine",
    "VerbalizedConfidenceEngine",
    "ContinuousPromptingEngine",
    "GroundingAlignmentEngine",
    "LLMAsAJudgeEngine",
]
