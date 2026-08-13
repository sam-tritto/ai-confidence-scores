"""
Calibration Method Module exporting BaseConfidenceMethod and all 10 specialized frameworks.
"""

from src.calibration.base import BaseConfidenceMethod
from src.calibration.native_logprob import NativeLogProbMethod
from src.calibration.logprob_delta import LogProbDeltaMethod
from src.calibration.temperature_scaling import TemperatureScalingMethod
from src.calibration.platt_scaling import PlattScalingMethod
from src.calibration.self_consistency import SelfConsistencyMethod
from src.calibration.verbalized_confidence import VerbalizedConfidenceMethod
from src.calibration.continuous_prompting import ContinuousPromptingMethod
from src.calibration.grounding_alignment import GroundingAlignmentMethod
from src.calibration.llm_as_a_judge import LLMAsAJudgeMethod
from src.calibration.structured_self_assessment_platt import StructuredSelfAssessmentPlattMethod

# Aliases for convenience / backward compatibility
StructuredSelfAssessmentMethod = VerbalizedConfidenceMethod
StructuredPlattMethod = StructuredSelfAssessmentPlattMethod

__all__ = [
    "BaseConfidenceMethod",
    "NativeLogProbMethod",
    "LogProbDeltaMethod",
    "TemperatureScalingMethod",
    "PlattScalingMethod",
    "SelfConsistencyMethod",
    "VerbalizedConfidenceMethod",
    "ContinuousPromptingMethod",
    "GroundingAlignmentMethod",
    "LLMAsAJudgeMethod",
    "StructuredSelfAssessmentPlattMethod",
    "StructuredSelfAssessmentMethod",
    "StructuredPlattMethod",
]
