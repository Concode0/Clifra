# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Differentiable Clifford transformation fields with injectable objectives."""

from .criteria import TargetFieldCriterion
from .curriculum import ConstantCurriculum, CurriculumKnot, LossWeightSchedule, PhaseCurriculum
from .engine import OptimizationStepContext, TransformationFieldEngine, TransformationFitResult
from .field import CoordinateChart, InvertibleBivectorField
from .inputs import CoordinateFieldInput, CoordinateLike
from .logging import MetricLogger, MetricRecord
from .policies import BivectorNormPolicy, InvertiblePathConsistencyPolicy
from .sampling import (
    BroadcastGeneratorSampler,
    GeneratorFieldSample,
    GeneratorFieldSampler,
    RBFGeneratorSampler,
    RegularGridGeneratorSampler,
)
from .types import (
    CoordinateTransformationField,
    CriterionResult,
    GeometricPolicy,
    PolicyResult,
    TargetCriterion,
    TransformationEvaluation,
    TransformationState,
)

__all__ = [
    "BivectorNormPolicy",
    "BroadcastGeneratorSampler",
    "ConstantCurriculum",
    "CoordinateChart",
    "CoordinateFieldInput",
    "CoordinateLike",
    "CoordinateTransformationField",
    "CriterionResult",
    "CurriculumKnot",
    "GeometricPolicy",
    "GeneratorFieldSample",
    "GeneratorFieldSampler",
    "InvertibleBivectorField",
    "InvertiblePathConsistencyPolicy",
    "LossWeightSchedule",
    "MetricLogger",
    "MetricRecord",
    "OptimizationStepContext",
    "PhaseCurriculum",
    "PolicyResult",
    "RBFGeneratorSampler",
    "RegularGridGeneratorSampler",
    "TargetCriterion",
    "TargetFieldCriterion",
    "TransformationFieldEngine",
    "TransformationEvaluation",
    "TransformationFitResult",
    "TransformationState",
]
