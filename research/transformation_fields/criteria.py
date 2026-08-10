# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Small generic target criteria for transformation fields.

Domain-specific criteria belong in the example or application that injects
them into :class:`TransformationFieldEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import CriterionResult, TransformationState


@dataclass(frozen=True)
class TargetFieldCriterion:
    """Fit transformed coordinates to a target coordinate tensor."""

    target_coordinates: torch.Tensor
    weight: float = 1.0
    name: str = "target_field"

    def __call__(self, engine, state: TransformationState) -> CriterionResult:
        target = self.target_coordinates.to(
            device=state.transformed_coordinates.device, dtype=state.transformed_coordinates.dtype
        )
        transformed, target = torch.broadcast_tensors(state.transformed_coordinates, target)
        residual = transformed - target
        mse = residual.square().mean()
        return CriterionResult(
            name=self.name,
            loss=mse * float(self.weight),
            metrics={
                "mse": mse,
                "rmse": mse.sqrt(),
                "max_abs": residual.abs().amax(),
                "weight": float(self.weight),
            },
        )
