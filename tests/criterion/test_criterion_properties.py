# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext
from clifra.criterion import (
    BivectorRegularization,
    GeometricMSELoss,
    IsometryLoss,
    OrthogonalitySettings,
    StrictOrthogonality,
    SubspaceLoss,
)
from clifra.functional.loss import bivector_regularization, geometric_mse, isometry_loss, subspace_penalty

pytestmark = pytest.mark.unit


def test_loss_modules_match_functional_formulas_on_precalculated_values():
    algebra = AlgebraContext(2, 0, 0, device="cpu", dtype=torch.float64)
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float64)

    mse = GeometricMSELoss(algebra)
    subspace = SubspaceLoss(algebra, target_indices=[0, 1])
    isometry = IsometryLoss(algebra)
    regularization = BivectorRegularization(algebra, grade=1)

    assert mse(pred, target) == geometric_mse(pred, target)
    assert subspace(pred) == subspace_penalty(pred, subspace.penalty_mask)
    assert isometry(pred, target) == isometry_loss(pred, target, isometry.metric_diag)
    assert regularization(pred) == bivector_regularization(algebra, pred, grade=1)


def test_bivector_regularization_module_accepts_compact_layout():
    algebra = AlgebraContext(4, 0, 0, device="cpu", dtype=torch.float64)
    layout = algebra.layout((1, 2))
    values = torch.arange(layout.dim, dtype=torch.float64).unsqueeze(0)
    regularization = BivectorRegularization(algebra, grade=2, layout=layout)

    assert regularization(values) == values[..., layout.positions_for_grades((1,))].square().sum()


def test_strict_orthogonality_supports_compact_layout_and_buffer_conversion():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    layout = algebra.layout((0, 2))
    settings = OrthogonalitySettings(mode="project", target_grades=[2])
    criterion = StrictOrthogonality(algebra, settings, layout=layout).to(dtype=torch.float64)
    values = torch.arange(1, layout.dim + 1, dtype=torch.float64).unsqueeze(0)
    expected = values.clone()
    expected[..., layout.positions_for_grades((0,))] = 0.0

    assert torch.equal(criterion(values), expected)
    assert criterion.target_mask.dtype == torch.bool
    assert torch.equal(criterion.state_dict()["target_mask"], criterion.target_mask)


def test_strict_orthogonality_rejects_foreign_layout():
    algebra = AlgebraContext(2, 0, 0)
    with pytest.raises(ValueError, match="layout signature"):
        StrictOrthogonality(algebra, layout=AlgebraContext(1, 1, 0).layout((1,)))
