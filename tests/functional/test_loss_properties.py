# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.runtime.algebra import AlgebraContext
from clifra.functional.loss import (
    asymmetry_penalty,
    bivector_regularization,
    chamfer_distance,
    conservative_force_loss,
    geometric_mse,
    involution_consistency_loss,
    isometry_loss,
    physics_informed_loss,
    subspace_penalty,
)
from tests.helpers.hypothesis_cases import PROPERTY_SETTINGS, signature_strategy, tensor_with_shape
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = [pytest.mark.unit, pytest.mark.property]


@PROPERTY_SETTINGS
@given(data=st.data())
def test_coefficient_losses_match_direct_formulas(data):
    batch = data.draw(st.integers(1, 4))
    dim = data.draw(st.integers(1, 12))
    pred = data.draw(tensor_with_shape((batch, dim)))
    target = data.draw(tensor_with_shape((batch, dim)))
    metric = data.draw(tensor_with_shape((dim,)))
    mask = data.draw(st.lists(st.booleans(), min_size=dim, max_size=dim)).copy()
    mask = torch.tensor(mask, dtype=torch.bool)

    assert torch.allclose(geometric_mse(pred, target), (pred - target).square().mean())
    assert torch.allclose(subspace_penalty(pred, mask), pred[..., mask].square().sum(-1).mean())
    expected_isometry = (((pred.square() * metric).sum(-1) - (target.square() * metric).sum(-1)).square()).mean()
    assert torch.allclose(isometry_loss(pred, target, metric), expected_isometry)


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=4), data=st.data())
def test_algebraic_losses_match_independent_grade_operations(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(*signature)
    values = data.draw(tensor_with_shape((data.draw(st.integers(1, 3)), algebra.dim)))
    grade = data.draw(st.integers(0, algebra.n))
    projected = oracle.project(values, (grade,))

    assert torch.allclose(
        bivector_regularization(algebra, values, grade=grade), (values - projected).square().sum(-1).mean()
    )
    assert torch.allclose(
        involution_consistency_loss(values, oracle.grade_involution(values), algebra),
        torch.zeros((), dtype=torch.float64),
    )


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=5), data=st.data())
def test_bivector_regularization_uses_declared_compact_layout(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    grades = data.draw(
        st.sets(st.integers(0, algebra.n), min_size=1, max_size=algebra.n + 1).map(lambda value: tuple(sorted(value)))
    )
    layout = algebra.layout(grades)
    grade = data.draw(st.integers(0, algebra.n))
    values = data.draw(tensor_with_shape((data.draw(st.integers(1, 3)), layout.dim)))
    penalty = layout.grade_indices_tensor() != grade

    assert torch.allclose(
        bivector_regularization(algebra, values, grade=grade, layout=layout),
        values[..., penalty].square().sum(-1).mean(),
    )


def test_point_cloud_and_force_losses_match_analytic_gold_values():
    pred = torch.tensor([[[0.0], [2.0]]], dtype=torch.float64)
    target = torch.tensor([[[1.0]]], dtype=torch.float64)
    assert chamfer_distance(pred, target) == torch.tensor(2.0, dtype=torch.float64)
    assert chamfer_distance(pred + 7.0, target + 7.0) == torch.tensor(2.0, dtype=torch.float64)

    position = torch.tensor([[1.0, -2.0]], dtype=torch.float64, requires_grad=True)
    energy = position.square().sum(-1)
    assert conservative_force_loss(energy, -2.0 * position, position) == torch.tensor(0.0, dtype=torch.float64)


def test_physics_and_asymmetry_losses_have_declared_gradient_behavior():
    forecast = torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]], dtype=torch.float64)
    target = torch.zeros_like(forecast)
    expected = forecast.square().mean() + 0.25 * forecast.mean(1).square().mean()
    assert torch.allclose(physics_informed_loss(forecast, target, physics_weight=0.25), expected)

    forward = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64, requires_grad=True)
    reverse = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64, requires_grad=True)
    loss = asymmetry_penalty(forward, reverse, margin=0.0)
    loss.backward()
    assert forward.grad is None
    assert reverse.grad is not None
    assert torch.isfinite(reverse.grad).all()
