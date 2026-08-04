# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.runtime.algebra import AlgebraContext
from clifra.functional.orthogonality import (
    cross_grade_coupling,
    diagnostics,
    grade_energies,
    grade_masks,
    parasitic_energy,
    parasitic_ratio,
    project_to_target_grades,
    target_mask_from_grades,
)
from tests.helpers.hypothesis_cases import PROPERTY_SETTINGS, grade_set_strategy, signature_strategy, tensor_with_shape

pytestmark = [pytest.mark.unit, pytest.mark.property]


@PROPERTY_SETTINGS
@given(signature=signature_strategy(max_n=5), data=st.data())
def test_compact_grade_masks_follow_basis_indices_and_partition_values(signature, data):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float64)
    grades = data.draw(grade_set_strategy(algebra.n))
    layout = algebra.layout(grades)
    target_grades = data.draw(st.sets(st.sampled_from(grades), min_size=1)).copy()
    values = data.draw(tensor_with_shape((data.draw(st.integers(1, 4)), layout.dim)))
    masks = grade_masks(algebra.n + 1, layout.dim, basis_indices=layout.basis_indices)
    target_mask = target_mask_from_grades(masks, sorted(target_grades))
    projected = project_to_target_grades(values, target_mask)

    assert torch.equal(masks.to(torch.long).argmax(0), layout.grade_indices_tensor())
    assert torch.equal(project_to_target_grades(projected, target_mask), projected)
    assert torch.equal(projected + project_to_target_grades(values, ~target_mask), values)
    parasitic = values[..., ~target_mask]
    expected_parasitic = values.new_zeros(()) if parasitic.numel() == 0 else parasitic.square().mean()
    assert torch.allclose(parasitic_energy(values, ~target_mask), expected_parasitic)
    assert set(grade_energies(values, masks)) == set(range(algebra.n + 1))
    assert 0.0 <= parasitic_ratio(values, masks, sorted(target_grades)) <= 1.0
    assert torch.isfinite(cross_grade_coupling(values, masks)).all()


def test_orthogonality_diagnostics_match_precalculated_grade_energy_gold():
    values = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=torch.float64)
    masks = grade_masks(2, 2)
    report = diagnostics(values, masks, target_grades=[0], tolerance=0.1)

    assert report["grade_energies"] == {0: 14.0 / 3.0, 1: 1.0 / 3.0}
    assert report["parasitic_ratio"] == pytest.approx(1.0 / 15.0)
    assert report["coupling_matrix"].shape == (2, 2)
    assert torch.allclose(report["coupling_matrix"], report["coupling_matrix"].T)


def test_grade_masks_rejects_mismatched_compact_basis_count():
    with pytest.raises(ValueError, match="basis_indices must contain"):
        grade_masks(3, 2, basis_indices=(1,))
