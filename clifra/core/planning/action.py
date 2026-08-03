# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Intent and layout plans for linear and versor-style actions."""

from __future__ import annotations

from dataclasses import dataclass, field

from clifra.core.foundation.basis import expand_output_grades
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.runtime.tensors import TensorContract, _check_contract_spec


@dataclass(frozen=True)
class LinearActionPlan:
    """Resolved contract for a vector-space action lifted to multivector grades."""

    input_layout: GradeLayout
    output_layout: GradeLayout
    input_contract: TensorContract = field(init=False, repr=False)
    output_contract: TensorContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        spec = self.input_layout.spec
        object.__setattr__(self, "input_contract", TensorContract.compact(spec, self.input_layout))
        output_contract = TensorContract.compact(self.output_layout.spec, self.output_layout)
        object.__setattr__(self, "output_contract", _check_contract_spec(spec, output_contract, "output_layout"))

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return the grades accepted by the action input layout."""
        return self.input_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return the grades emitted by the action output layout."""
        return self.output_layout.grades


@dataclass(frozen=True)
class VersorActionPlan:
    """Resolved contract for grade-1 or grade-2 versor actions."""

    grade: int
    input_layout: GradeLayout
    output_layout: GradeLayout
    parameter_layout: GradeLayout
    input_contract: TensorContract = field(init=False, repr=False)
    output_contract: TensorContract = field(init=False, repr=False)
    parameter_contract: TensorContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        spec = self.input_layout.spec
        object.__setattr__(self, "input_contract", TensorContract.compact(spec, self.input_layout))
        for role in ("output", "parameter"):
            layout = getattr(self, f"{role}_layout")
            contract = TensorContract.compact(layout.spec, layout)
            object.__setattr__(
                self,
                f"{role}_contract",
                _check_contract_spec(spec, contract, f"{role}_layout"),
            )

    @property
    def linear_action(self) -> LinearActionPlan:
        """Return the equivalent linear action over the same input/output layouts."""
        return LinearActionPlan(input_layout=self.input_layout, output_layout=self.output_layout)


@dataclass(frozen=True)
class PairedBivectorActionPlan:
    """Resolved contract for independent left/right bivector rotor actions."""

    input_layout: GradeLayout
    output_layout: GradeLayout
    parameter_layout: GradeLayout
    rotor_layout: GradeLayout
    middle_layout: GradeLayout
    input_contract: TensorContract = field(init=False, repr=False)
    output_contract: TensorContract = field(init=False, repr=False)
    parameter_contract: TensorContract = field(init=False, repr=False)
    rotor_contract: TensorContract = field(init=False, repr=False)
    middle_contract: TensorContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        spec = self.input_layout.spec
        object.__setattr__(self, "input_contract", TensorContract.compact(spec, self.input_layout))
        for role in ("output", "parameter", "rotor", "middle"):
            layout = getattr(self, f"{role}_layout")
            contract = TensorContract.compact(layout.spec, layout)
            object.__setattr__(
                self,
                f"{role}_contract",
                _check_contract_spec(spec, contract, f"{role}_layout"),
            )

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return the grades accepted before the paired bivector action."""
        return self.input_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return the grades retained after the paired bivector action."""
        return self.output_layout.grades


def build_linear_action_plan(
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
) -> LinearActionPlan:
    """Build a plan-only linear action contract."""
    spec = input_layout.spec
    input_contract = TensorContract.compact(spec, input_layout)
    output_layout = input_layout if output_layout is None else output_layout
    output_contract = TensorContract.compact(output_layout.spec, output_layout)
    _check_contract_spec(spec, output_contract, "output_layout")
    return LinearActionPlan(input_layout=input_contract.layout, output_layout=output_contract.layout)


def build_versor_action_plan(
    algebra,
    *,
    grade: int,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
    parameter_layout: GradeLayout | None = None,
) -> VersorActionPlan:
    """Build a plan-only versor action contract."""
    spec = AlgebraSpec.from_algebra(algebra)
    grade = int(grade)
    if grade not in {1, 2}:
        raise ValueError("planned versor actions currently support grade=1 and grade=2")
    output_layout = input_layout if output_layout is None else output_layout
    parameter_layout = algebra.layout((grade,)) if parameter_layout is None else parameter_layout
    input_contract = _check_contract_spec(
        spec, TensorContract.compact(input_layout.spec, input_layout), "input_layout"
    )
    output_contract = _check_contract_spec(
        spec, TensorContract.compact(output_layout.spec, output_layout), "output_layout"
    )
    parameter_contract = _check_contract_spec(
        spec, TensorContract.compact(parameter_layout.spec, parameter_layout), "parameter_layout"
    )
    if parameter_layout.grades != (grade,):
        raise ValueError(f"parameter_layout must contain grade {grade}, got {parameter_layout.grades}")
    return VersorActionPlan(
        grade=grade,
        input_layout=input_contract.layout,
        output_layout=output_contract.layout,
        parameter_layout=parameter_contract.layout,
    )


def build_paired_bivector_action_plan(
    algebra,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
    parameter_layout: GradeLayout | None = None,
) -> PairedBivectorActionPlan:
    """Build a plan for ``R_left x R_right_reverse`` with independent rotors.

    Unlike a true versor sandwich ``R x R~``, independent left/right rotors are
    not generally grade-preserving. The planner therefore expands the default
    output layout through both geometric products and lets callers explicitly
    project with ``output_layout`` when they want a narrower result.
    """
    spec = AlgebraSpec.from_algebra(algebra)
    parameter_layout = algebra.layout((2,)) if parameter_layout is None else parameter_layout
    input_contract = _check_contract_spec(
        spec, TensorContract.compact(input_layout.spec, input_layout), "input_layout"
    )
    parameter_contract = _check_contract_spec(
        spec, TensorContract.compact(parameter_layout.spec, parameter_layout), "parameter_layout"
    )
    input_layout = input_contract.layout
    parameter_layout = parameter_contract.layout
    if parameter_layout.grades != (2,):
        raise ValueError(f"parameter_layout must contain grade 2, got {parameter_layout.grades}")

    rotor_layout = spec.layout(range(0, spec.n + 1, 2))
    middle_grades = expand_output_grades(rotor_layout.grades, input_layout.grades, spec.n, op="gp")
    middle_layout = spec.layout(middle_grades)
    inferred_output = spec.layout(expand_output_grades(middle_layout.grades, rotor_layout.grades, spec.n, op="gp"))
    output_layout = inferred_output if output_layout is None else output_layout
    output_contract = _check_contract_spec(
        spec, TensorContract.compact(output_layout.spec, output_layout), "output_layout"
    )
    return PairedBivectorActionPlan(
        input_layout=input_layout,
        output_layout=output_contract.layout,
        parameter_layout=parameter_layout,
        rotor_layout=rotor_layout,
        middle_layout=middle_layout,
    )
