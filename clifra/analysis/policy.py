# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static analysis budgets over compositions of core operation estimates.

Analysis is deliberately not a planning-policy family.  Optional reports
describe the core operations they compose and apply analysis-local budgets to
those estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from clifra.core.foundation.layout import GradeLayout
from clifra.core.planning.policy import PlanFacts, compose_plan_facts
from clifra.core.planning.product import estimate_product_executor_cost


@dataclass(frozen=True)
class AnalysisBudget:
    """Independent upper bounds for an optional analysis composition."""

    max_forward_work: float = float("inf")
    max_backward_work: float = float("inf")
    max_peak_bytes: int = (1 << 63) - 1
    max_compile_work: float = float("inf")


@dataclass(frozen=True)
class AnalysisComponent:
    """One core operation or analysis-local kernel in a report composition."""

    family: str
    route: str
    facts: PlanFacts


@dataclass(frozen=True)
class AnalysisComposition:
    """Static composable work description for one optional report."""

    components: tuple[AnalysisComponent, ...]

    @property
    def facts(self) -> PlanFacts:
        # Owners must pass an explicit peak when component lifetimes overlap.
        return compose_plan_facts(*(component.facts for component in self.components))


@dataclass(frozen=True)
class AnalysisFeasibility:
    """Static budget verdict for optional analysis materialization."""

    supported: bool
    reason: str
    details: Mapping[str, object]

    def __bool__(self) -> bool:
        return self.supported


@dataclass(frozen=True)
class MatrixAnalysisCost:
    """Analysis-local explicit matrix component."""

    role: str
    matrix_kind: str
    matrix_dim: int
    max_entries: int
    dtype: torch.dtype

    @property
    def matrix_entries(self) -> int:
        return self.matrix_dim * self.matrix_dim

    @property
    def estimated_bytes(self) -> int:
        return self.matrix_entries * _dtype_bytes(self.dtype)

    @property
    def composition(self) -> AnalysisComposition:
        work = float(self.matrix_entries)
        return AnalysisComposition(
            (
                AnalysisComponent(
                    "analysis_local",
                    f"{self.matrix_kind}_matrix",
                    PlanFacts(work, 0.0, self.estimated_bytes, work),
                ),
            )
        )

    def details(self) -> dict[str, object]:
        facts = self.composition.facts
        return {
            "role": self.role,
            "matrix_kind": self.matrix_kind,
            "matrix_dim": self.matrix_dim,
            "matrix_entries": self.matrix_entries,
            "max_entries": self.max_entries,
            "estimated_bytes": self.estimated_bytes,
            "forward_work": facts.forward_work,
            "peak_bytes": facts.peak_bytes,
            "components": (f"analysis_local:{self.matrix_kind}_matrix",),
            "dtype": str(self.dtype).removeprefix("torch."),
        }


@dataclass(frozen=True)
class ProductAnalysisCost:
    """Core product component used by an optional analysis report."""

    role: str
    op: str
    left_layout: GradeLayout
    right_layout: GradeLayout
    output_layout: GradeLayout
    max_pairs: int
    executor_family: str
    pair_count: int
    estimated_pairs: int
    estimated_bytes: int
    path_count: int
    backend: str
    dtype: torch.dtype
    composition: AnalysisComposition

    def details(self) -> dict[str, object]:
        facts = self.composition.facts
        return {
            "role": self.role,
            "op": self.op,
            "n": self.left_layout.spec.n,
            "left_grades": self.left_layout.grades,
            "right_grades": self.right_layout.grades,
            "output_grades": self.output_layout.grades,
            "left_lanes": self.left_layout.dim,
            "right_lanes": self.right_layout.dim,
            "output_lanes": self.output_layout.dim,
            "estimated_pairs": self.estimated_pairs,
            "pair_count": self.pair_count,
            "max_pairs": self.max_pairs,
            "path_count": self.path_count,
            "executor_family": self.executor_family,
            "backend": self.backend,
            "estimated_bytes": self.estimated_bytes,
            "forward_work": facts.forward_work,
            "backward_work": facts.backward_work,
            "peak_bytes": facts.peak_bytes,
            "compile_work": facts.compile_work,
            "components": (f"product:{self.executor_family}",),
            "dtype": str(self.dtype).removeprefix("torch."),
        }


def feasibility_record(feasibility: AnalysisFeasibility) -> dict[str, object]:
    return {"reason": feasibility.reason, "details": dict(feasibility.details)}


def evaluate_composition(
    composition: AnalysisComposition,
    budget: AnalysisBudget,
    *,
    reason: str = "analysis_budget",
    details: Optional[Mapping[str, object]] = None,
) -> AnalysisFeasibility:
    facts = composition.facts
    violations = {}
    if facts.forward_work > budget.max_forward_work:
        violations["forward_work"] = (facts.forward_work, budget.max_forward_work)
    if facts.backward_work > budget.max_backward_work:
        violations["backward_work"] = (facts.backward_work, budget.max_backward_work)
    if facts.peak_bytes > budget.max_peak_bytes:
        violations["peak_bytes"] = (facts.peak_bytes, budget.max_peak_bytes)
    if facts.compile_work > budget.max_compile_work:
        violations["compile_work"] = (facts.compile_work, budget.max_compile_work)
    payload = dict(details or {})
    payload["violations"] = violations
    return AnalysisFeasibility(not violations, "ok" if not violations else reason, payload)


def evaluate_matrix_cost(cost: MatrixAnalysisCost) -> AnalysisFeasibility:
    details = cost.details()
    if cost.matrix_entries > cost.max_entries:
        return AnalysisFeasibility(False, f"{cost.matrix_kind}_matrix_cap", details)
    return AnalysisFeasibility(True, "ok", details)


def build_product_analysis_cost(
    algebra,
    *,
    role: str,
    op: str,
    left_layout: GradeLayout,
    right_layout: GradeLayout,
    output_layout: GradeLayout,
    max_pairs: int,
    dtype: Optional[torch.dtype] = None,
    device=None,
) -> ProductAnalysisCost:
    """Map an analysis product to the selected core product route estimate."""
    resolved_dtype = getattr(algebra, "dtype", torch.float32) if dtype is None else dtype
    resolved_device = getattr(algebra, "device", "cpu") if device is None else device
    executor_cost = estimate_product_executor_cost(
        algebra,
        op=op,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        dtype=resolved_dtype,
        device=resolved_device,
    )
    composition = AnalysisComposition(
        (AnalysisComponent("product", executor_cost.decision.route, executor_cost.decision.facts),)
    )
    return ProductAnalysisCost(
        role=str(role),
        op=str(op),
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        max_pairs=int(max_pairs),
        executor_family=executor_cost.decision.route,
        pair_count=int(executor_cost.pair_count),
        estimated_pairs=int(left_layout.dim) * int(right_layout.dim),
        estimated_bytes=int(executor_cost.decision.facts.peak_bytes),
        path_count=int(executor_cost.path_count),
        backend=executor_cost.backend,
        dtype=resolved_dtype,
        composition=composition,
    )


def evaluate_product_cost(cost: ProductAnalysisCost) -> AnalysisFeasibility:
    details = cost.details()
    if cost.pair_count > cost.max_pairs:
        return AnalysisFeasibility(False, "product_pair_cap", details)
    return AnalysisFeasibility(True, "ok", details)


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()
