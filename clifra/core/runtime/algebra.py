# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Differentiable Clifford Algebra core.

Implements the geometric product, grade projections, and rotor operations
for arbitrary signatures Cl(p, q, r).
"""

from typing import Iterable, Optional

import torch

from clifra.core.foundation.basis import normalize_grades
from clifra.core.foundation.device import resolve_device, resolve_dtype
from clifra.core.foundation.host import AlgebraHostMixin
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.planning.exp import DEFAULT_BIVECTOR_EXP_OPTIONS, BivectorExpOptions
from clifra.core.planning.planner import GradePlanner
from clifra.core.planning.policy import DEFAULT_PLANNING_POLICY, PlanningPolicy
from clifra.core.planning.resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits


class AlgebraContext(AlgebraHostMixin):
    """Signature and planning host for layout-first Clifford Algebra."""

    def __init__(
        self,
        p: int,
        q: int = 0,
        r: int = 0,
        *,
        device="cpu",
        dtype: torch.dtype = torch.float32,
        default_grades: Optional[Iterable[int]] = None,
        planning_policy: Optional[PlanningPolicy] = None,
        resource_limits: Optional[ResourceLimits] = None,
        bivector_exp_options: Optional[BivectorExpOptions] = None,
    ):
        if p < 0 or q < 0 or r < 0:
            raise ValueError(f"signature counts must be non-negative, got Cl({p},{q},{r})")

        self.p = int(p)
        self.q = int(q)
        self.r = int(r)
        self.n = self.p + self.q + self.r
        self.dim = 1 << self.n
        self.num_grades = self.n + 1
        self.spec = AlgebraSpec(self.p, self.q, self.r)
        self._device = torch.device(resolve_device(device) if str(device) == "auto" else device)
        self._dtype = resolve_dtype(dtype)
        self._planning_policy = DEFAULT_PLANNING_POLICY if planning_policy is None else planning_policy
        self.resource_limits = DEFAULT_RESOURCE_LIMITS if resource_limits is None else resource_limits
        self.bivector_exp_options = (
            DEFAULT_BIVECTOR_EXP_OPTIONS if bivector_exp_options is None else bivector_exp_options
        )
        self._default_grades = None if default_grades is None else normalize_grades(default_grades, self.n)
        self._default_layout: Optional[GradeLayout] = None
        self._g1_indices_cache: dict[str, torch.Tensor] = {}
        self.planner = GradePlanner(self)
        self._sync_eps()

    @property
    def planning_policy(self) -> PlanningPolicy:
        """Return the immutable route policy injected at construction."""
        return self._planning_policy

    @property
    def device(self):
        """Return the context device used for planned executor buffers."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the context floating-point dtype."""
        return self._dtype

    def bivector_squared_signs(self, *, device=None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Return ``(e_ab)^2`` signs in canonical grade-2 layout order."""
        return self.planner.bivector_squared_signs(
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )

    def _apply(self, fn):
        """Apply a PyTorch module-style device/dtype transform to cached executors."""
        probe = fn(torch.empty((), device=self.device, dtype=self.dtype))
        self._device = probe.device
        if probe.dtype.is_floating_point:
            self._dtype = probe.dtype
        self._sync_eps()
        self._g1_indices_cache.clear()
        self.planner.clear_cache()
        return self

    def to(self, device=None, dtype=None):
        """Move the context and cached executors."""
        if device is not None:
            self._device = torch.device(resolve_device(device) if str(device) == "auto" else device)
        if dtype is not None:
            self._dtype = resolve_dtype(dtype)
        self._sync_eps()
        self._g1_indices_cache.clear()
        self.planner.clear_cache()
        return self

    def _basis_vector_indices(self, device) -> torch.Tensor:
        resolved = torch.device(device)
        key = str(resolved)
        cached = self._g1_indices_cache.get(key)
        if cached is None:
            cached = torch.tensor([1 << bit for bit in range(self.n)], dtype=torch.long, device=resolved)
            self._g1_indices_cache[key] = cached
        return cached

    def _sync_eps(self) -> None:
        finfo = torch.finfo(self.dtype)
        self.eps = float(finfo.eps)
        self.eps_sq = float(finfo.eps**2)
