# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Input contracts for geometric transformation fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias, Union

import torch


@dataclass(frozen=True)
class CoordinateFieldInput:
    """Coordinates together with the labels used to sample a generator field.

    ``coordinates`` are the values transformed by the Clifford action.
    ``sample_coordinates`` are optional persistent domain or parameter-space
    labels. They need only share the same prefix shape as ``coordinates`` and
    may have a different final dimension. Coordinate-driven samplers use these
    labels when present and otherwise sample at ``coordinates``. Sampling and
    interpolation affect generator weights, never the coordinate values.

    ``domain_shape`` explicitly identifies the suffix of the coordinate prefix
    that represents one structured sample domain. It is metadata only: it does
    not constrain the numerical coordinate values.
    """

    coordinates: torch.Tensor
    sample_coordinates: torch.Tensor | None = None
    domain_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, torch.Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if self.coordinates.ndim < 1:
            raise ValueError("coordinates must have at least one dimension")
        if self.sample_coordinates is not None:
            if not isinstance(self.sample_coordinates, torch.Tensor):
                raise TypeError("sample_coordinates must be a torch.Tensor")
            if self.sample_coordinates.ndim < 1:
                raise ValueError("sample_coordinates must have at least one dimension")
            if self.sample_coordinates.shape[:-1] != self.coordinates.shape[:-1]:
                raise ValueError(
                    "sample_coordinates must share the coordinate prefix shape, got "
                    f"{tuple(self.sample_coordinates.shape[:-1])} and {tuple(self.coordinates.shape[:-1])}"
                )
        if self.domain_shape is not None:
            domain_shape = tuple(_positive_int(value, "domain_shape") for value in self.domain_shape)
            prefix_shape = self.prefix_shape
            suffix_mismatch = bool(domain_shape) and prefix_shape[-len(domain_shape) :] != domain_shape
            if len(domain_shape) > len(prefix_shape) or suffix_mismatch:
                raise ValueError(
                    f"domain_shape={domain_shape} must be a suffix of coordinate prefix shape {prefix_shape}"
                )
            object.__setattr__(self, "domain_shape", domain_shape)

    @property
    def prefix_shape(self) -> tuple[int, ...]:
        """Return all coordinate axes except the final coordinate lane."""
        return tuple(int(value) for value in self.coordinates.shape[:-1])

    @property
    def sampling_coordinates(self) -> torch.Tensor:
        """Return persistent sample labels, falling back to coordinate values."""
        return self.coordinates if self.sample_coordinates is None else self.sample_coordinates

    def topology_shapes(self, *, default_domain_rank: int | None = None) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return ``(domain_shape, batch_shape)`` for sampler diagnostics."""
        prefix_shape = self.prefix_shape
        if self.domain_shape is not None:
            rank = len(self.domain_shape)
            return self.domain_shape, prefix_shape[:-rank] if rank else prefix_shape
        if default_domain_rank is None:
            return prefix_shape, ()
        rank = int(default_domain_rank)
        if rank < 0 or rank > len(prefix_shape):
            raise ValueError(f"coordinate prefix shape {prefix_shape} has fewer axes than required domain rank {rank}")
        return prefix_shape[-rank:] if rank else (), prefix_shape[:-rank] if rank else prefix_shape

    def with_coordinates(self, coordinates: torch.Tensor) -> "CoordinateFieldInput":
        """Replace transformed values while retaining sample identity and topology."""
        return CoordinateFieldInput(
            coordinates=coordinates,
            sample_coordinates=self.sample_coordinates,
            domain_shape=self.domain_shape,
        )

    def retain_sample_identity(self) -> "CoordinateFieldInput":
        """Persist the current sampling positions as sample labels."""
        if self.sample_coordinates is not None:
            return self
        return CoordinateFieldInput(
            coordinates=self.coordinates,
            sample_coordinates=self.coordinates,
            domain_shape=self.domain_shape,
        )


CoordinateLike: TypeAlias = Union[torch.Tensor, CoordinateFieldInput]


def as_coordinate_field_input(
    value: torch.Tensor | CoordinateFieldInput,
    *,
    sample_coordinates: torch.Tensor | None = None,
    domain_shape: Sequence[int] | None = None,
) -> CoordinateFieldInput:
    """Normalize tensor and structured input forms to one immutable contract."""
    if isinstance(value, CoordinateFieldInput):
        if sample_coordinates is not None or domain_shape is not None:
            raise ValueError("sample_coordinates and domain_shape cannot override a CoordinateFieldInput")
        return value
    return CoordinateFieldInput(
        coordinates=value,
        sample_coordinates=sample_coordinates,
        domain_shape=None if domain_shape is None else tuple(int(item) for item in domain_shape),
    )


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} entries must be positive, got {value}")
    return value
