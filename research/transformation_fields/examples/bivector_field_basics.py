"""Learn a spatially varying 2D Clifford rotation field from target coordinates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from clifra.core.runtime.algebra import AlgebraContext
from research.transformation_fields import (
    CoordinateFieldInput,
    InvertibleBivectorField,
    RBFGeneratorSampler,
    TargetFieldCriterion,
    TransformationFieldEngine,
)

GRID_SIZE = 13
CONTROL_SIZE = 5
FIT_STEPS = 500
DTYPE = torch.float64
ASSERT_TOLERANCE = 2e-10

def regular_grid(size: int) -> torch.Tensor:
    """Return a square [-1, 1]^2 coordinate grid."""
    axis = torch.linspace(-1.0, 1.0, size, dtype=DTYPE)
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack((x, y), dim=-1)


def analytic_angles(coordinates: torch.Tensor) -> torch.Tensor:
    """Define the target spatially varying rotation angle."""
    x, y = coordinates.unbind(dim=-1)
    return 0.70 + 0.32 * torch.sin(torch.pi * x / 2.0) * torch.cos(
        torch.pi * y / 2.0
    )

def rotate(coordinates: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply the analytic 2D rotation independently of the learned field."""
    cosine = angles.cos()
    sine = angles.sin()
    x, y = coordinates.unbind(dim=-1)

    return torch.stack(
        (
            cosine * x - sine * y,
            sine * x + cosine * y,
        ),
        dim=-1,
    )

def point_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return Euclidean RMSE over points."""
    residual_norm = torch.linalg.vector_norm(prediction - target, dim=-1)
    return residual_norm.square().mean().sqrt()


def make_control_points(size: int) -> torch.Tensor:
    """Return a regular RBF control lattice over the sampling domain."""
    axis = torch.linspace(-1.0, 1.0, size, dtype=DTYPE)
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)

def verify_structure(
    field: InvertibleBivectorField,
    field_input: CoordinateFieldInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify indexed inversion and sample-permutation equivariance."""
    source = field_input.coordinates
    labels = field_input.sampling_coordinates

    with torch.no_grad():
        state = field.state(field_input)

        reconstructed = field.inverse(state.inverse_input())
        inverse_error = (reconstructed - source).abs().amax()

        flat_source = source.reshape(-1, source.shape[-1])
        flat_labels = labels.reshape(-1, labels.shape[-1])

        baseline_input = CoordinateFieldInput(
            flat_source,
            sample_coordinates=flat_labels,
        )
        baseline = field(baseline_input)

        permutation = torch.randperm(flat_source.shape[0])

        permuted_input = CoordinateFieldInput(
            flat_source[permutation],
            sample_coordinates=flat_labels[permutation],
        )
        permuted = field(permuted_input)

        permutation_error = (
            permuted - baseline[permutation]
        ).abs().amax()

        permuted_state = field.state(permuted_input)
        permuted_reconstructed = field.inverse(
            permuted_state.inverse_input()
        )
        permuted_inverse_error = (
            permuted_reconstructed - flat_source[permutation]
        ).abs().amax()

    torch.testing.assert_close(
        reconstructed,
        source,
        atol=ASSERT_TOLERANCE,
        rtol=ASSERT_TOLERANCE,
    )
    torch.testing.assert_close(
        permuted,
        baseline[permutation],
        atol=ASSERT_TOLERANCE,
        rtol=ASSERT_TOLERANCE,
    )
    torch.testing.assert_close(
        permuted_reconstructed,
        flat_source[permutation],
        atol=ASSERT_TOLERANCE,
        rtol=ASSERT_TOLERANCE,
    )

    return inverse_error, permutation_error, permuted_inverse_error


def plot_results(
    source: torch.Tensor,
    target: torch.Tensor,
    learned: torch.Tensor,
    learned_angles: torch.Tensor,
    control_points: torch.Tensor,
    output_path: Path,
) -> None:
    """Visualize target action, learned action, and learned generator field."""
    source = source.cpu()
    target = target.cpu()
    learned = learned.cpu()
    learned_angles = learned_angles.cpu()
    control_points = control_points.cpu()

    source_flat = source.reshape(-1, 2)
    target_flat = target.reshape(-1, 2)
    learned_flat = learned.reshape(-1, 2)

    x = source[..., 0]
    y = source[..., 1]

    # Infinitesimal vector field induced by the learned e12 coefficient.
    generator_x = -learned_angles * y
    generator_y = learned_angles * x

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        sharex=True,
        sharey=True,
    )

    for axis, transformed, title, color in (
        (
            axes[0],
            target_flat,
            "Analytic target transformation",
            "tab:blue",
        ),
        (
            axes[1],
            learned_flat,
            "Learned target transformation",
            "tab:orange",
        ),
    ):
        axis.scatter(
            source_flat[:, 0],
            source_flat[:, 1],
            s=9,
            color="0.65",
            label="source",
        )
        axis.quiver(
            source_flat[:, 0],
            source_flat[:, 1],
            transformed[:, 0] - source_flat[:, 0],
            transformed[:, 1] - source_flat[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color=color,
        )
        axis.scatter(
            transformed[:, 0],
            transformed[:, 1],
            s=7,
            color=color,
            label="transformed",
        )
        axis.set_title(title)
        axis.legend(loc="upper left", fontsize=8)

    generator_plot = axes[2].quiver(
        x,
        y,
        generator_x,
        generator_y,
        learned_angles,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
    )

    axes[2].scatter(
        control_points[:, 0],
        control_points[:, 1],
        marker="x",
        color="black",
        label="RBF controls",
    )
    axes[2].set_title("Learned e12 coefficient and induced field")
    axes[2].legend(loc="upper left", fontsize=8)

    figure.colorbar(
        generator_plot,
        ax=axes[2],
        label="e12 coefficient (radians)",
    )

    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlim(-1.35, 1.35)
        axis.set_ylim(-1.35, 1.35)
        axis.set_xlabel("x")

    axes[0].set_ylabel("y")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(7)

    # Domain values and persistent sampling identities.
    source = regular_grid(GRID_SIZE)
    labels = source.clone()

    field_input = CoordinateFieldInput(
        source,
        sample_coordinates=labels,
        domain_shape=(GRID_SIZE, GRID_SIZE),
    )

    # Independent analytic target.
    target_angles = analytic_angles(source)
    target = rotate(source, target_angles)

    # Learned Cl(2,0) bivector field.
    control_points = make_control_points(CONTROL_SIZE)
    algebra = AlgebraContext(2, 0, 0, dtype=DTYPE)

    field = InvertibleBivectorField(
        algebra,
        coordinate_dim=2,
        path_steps=1,
        generator_sampler=RBFGeneratorSampler(
            control_points,
            length_scale=0.42,
        ),
        init_scale=0.02,
    )

    engine = TransformationFieldEngine(
        field,
        target_criterion=TargetFieldCriterion(target),
    )

    engine.fit(
        field_input,
        steps=FIT_STEPS,
        lr=0.08,
        log_every=FIT_STEPS,
    )

    # Learned transformation and generator field.
    with torch.no_grad():
        state = field.state(field_input)
        learned = state.transformed_coordinates

        e12_index = field.bivector_layout.basis_indices.index(0b11)
        learned_angles = state.generator_weights[
            0, ..., e12_index
        ]

        fit_rmse = point_rmse(learned, target)
        generator_rmse = (
            learned_angles - target_angles
        ).square().mean().sqrt()

    # Structural properties of the field.
    (
        inverse_error,
        permutation_error,
        permuted_inverse_error,
    ) = verify_structure(field, field_input)

    output_path = (
        Path(__file__).resolve().parents[3]
        / "outputs"
        / "bivector_field_basics.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_results(
        source,
        target,
        learned,
        learned_angles,
        control_points,
        output_path,
    )

    print("Cl(2,0) RBF bivector field")
    print()
    print("fit")
    print(f"  point RMSE:             {fit_rmse:.3e}")
    print(f"  generator-angle RMSE:   {generator_rmse:.3e}")
    print()
    print("structure")
    print(f"  indexed inverse max:    {inverse_error:.3e}")
    print(f"  permutation equiv max:  {permutation_error:.3e}")
    print(f"  permuted inverse max:    {permuted_inverse_error:.3e}")
    print()
    print(f"figure: {output_path}")


if __name__ == "__main__":
    main()