"""Optimize a 3D cantilever shape with a Clifford Transformation Field.

Run from the repository root::

    uv run --group fem --group viz python research/transformation_fields/examples/transformation_field_fem_design.py

The optimization path is deliberately compact and local to this example:

``Transformation Field -> design coordinates -> differentiable Tet4 FEM``.

``torch-fem`` is intentionally used before and after optimization as an
independent FEM implementation cross-check of the same Tet4 problem. It only
receives detached design coordinates and never receives internal displacements.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifra.core.runtime.algebra import AlgebraContext
from research.transformation_fields import (
    CoordinateFieldInput,
    InvertibleBivectorField,
    RBFGeneratorSampler,
)

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


@dataclass(frozen=True)
class DesignSpec:
    """All numerical choices for this one experiment."""

    length: float = 2.0
    width: float = 0.45
    height: float = 0.45
    cells_x: int = 6
    cells_y: int = 2
    cells_z: int = 2
    controls_x: int = 4
    controls_y: int = 2
    controls_z: int = 2
    young: float = 20_000.0
    poisson: float = 0.30
    total_load_z: float = -1.0
    target_tip_y: float = 0.012
    target_tip_z: float = -0.075
    response_scale: float = 0.010
    steps: int = 180
    learning_rate: float = 0.035
    max_design_displacement: float = 0.80
    min_quality: float = 0.13
    quality_guard_band: float = 0.01
    seed: int = 17


@dataclass(frozen=True)
class Mesh:
    """Topology and semantic sets defined once on the reference mesh."""

    nodes: torch.Tensor
    elements: torch.Tensor
    fixed_nodes: torch.Tensor
    loaded_nodes: torch.Tensor
    observation_nodes: torch.Tensor
    boundary_faces: torch.Tensor
    boundary_face_elements: torch.Tensor


@dataclass
class FEMResult:
    displacement: torch.Tensor
    reactions: torch.Tensor
    compliance: torch.Tensor
    strain_energy: torch.Tensor
    response: torch.Tensor
    stress: torch.Tensor
    von_mises: torch.Tensor
    strain: torch.Tensor | None = None
    displacement_gradient: torch.Tensor | None = None


def make_mesh(spec: DesignSpec) -> Mesh:
    """Make a structured block and split every cell into six positive Tet4s."""
    x = torch.linspace(0.0, spec.length, spec.cells_x + 1, dtype=DTYPE)
    y = torch.linspace(-spec.width / 2.0, spec.width / 2.0, spec.cells_y + 1, dtype=DTYPE)
    z = torch.linspace(-spec.height / 2.0, spec.height / 2.0, spec.cells_z + 1, dtype=DTYPE)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    nodes = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)

    def node(i: int, j: int, k: int) -> int:
        return (k * (spec.cells_y + 1) + j) * (spec.cells_x + 1) + i

    raw_elements: list[list[int]] = []
    for k in range(spec.cells_z):
        for j in range(spec.cells_y):
            for i in range(spec.cells_x):
                v0, v1, v2, v3 = node(i, j, k), node(i + 1, j, k), node(i, j + 1, k), node(i + 1, j + 1, k)
                v4, v5, v6, v7 = (
                    node(i, j, k + 1),
                    node(i + 1, j, k + 1),
                    node(i, j + 1, k + 1),
                    node(i + 1, j + 1, k + 1),
                )
                raw_elements.extend(((v0, v1, v3, v7), (v0, v3, v2, v7), (v0, v2, v6, v7), (v0, v6, v4, v7), (v0, v4, v5, v7), (v0, v5, v1, v7)))
    elements = torch.tensor(raw_elements, dtype=torch.long)
    elements = orient_tetrahedra(nodes, elements)

    tolerance = 1e-12
    fixed_nodes = torch.where(nodes[:, 0] < tolerance)[0]
    loaded_nodes = torch.where(nodes[:, 0] > spec.length - tolerance)[0]
    faces, owners = exterior_faces(elements)
    return Mesh(nodes, elements, fixed_nodes, loaded_nodes, loaded_nodes, faces, owners)


def orient_tetrahedra(nodes: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
    """Ensure the reference node order has positive signed volume."""
    coords = nodes[elements]
    signed_six_volume = torch.linalg.det(coords[:, 1:] - coords[:, :1])
    result = elements.clone()
    negative = signed_six_volume < 0.0
    result[negative, 2], result[negative, 3] = elements[negative, 3], elements[negative, 2]
    if not torch.all(tetra_volumes(nodes, result) > 0.0):
        raise RuntimeError("structured tetrahedral subdivision is not positively oriented")
    return result


def exterior_faces(elements: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unshared Tet4 faces and their parent element ids."""
    local_faces = torch.tensor(((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)), dtype=torch.long)
    faces = elements[:, local_faces].reshape(-1, 3)
    owners = torch.arange(len(elements), dtype=torch.long).repeat_interleave(4)
    canonical = torch.sort(faces, dim=1).values
    _, inverse, counts = torch.unique(canonical, dim=0, return_inverse=True, return_counts=True)
    exterior = counts[inverse] == 1
    return faces[exterior], owners[exterior]


def tetra_volumes(nodes: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
    coordinates = nodes[elements]
    return torch.linalg.det(coordinates[:, 1:] - coordinates[:, :1]) / 6.0


def tetra_quality(nodes: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
    """Mean-edge normalized signed quality: one for an equilateral tetrahedron."""
    x = nodes[elements]
    edge_pairs = torch.tensor(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), device=x.device)
    edge_squared = (x[:, edge_pairs[:, 0]] - x[:, edge_pairs[:, 1]]).square().sum(dim=-1)
    mean_edge_squared = edge_squared.mean(dim=-1).clamp_min(torch.finfo(x.dtype).eps)
    return 6.0 * 2.0**0.5 * tetra_volumes(nodes, elements) / mean_edge_squared.pow(1.5)


def elastic_matrix(spec: DesignSpec, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """3D isotropic stiffness in engineering-strain Voigt order."""
    e, nu = torch.as_tensor(spec.young, device=device, dtype=dtype), torch.as_tensor(spec.poisson, device=device, dtype=dtype)
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    shear = e / (2.0 * (1.0 + nu))
    d = torch.zeros((6, 6), device=device, dtype=dtype)
    d[:3, :3] = lam
    d[range(3), range(3)] += 2.0 * shear
    d[3:, 3:] = torch.eye(3, device=device, dtype=dtype) * shear
    return d


def tet_kinematics(nodes: torch.Tensor, elements: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Tet4 strain-displacement matrices, volumes, and shape gradients."""
    x = nodes[elements]
    a = torch.cat((torch.ones_like(x[..., :1]), x), dim=-1)
    inv_a = torch.linalg.inv(a)
    gradients = inv_a[:, 1:, :].transpose(1, 2)  # [element, local node, xyz]
    b = x.new_zeros((len(elements), 6, 12))
    for node_index in range(4):
        gx, gy, gz = gradients[:, node_index].unbind(dim=-1)
        column = 3 * node_index
        b[:, 0, column] = gx
        b[:, 1, column + 1] = gy
        b[:, 2, column + 2] = gz
        b[:, 3, column], b[:, 3, column + 1] = gy, gx
        b[:, 4, column + 1], b[:, 4, column + 2] = gz, gy
        b[:, 5, column], b[:, 5, column + 2] = gz, gx
    return b, tetra_volumes(nodes, elements), gradients


def solve_internal(nodes: torch.Tensor, mesh: Mesh, spec: DesignSpec) -> FEMResult:
    """A direct, autograd-friendly small-strain linear Tet4 solve."""
    b, volumes, gradients = tet_kinematics(nodes, mesh.elements)
    if torch.any(volumes <= 0.0):
        raise ValueError("internal FEM received an inverted tetrahedron")
    d = elastic_matrix(spec, device=nodes.device, dtype=nodes.dtype)
    element_stiffness = torch.einsum("eji,jk,ekl,e->eil", b, d, b, volumes)
    node_dofs = mesh.elements.to(nodes.device).unsqueeze(-1) * 3 + torch.arange(3, device=nodes.device)
    dofs = node_dofs.reshape(-1, 12)
    row = dofs.unsqueeze(-1).expand(-1, 12, 12).reshape(-1)
    column = dofs.unsqueeze(1).expand(-1, 12, 12).reshape(-1)
    stiffness = nodes.new_zeros((nodes.shape[0] * 3, nodes.shape[0] * 3)).index_put(
        (row, column), element_stiffness.reshape(-1), accumulate=True
    )
    force = nodes.new_zeros((nodes.shape[0], 3))
    force = force.index_put(
        (mesh.loaded_nodes.to(nodes.device), torch.full_like(mesh.loaded_nodes, 2, device=nodes.device)),
        nodes.new_full((len(mesh.loaded_nodes),), spec.total_load_z / len(mesh.loaded_nodes)),
    ).reshape(-1)
    fixed_dofs = (mesh.fixed_nodes.to(nodes.device).unsqueeze(-1) * 3 + torch.arange(3, device=nodes.device)).reshape(-1)
    free = torch.ones(nodes.numel(), dtype=torch.bool, device=nodes.device)
    free[fixed_dofs] = False
    u_free = torch.linalg.solve(stiffness[free][:, free], force[free])
    displacement = nodes.new_zeros(nodes.numel()).index_put((torch.where(free)[0],), u_free).reshape(-1, 3)
    reaction = (stiffness @ displacement.reshape(-1) - force).reshape(-1, 3)
    local_displacement = displacement[mesh.elements.to(nodes.device)].reshape(-1, 12)
    strain = torch.einsum("eij,ej->ei", b, local_displacement)
    displacement_gradient = torch.einsum(
        "eni,enj->eij", displacement[mesh.elements.to(nodes.device)], gradients
    )
    stress_voigt = torch.einsum("ij,ej->ei", d, strain)
    stress = voigt_to_tensor(stress_voigt)
    von_mises = von_mises_stress(stress)
    return FEMResult(
        displacement=displacement,
        reactions=reaction,
        compliance=torch.dot(force, displacement.reshape(-1)),
        strain_energy=0.5 * torch.dot(displacement.reshape(-1), stiffness @ displacement.reshape(-1)),
        response=displacement[mesh.observation_nodes.to(nodes.device)].mean(dim=0),
        stress=stress,
        von_mises=von_mises,
        strain=strain,
        displacement_gradient=displacement_gradient,
    )


def voigt_to_tensor(stress: torch.Tensor) -> torch.Tensor:
    result = stress.new_zeros((*stress.shape[:-1], 3, 3))
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = stress[..., 0], stress[..., 1], stress[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = stress[..., 3]
    result[..., 1, 2] = result[..., 2, 1] = stress[..., 4]
    result[..., 0, 2] = result[..., 2, 0] = stress[..., 5]
    return result


def strain_voigt_to_tensor(strain: torch.Tensor) -> torch.Tensor:
    """Convert engineering-strain Voigt vectors to symmetric strain tensors."""
    result = strain.new_zeros((*strain.shape[:-1], 3, 3))
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = strain[..., 0], strain[..., 1], strain[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = 0.5 * strain[..., 3]
    result[..., 1, 2] = result[..., 2, 1] = 0.5 * strain[..., 4]
    result[..., 0, 2] = result[..., 2, 0] = 0.5 * strain[..., 5]
    return result


def von_mises_stress(stress: torch.Tensor) -> torch.Tensor:
    mean = torch.diagonal(stress, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
    deviator = stress - torch.eye(3, device=stress.device, dtype=stress.dtype) * mean[..., None]
    return (1.5 * deviator.square().sum(dim=(-1, -2))).clamp_min(0.0).sqrt()


def solve_torch_fem(nodes: torch.Tensor, mesh: Mesh, spec: DesignSpec) -> FEMResult:
    """Solve the independently assembled reference problem with torch-fem.

    The import is local so importing this file does not require the optional
    dependency.  The command at the top of this file selects the ``fem`` group.
    """
    try:
        from torchfem import Solid
        from torchfem.materials import IsotropicElasticity3D
    except ImportError as error:  # pragma: no cover - depends on optional group
        raise RuntimeError("install the optional FEM group: uv sync --group fem") from error

    torch_fem_nodes = nodes.detach().clone().cpu()
    material = IsotropicElasticity3D(
        torch.tensor(spec.young, dtype=torch_fem_nodes.dtype),
        torch.tensor(spec.poisson, dtype=torch_fem_nodes.dtype),
    )
    model = Solid(torch_fem_nodes, mesh.elements.cpu(), material)
    model.constraints[mesh.fixed_nodes.cpu(), :] = True
    model.forces[mesh.loaded_nodes.cpu(), 2] = spec.total_load_z / len(mesh.loaded_nodes)
    displacement, assembled_force, stress, _, _ = model.solve(method="spsolve")
    if stress.ndim == 4:
        stress = stress.mean(dim=1)
    reactions = assembled_force - model.forces
    return FEMResult(
        displacement=displacement,
        reactions=reactions,
        compliance=torch.sum(model.forces * displacement),
        strain_energy=0.5 * torch.sum(model.forces * displacement),
        response=displacement[mesh.observation_nodes.cpu()].mean(dim=0),
        stress=stress,
        von_mises=von_mises_stress(stress),
    )


def control_points(spec: DesignSpec) -> torch.Tensor:
    x = torch.linspace(0.0, spec.length, spec.controls_x, dtype=DTYPE)
    y = torch.linspace(-spec.width / 2.0, spec.width / 2.0, spec.controls_y, dtype=DTYPE)
    z = torch.linspace(-spec.height / 2.0, spec.height / 2.0, spec.controls_z, dtype=DTYPE)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


def make_field(spec: DesignSpec) -> InvertibleBivectorField:
    algebra = AlgebraContext(3, 0, 0, dtype=DTYPE)
    return InvertibleBivectorField(
        algebra,
        coordinate_dim=3,
        path_steps=1,
        generator_sampler=RBFGeneratorSampler(control_points(spec), length_scale=0.42),
        init_scale=2e-3,
    )


def transform_design(field: InvertibleBivectorField, mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform points while sampling generators at persistent material labels."""
    state = field.state(CoordinateFieldInput(mesh.nodes, sample_coordinates=mesh.nodes))
    return state.transformed_coordinates, state.generator_weights[0]


def target_response(spec: DesignSpec) -> torch.Tensor:
    """Ask for a small lateral motion without losing useful downward compliance."""
    return torch.tensor((0.0, spec.target_tip_y, spec.target_tip_z), dtype=DTYPE)


def policy_terms(
    nodes: torch.Tensor,
    generator_weights: torch.Tensor,
    field: InvertibleBivectorField,
    mesh: Mesh,
    spec: DesignSpec,
    reference_volume: torch.Tensor,
) -> dict[str, torch.Tensor]:
    quality = tetra_quality(nodes, mesh.elements)
    volumes = tetra_volumes(nodes, mesh.elements)
    design_displacement = nodes - mesh.nodes.to(nodes.device)
    anchor = generator_weights[mesh.fixed_nodes.to(nodes.device)].square().mean()
    generator_energy = field.bivectors.square().mean()
    return {
        "anchor": anchor,
        "volume": ((volumes.sum() / reference_volume - 1.0) / 0.03).square(),
        "quality": torch.relu(spec.min_quality + spec.quality_guard_band - quality).square().mean() / spec.min_quality**2,
        "envelope": torch.relu(torch.linalg.vector_norm(design_displacement, dim=-1) - spec.max_design_displacement).square().mean(),
        "generator_energy": generator_energy,
        "minimum_quality": quality.min(),
        "minimum_volume": volumes.min(),
        "relative_volume": volumes.sum() / reference_volume,
        "maximum_design_displacement": torch.linalg.vector_norm(design_displacement, dim=-1).max(),
    }


def loss_for_design(
    field: InvertibleBivectorField,
    mesh: Mesh,
    spec: DesignSpec,
    reference_volume: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, FEMResult, torch.Tensor, dict[str, torch.Tensor]]:
    nodes, weights = transform_design(field, mesh)
    fem = solve_internal(nodes, mesh, spec)
    response = ((fem.response - target.to(nodes)) / spec.response_scale).square().mean()
    policies = policy_terms(nodes, weights, field, mesh, spec, reference_volume)
    loss = response + 20.0 * policies["anchor"] + 1.5 * policies["volume"] + 1_000.0 * policies["quality"] + 25.0 * policies["envelope"] + 0.02 * policies["generator_energy"]
    return loss, fem, nodes, {"response": response, **policies}


def gradient_check(field: InvertibleBivectorField, mesh: Mesh, spec: DesignSpec, reference_volume: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compare a few generator derivatives against central finite differences."""
    field.zero_grad(set_to_none=True)
    loss, _, _, _ = loss_for_design(field, mesh, spec, reference_volume, target)
    loss.backward()
    analytic = field.bivectors.grad.detach().clone()
    epsilon = 2e-5
    samples = ((0, 0, 0), (0, field.bivectors.shape[1] // 2, 1), (0, field.bivectors.shape[1] - 1, 2))
    relative_errors: list[float] = []
    with torch.no_grad():
        for index in samples:
            original = field.bivectors[index].item()
            field.bivectors[index] = original + epsilon
            plus = loss_for_design(field, mesh, spec, reference_volume, target)[0]
            field.bivectors[index] = original - epsilon
            minus = loss_for_design(field, mesh, spec, reference_volume, target)[0]
            field.bivectors[index] = original
            finite_difference = (plus - minus) / (2.0 * epsilon)
            denominator = max(1e-8, abs(float(finite_difference)), abs(float(analytic[index])))
            relative_errors.append(abs(float(finite_difference - analytic[index])) / denominator)
    field.zero_grad(set_to_none=True)
    return {"max_relative_error": max(relative_errors), "mean_relative_error": sum(relative_errors) / len(relative_errors)}


def optimize(field: InvertibleBivectorField, mesh: Mesh, spec: DesignSpec, reference_volume: torch.Tensor, target: torch.Tensor) -> tuple[list[dict[str, float]], list[torch.Tensor]]:
    optimizer = torch.optim.Adam(field.parameters(), lr=spec.learning_rate)
    history: list[dict[str, float]] = []
    snapshots: list[torch.Tensor] = []
    capture_steps = set(round(value * (spec.steps - 1) / 4) for value in range(5))
    for step in range(spec.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, fem, nodes, terms = loss_for_design(field, mesh, spec, reference_volume, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss),
                    "response_loss": float(terms["response"]),
                    "tip_y": float(fem.response[1]),
                    "tip_z": float(fem.response[2]),
                    "minimum_quality": float(terms["minimum_quality"]),
                    "relative_volume": float(terms["relative_volume"]),
                }
            )
            if step in capture_steps:
                snapshots.append(nodes.detach().clone())
    return history, snapshots


def torch_fem_comparison(internal: FEMResult, torch_fem: FEMResult) -> dict[str, float]:
    """Compare observables from the internal and detached-geometry torch-fem solves."""
    response_error = torch.linalg.vector_norm(internal.response - torch_fem.response)
    response_size = torch.linalg.vector_norm(torch_fem.response).clamp_min(1e-12)
    return {
        "response_absolute_error": float(response_error),
        "response_relative_error": float(response_error / response_size),
        "compliance_relative_error": float(abs(internal.compliance - torch_fem.compliance) / torch_fem.compliance.abs().clamp_min(1e-12)),
    }


def small_strain_diagnostics(result: FEMResult) -> dict[str, float | bool]:
    """Report physical small-strain measures about the current design geometry."""
    if result.strain is None or result.displacement_gradient is None:
        raise ValueError("small-strain diagnostics require an internal FEM result")
    strain = strain_voigt_to_tensor(result.strain)
    principal = torch.linalg.eigvalsh(strain)
    rotation = 0.5 * (result.displacement_gradient - result.displacement_gradient.transpose(-1, -2))
    values = torch.cat(
        (
            principal.reshape(-1),
            torch.linalg.matrix_norm(strain, dim=(-2, -1)),
            torch.linalg.matrix_norm(result.displacement_gradient, dim=(-2, -1)),
            torch.linalg.matrix_norm(rotation, dim=(-2, -1)),
        )
    )
    return {
        "maximum_principal_strain": float(principal.max()),
        "minimum_principal_strain": float(principal.min()),
        "maximum_absolute_principal_strain": float(principal.abs().max()),
        "maximum_strain_frobenius_norm": float(torch.linalg.matrix_norm(strain, dim=(-2, -1)).max()),
        "maximum_displacement_gradient_frobenius_norm": float(
            torch.linalg.matrix_norm(result.displacement_gradient, dim=(-2, -1)).max()
        ),
        "maximum_infinitesimal_rotation_frobenius_norm": float(torch.linalg.matrix_norm(rotation, dim=(-2, -1)).max()),
        "all_diagnostics_finite": bool(torch.isfinite(values).all()),
    }


def transformation_field_diagnostics(field: InvertibleBivectorField, mesh: Mesh) -> dict[str, float]:
    """Measure indexed forward/inverse consistency for the persistent mesh labels."""
    field_input = CoordinateFieldInput(mesh.nodes, sample_coordinates=mesh.nodes)
    with torch.no_grad():
        state = field.state(field_input)
        reconstructed = field.inverse(state.inverse_input())
        residual = reconstructed - mesh.nodes
        generator_norms = torch.linalg.vector_norm(state.generator_weights[0], dim=-1)
    return {
        "inverse_rmse": float(residual.square().mean().sqrt()),
        "inverse_max_abs": float(residual.abs().max()),
        "max_generator_norm": float(generator_norms.max()),
        "mean_generator_norm": float(generator_norms.mean()),
    }


def write_vtu(path: Path, nodes: torch.Tensor, mesh: Mesh, *, point_data: dict[str, torch.Tensor], cell_data: dict[str, torch.Tensor]) -> None:
    """Write only the arrays useful to inspect this example, with no mesh I/O dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = nodes.detach().cpu()
    elements = mesh.elements.detach().cpu()

    def values(data: torch.Tensor) -> str:
        return " ".join(f"{float(value):.12g}" for value in data.detach().cpu().reshape(-1))

    point_arrays = "\n".join(
        f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{tensor.shape[-1] if tensor.ndim > 1 else 1}" format="ascii">{values(tensor)}</DataArray>'
        for name, tensor in point_data.items()
    )
    cell_arrays = "\n".join(
        f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{tensor.shape[-1] if tensor.ndim > 1 else 1}" format="ascii">{values(tensor)}</DataArray>'
        for name, tensor in cell_data.items()
    )
    connectivity = " ".join(str(int(value)) for value in elements.reshape(-1))
    offsets = " ".join(str(4 * (index + 1)) for index in range(len(elements)))
    cell_types = " ".join("10" for _ in range(len(elements)))
    path.write_text(
        f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid><Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(elements)}">
    <PointData>{point_arrays}</PointData><CellData>{cell_arrays}</CellData>
    <Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">{values(points)}</DataArray></Points>
    <Cells><DataArray type="Int64" Name="connectivity" format="ascii">{connectivity}</DataArray>
      <DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray>
      <DataArray type="UInt8" Name="types" format="ascii">{cell_types}</DataArray></Cells>
  </Piece></UnstructuredGrid>
</VTKFile>'''
    )


def write_artifacts(
    output: Path,
    mesh: Mesh,
    reference_nodes: torch.Tensor,
    optimized_nodes: torch.Tensor,
    field: InvertibleBivectorField,
    internal: FEMResult,
    reference_torch_fem: FEMResult,
    optimized_torch_fem: FEMResult,
) -> None:
    boundary_role = torch.zeros(len(reference_nodes), dtype=DTYPE)
    boundary_role[mesh.fixed_nodes] = 1.0
    boundary_role[mesh.loaded_nodes] = 2.0
    generator_magnitude = torch.linalg.vector_norm(field.weights_for_input(CoordinateFieldInput(reference_nodes, sample_coordinates=reference_nodes))[0], dim=-1)
    write_vtu(
        output / "reference_mesh.vtu",
        reference_nodes,
        mesh,
        point_data={"boundary_role": boundary_role},
        cell_data={"element_quality": tetra_quality(reference_nodes, mesh.elements)},
    )
    write_vtu(
        output / "optimized_mesh.vtu",
        optimized_nodes,
        mesh,
        point_data={
            "design_displacement": optimized_nodes - reference_nodes,
            "generator_magnitude": generator_magnitude,
            "boundary_role": boundary_role,
        },
        cell_data={"element_quality": tetra_quality(optimized_nodes, mesh.elements)},
    )
    for stem, result_nodes, result in (
        ("internal_optimized_result", optimized_nodes, internal),
        ("torch_fem_reference_result", reference_nodes, reference_torch_fem),
        ("torch_fem_optimized_result", optimized_nodes, optimized_torch_fem),
    ):
        write_vtu(
            output / f"{stem}.vtu",
            result_nodes,
            mesh,
            point_data={
                "physical_displacement": result.displacement,
                "displacement_magnitude": torch.linalg.vector_norm(result.displacement, dim=-1),
            },
            cell_data={
                "von_mises_stress": result.von_mises,
                "element_quality": tetra_quality(result_nodes, mesh.elements),
            },
        )


def visualizations(
    output: Path,
    mesh: Mesh,
    spec: DesignSpec,
    reference_nodes: torch.Tensor,
    optimized_nodes: torch.Tensor,
    reference: FEMResult,
    optimized: FEMResult,
    snapshots: Iterable[torch.Tensor],
    history: list[dict[str, float]],
) -> list[str]:
    """Create the concise evidence set described in the experiment plan."""
    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return ["visualization dependencies absent; use --group viz"]

    faces = mesh.boundary_faces.cpu().numpy()
    owners = mesh.boundary_face_elements.cpu()
    fixed = mesh.fixed_nodes.cpu().numpy()
    loaded = mesh.loaded_nodes.cpu().numpy()

    def camera_limits(points: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        p = points.detach().cpu()
        center = p.mean(dim=0)
        radius = float((p.max(dim=0).values - p.min(dim=0).values).max()) * 0.62
        return (
            (float(center[0] - radius), float(center[0] + radius)),
            (float(center[1] - radius), float(center[1] + radius)),
            (float(center[2] - radius), float(center[2] + radius)),
        )

    def set_equal_axes(axis, limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
        axis.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")

    def add_reference_ghost(axis, reference_outline: torch.Tensor | None) -> None:
        if reference_outline is None:
            return
        outline = reference_outline.detach().cpu().numpy()
        axis.add_collection3d(
            Poly3DCollection(outline[faces], facecolors="none", edgecolors="0.82", alpha=0.45, linewidths=0.45)
        )

    def draw_unloaded_design(
        axis,
        nodes: torch.Tensor,
        title: str,
        limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        *,
        reference_outline: torch.Tensor | None = None,
    ) -> None:
        unloaded = nodes.detach().cpu().numpy()
        add_reference_ghost(axis, reference_outline)
        axis.add_collection3d(
            Poly3DCollection(unloaded[faces], facecolors="#6baed6", edgecolors="#236192", alpha=0.85, linewidths=0.35)
        )
        axis.scatter(unloaded[fixed, 0], unloaded[fixed, 1], unloaded[fixed, 2], color="tab:green", s=14, label="fixed")
        tip = unloaded[loaded].mean(axis=0)
        axis.quiver(*tip, 0.0, 0.0, -0.20, color="tab:red", arrow_length_ratio=0.18, linewidth=1.6)
        axis.set_title(title)
        set_equal_axes(axis, limits)
        axis.view_init(elev=23, azim=-58)

    def draw_physical_response(
        axis,
        nodes: torch.Tensor,
        result: FEMResult,
        title: str,
        limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        *,
        element_field: torch.Tensor | None = None,
        reference_outline: torch.Tensor | None = None,
    ) -> None:
        unloaded = nodes.detach().cpu().numpy()
        deformed = (nodes + 2.5 * result.displacement).detach().cpu().numpy()
        add_reference_ghost(axis, reference_outline)
        axis.add_collection3d(Poly3DCollection(unloaded[faces], facecolors="none", edgecolors="0.55", linewidths=0.35))
        if element_field is None:
            axis.add_collection3d(
                Poly3DCollection(deformed[faces], facecolors="#6baed6", edgecolors="#236192", alpha=0.82, linewidths=0.35)
            )
        else:
            colors = plt.get_cmap("magma")((element_field[owners] / element_field.max().clamp_min(1e-12)).detach().cpu().numpy())
            axis.add_collection3d(Poly3DCollection(deformed[faces], facecolors=colors, edgecolors="0.2", linewidths=0.2))
        axis.scatter(unloaded[fixed, 0], unloaded[fixed, 1], unloaded[fixed, 2], color="tab:green", s=14, label="fixed")
        tip = unloaded[loaded].mean(axis=0)
        axis.quiver(*tip, 0.0, 0.0, -0.20, color="tab:red", arrow_length_ratio=0.18, linewidth=1.6)
        axis.set_title(title)
        set_equal_axes(axis, limits)
        axis.view_init(elev=23, azim=-58)

    reference_deformed = reference_nodes + 2.5 * reference.displacement
    optimized_deformed = optimized_nodes + 2.5 * optimized.displacement
    design_limits = camera_limits(torch.cat((reference_nodes, optimized_nodes)))
    physical_limits = camera_limits(torch.cat((reference_nodes, reference_deformed, optimized_nodes, optimized_deformed)))
    figure = plt.figure(figsize=(12, 5))
    left, right = figure.add_subplot(1, 2, 1, projection="3d"), figure.add_subplot(1, 2, 2, projection="3d")
    draw_unloaded_design(left, reference_nodes, "Reference design: unloaded reference geometry", design_limits)
    draw_unloaded_design(
        right,
        optimized_nodes,
        "Optimized design: unloaded Transformation-Field geometry",
        design_limits,
        reference_outline=reference_nodes,
    )
    left.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "design_comparison.png", dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(12, 5))
    left, right = figure.add_subplot(1, 2, 1, projection="3d"), figure.add_subplot(1, 2, 2, projection="3d")
    draw_physical_response(
        left,
        reference_nodes,
        reference,
        "Reference geometry -> FEM load: physical deformation ×2.5",
        physical_limits,
    )
    draw_physical_response(
        right,
        optimized_nodes,
        optimized,
        "Optimized design geometry -> FEM load: physical deformation ×2.5",
        physical_limits,
        reference_outline=reference_nodes,
    )
    left.legend(loc="upper left", fontsize=8)
    figure.text(0.5, 0.01, "gray wireframe: unloaded geometry; blue surface: physical deformation", ha="center", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    figure.savefig(output / "physical_fem_response.png", dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(9, 5.5))
    axis = figure.add_subplot(111, projection="3d")
    draw_physical_response(
        axis,
        optimized_nodes,
        optimized,
        "Optimized design: von Mises stress; physical deformation ×2.5 (torch-fem)",
        physical_limits,
        element_field=optimized.von_mises,
        reference_outline=reference_nodes,
    )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(0.0, float(optimized.von_mises.max())),
            cmap="magma",
        ),
        ax=axis,
        shrink=0.7,
        pad=0.08,
    )
    colorbar.set_label("von Mises stress")
    figure.tight_layout()
    figure.savefig(output / "fem_response.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    steps = [entry["step"] for entry in history]
    axes[0].plot(steps, [entry["tip_y"] for entry in history], label="tip y")
    axes[0].plot(steps, [entry["tip_z"] for entry in history], label="tip z")
    axes[0].axhline(spec.target_tip_y, color="tab:blue", linestyle="--", linewidth=0.8)
    axes[0].axhline(spec.target_tip_z, color="tab:orange", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="optimization step", ylabel="mean observation displacement")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [entry["minimum_quality"] for entry in history], label="minimum quality")
    axes[1].plot(steps, [entry["relative_volume"] for entry in history], label="relative volume")
    axes[1].set(xlabel="optimization step")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "optimization_history.png", dpi=180)
    plt.close(figure)

    snapshot_nodes = list(snapshots)
    evolution_points = [reference_nodes, *snapshot_nodes]
    evolution_limits = camera_limits(torch.cat(evolution_points))
    frames: list[object] = []
    for index, nodes in enumerate(snapshot_nodes):
        figure = plt.figure(figsize=(5, 4.2))
        axis = figure.add_subplot(111, projection="3d")
        draw_unloaded_design(
            axis,
            nodes,
            f"Unloaded Transformation-Field design {index + 1}/{len(snapshot_nodes)}",
            evolution_limits,
            reference_outline=reference_nodes,
        )
        frame_path = output / f".evolution_{index}.png"
        figure.tight_layout()
        figure.savefig(frame_path, dpi=130)
        plt.close(figure)
        frames.append(imageio.imread(frame_path))
        frame_path.unlink()
    imageio.mimsave(output / "design_evolution.gif", frames, duration=0.75)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, help="override the default optimization length")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "fem_validated_design")
    parser.add_argument("--skip-figures", action="store_true", help="write numerical artifacts only")
    arguments = parser.parse_args()
    spec = DesignSpec(steps=arguments.steps) if arguments.steps is not None else DesignSpec()
    torch.manual_seed(spec.seed)
    output = arguments.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    mesh = make_mesh(spec)
    reference_volume = tetra_volumes(mesh.nodes, mesh.elements).sum()
    reference_internal = solve_internal(mesh.nodes, mesh, spec)
    reference_torch_fem = solve_torch_fem(mesh.nodes, mesh, spec)
    target = target_response(spec)
    field = make_field(spec)
    derivative_check = gradient_check(field, mesh, spec, reference_volume, target)
    history, snapshots = optimize(field, mesh, spec, reference_volume, target)
    with torch.no_grad():
        _, optimized_internal, optimized_nodes, terms = loss_for_design(field, mesh, spec, reference_volume, target)
    optimized_torch_fem = solve_torch_fem(optimized_nodes, mesh, spec)
    field_diagnostics = transformation_field_diagnostics(field, mesh)
    physical_diagnostics = small_strain_diagnostics(optimized_internal)
    reference_comparison = torch_fem_comparison(reference_internal, reference_torch_fem)
    optimized_comparison = torch_fem_comparison(optimized_internal, optimized_torch_fem)
    target_error = torch.linalg.vector_norm(optimized_internal.response - target)
    acceptance = {
        "target_response_error": float(target_error),
        "target_response_within_scale": bool(target_error <= spec.response_scale),
        "minimum_quality_threshold_satisfied": bool(terms["minimum_quality"] >= spec.min_quality),
        "all_volumes_positive": bool(terms["minimum_volume"] > 0.0),
        "indexed_inverse_consistency": bool(field_diagnostics["inverse_max_abs"] <= 2e-10),
        "gradient_check_within_1e-3": bool(derivative_check["max_relative_error"] <= 1e-3),
        "torch_fem_response_cross_check_within_1e-8": bool(optimized_comparison["response_relative_error"] <= 1e-8),
        "torch_fem_compliance_cross_check_within_1e-8": bool(optimized_comparison["compliance_relative_error"] <= 1e-8),
        "small_strain_diagnostics_finite": physical_diagnostics["all_diagnostics_finite"],
    }
    acceptance["passed"] = all(value for key, value in acceptance.items() if key != "target_response_error")
    write_artifacts(
        output,
        mesh,
        mesh.nodes,
        optimized_nodes,
        field,
        optimized_internal,
        reference_torch_fem,
        optimized_torch_fem,
    )
    warnings = (
        []
        if arguments.skip_figures
        else visualizations(
            output,
            mesh,
            spec,
            mesh.nodes,
            optimized_nodes,
            reference_torch_fem,
            optimized_torch_fem,
            snapshots,
            history,
        )
    )
    report = {
        "specification": asdict(spec),
        "transformation_field": {
            "indexed_forward_inverse_consistency": field_diagnostics,
            "generator_magnitude_penalty": float(terms["generator_energy"]),
        },
        "geometry": {
            **{
                key: float(value)
                for key, value in terms.items()
                if key in {"minimum_quality", "minimum_volume", "relative_volume", "maximum_design_displacement"}
            },
            "policy_losses": {
                key: float(terms[key]) for key in {"anchor", "volume", "quality", "envelope"}
            },
        },
        "mechanics": {
            "target_response": [float(value) for value in target],
            "achieved_response": [float(value) for value in optimized_internal.response],
            "target_error": float(target_error),
            "compliance": float(optimized_internal.compliance),
            "maximum_displacement": float(torch.linalg.vector_norm(optimized_internal.displacement, dim=-1).max()),
            "maximum_von_mises": float(optimized_internal.von_mises.max()),
            "small_strain_diagnostics": {
                "about_geometry": "optimized Transformation-Field geometry",
                "excludes_design_geometry_change": True,
                **physical_diagnostics,
            },
        },
        "validation": {
            "torch_fem_cross_check": optimized_comparison,
            "reference_torch_fem_cross_check": reference_comparison,
            "gradient_check": derivative_check,
            "acceptance": acceptance,
        },
        "visualization_warnings": warnings,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Transformation-Field FEM design completed")
    print(f"  target response: {report['mechanics']['target_response']}")
    print(f"  achieved internal response: {report['mechanics']['achieved_response']}")
    print(f"  internal/torch-fem relative response error: {optimized_comparison['response_relative_error']:.3e}")
    print(f"  derivative max relative error: {derivative_check['max_relative_error']:.3e}")
    print(f"  artifacts: {output}")


if __name__ == "__main__":
    main()
