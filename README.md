# clifra

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/docs-MkDocs-brightgreen)](https://concode0.github.io/clifra/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18939518.svg)](https://doi.org/10.5281/zenodo.18939518)

Layout-first Clifford algebra tools for PyTorch.

A clifra algebra host owns its layouts, policies, and operation plans. Full-lane
tensors and compact grade layouts share that algebra, while planning builds
static executors for products, metrics, exponentials, and actions. Layers and
other library components reuse the same operations.

## Install

```bash
uv sync
```

Development:

```bash
uv sync --group dev
```

Docs:

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

## Minimal Use

```python
import torch

from clifra import make_algebra

algebra = make_algebra(3, 0, device="cpu")
vectors = algebra.layout((1,))
products = algebra.plan_product(
    op="gp",
    left_layout=vectors,
    right_layout=vectors,
    output_layout=algebra.layout((0, 2)),
)

left = torch.randn(8, vectors.dim)
right = torch.randn(8, vectors.dim)
out = products(left, right)
```

## Checks

```bash
uv run --group dev pytest tests/ -n12 -q --tb=short
uv run --group dev pytest tests/ --hypothesis-profile=full -n12 -q --tb=short
uv run --group dev ruff check .
uv run --group docs mkdocs build
```

See the [documentation](https://concode0.github.io/clifra/) for tutorials,
explanations, benchmarks, and the generated API reference.

## Transformation Fields

`research/transformation_fields` explores differentiable fields of
Clifford-generated geometric actions. A field samples bivector generators over
a domain and compiles them into local transformations, while transformed values
and persistent sampling coordinates remain distinct.

This separation keeps the construction general: samplers determine how
generators vary over spatial, material, temporal, or other parameter domains;
ordered action steps determine how local Clifford transformations compose; and
user-defined differentiable objectives determine what the field learns.

[Bivector field basics](research/transformation_fields/examples/bivector_field_basics.py)
is the compact introduction. In `Cl(2,0)`, it learns an RBF-sampled,
coordinate-dependent rotation field from an analytic target and exposes the
corresponding bivector generator field. It also verifies indexed inversion and
sample-permutation equivariance.

```bash
uv run research/transformation_fields/examples/bivector_field_basics.py
```

The example is intentionally small. It introduces the transformation-field
primitive itself rather than committing it to a particular scientific application: 
domain labels select local generators, Clifford algebra determines
the resulting geometric action, and optimization identifies the field from a chosen objective.

See [Why bivector coordinate fields work](https://concode0.github.io/clifra/explanations/transformation-fields/)
for the derivation, field semantics, ordered action model, and inversion contract.

## Contribution

Found a problem or want to propose a change? Please open an Issue first,
especially before a PR, so the scope is clear.

For direct contact, email: nemonanconcode@gmail.com

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

```bibtex
@software{kim2026clifra,
  author  = {Kim, Eunkyum},
  title   = {clifra: Clifford Algebra Layers for PyTorch},
  url     = {https://github.com/Concode0/clifra},
  version = {1.3.1},
  year    = {2026},
  doi     = {10.5281/zenodo.18939518},
  license = {Apache-2.0}
}
```

## References

These works provide conceptual background for `RotorGadget` and the tag-aware
optimizers. Clifra's behavior is defined by its public API, source, and tests.

### RotorGadget Background

- Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers from Irreducibles." *arXiv:2507.11688*.

### Optimization Background

The tag-aware optimizers in `clifra/optimizers/` dispatch `spin`, `sphere`, and
`euclidean` post-update handling.

- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
- Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press.
