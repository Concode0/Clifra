# Repository Guidelines

## Project Structure & Module Organization

`clifra/core/` is the source of truth for algebra layouts, planning, execution,
runtime tensor contracts, and foundation types. Keep these contracts explicit
and stable.

`clifra/layers/`, `clifra/functional/`, `clifra/criterion/`, and
`clifra/optimizers/` build on core contracts. Reuse core algebra semantics and
planning machinery rather than reimplementing them in higher-level modules.

Documentation lives in `docs/`; runnable research systems and demonstrations
live in `research/`; benchmark harnesses and generated benchmark artifacts live
under `benchmarks/`.

## Build, Test, and Development Commands

Use `uv` for project tooling. Run Python and project commands through `uv run`;
use `uv sync` for environment synchronization.

```bash
uv sync --group dev
uv run --group dev pytest tests/ -n12 -q --tb=short
uv run --group dev pytest tests/planning/ -n12 -q --tb=short
uv run --group dev ruff check .
uv run --group docs mkdocs build
```

Prefer focused tests while iterating, then run the full suite before handing off
a cross-cutting change. Property tests use Hypothesis; the full profile is:

```bash
uv run --group dev pytest tests/ --hypothesis-profile=full -n12 -q --tb=short
```

Use fewer pytest workers when the execution environment is resource-constrained.

## Coding Style & Naming Conventions

Run `uv run --group dev ruff check .` on touched work. Ruff is the source of
truth for Python linting conventions.

Respect module boundaries: add algebra semantics, layouts, plans, executors, and
shared tensor-contract behavior to `clifra/core/`. Do not duplicate core
semantics in higher-level modules.

Preserve compact/canonical storage and tensor contracts rather than
introducing ad-hoc shape conventions.