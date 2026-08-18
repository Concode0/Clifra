# Planning Policies

clifra chooses execution routes during static planning. The choice depends on
the algebra specification, declared layouts, dtype, device, numerical options,
and the injected planning policy. Tensor values and runtime batch dimensions do
not participate in route selection.

Each operation:

1. validates its static contracts;
2. enumerates the routes it implements;
3. marks routes that cannot satisfy the request as unavailable;
4. publishes facts for each available route;
5. asks the policy to score or reject those routes;
6. constructs and caches the selected plan.

Capability checks remain part of the operation. A policy can prefer an
implemented route, but it cannot make an unsupported route available.

## Formula policies

`FormulaPolicy` defines an acceptance region and score for each `(family,
route)` pair. A finite score accepts a candidate, and the lowest score wins.
Equal scores use the operation's candidate order.

The following policy replaces the default product rules and leaves the other
operation families unchanged:

```python
import torch

from clifra.core import (
    DEFAULT_PLANNING_POLICY,
    BoundaryRegion,
    FormulaConstraint,
    FormulaPolicy,
    Polynomial,
    ResourceLimits,
    RouteRule,
    make_algebra,
)

non_product_rules = tuple(
    rule for rule in DEFAULT_PLANNING_POLICY.rules
    if rule.family != "product"
)

full_table_region = BoundaryRegion(
    constraints=(
        FormulaConstraint(
            Polynomial.feature("layout.output_lanes", constant=-8_192.0),
            reason="full_table_lane_boundary",
        ),
    ),
    name="bounded_full_table",
)

policy = FormulaPolicy(
    rules=(
        *non_product_rules,
        RouteRule(
            "product",
            "full_table",
            regions=(full_table_region,),
            score=Polynomial.feature("forward_work"),
        ),
        RouteRule(
            "product",
            "sparse",
            score=Polynomial.feature("forward_work", coefficient=1.1),
        ),
    )
)

algebra = make_algebra(
    10,
    device="cpu",
    dtype=torch.float32,
    planning_policy=policy,
    resource_limits=ResourceLimits(
        max_lanes=8_192,
        max_pairs=12_000_000,
    ),
)
```

A `FormulaConstraint` represents the inclusive inequality

\[
f(x_1, \ldots, x_n) \leq 0.
\]

Constraints within one `BoundaryRegion` are intersected. Multiple regions on a
route form a union. Polynomial terms may use non-negative integer powers, so a
boundary or score may be nonlinear.

`FormulaPolicy` rejects a candidate when it has no matching rule. If no
available candidate is accepted, planning raises `PolicyCoverageError`.

## Resource limits

`ResourceLimits` is separate from route policy. Lane and interaction limits are
hard allocation boundaries rather than preferences, so a low route score cannot
override them.

```python
limits = ResourceLimits(
    warn_lanes=4_096,
    max_lanes=8_192,
    warn_pairs=2_000_000,
    max_pairs=12_000_000,
)
```

![Canonical, vector, and vector-plus-bivector lane growth across algebra dimensions](../assets/explanations/planning-lane-growth.png)

Lane width and interaction count protect different resources. A compact layout
can still produce many candidate product interactions, while a broad output can
be expensive to store even when relatively few pairs contribute. Canonical
storage grows as \(2^n\), so removing a lane cap transfers responsibility for
that allocation to the caller; it does not make the representation scale
linearly.

The preflight pair count is conservative. It may reject a request based on declared input widths before constructing every basis interaction. Grade, projection, and metric-zero filtering can make the realized plan smaller.

The same `planning_policy` and `resource_limits` arguments are available through
`AlgebraConfig`, `make_algebra`, and `make_algebra_from_config`.

## Route facts

Every candidate carries an immutable `PlanFacts` value. The common facts have
the same meaning across operation families.

| Fact | Meaning |
| --- | --- |
| `forward_work` | Relative forward work |
| `backward_work` | Relative backward work |
| `peak_bytes` | Estimated peak temporary storage |
| `compile_work` | Relative graph and construction complexity |
| `quality_exact` | The route provides the operation's exact semantic guarantee |
| `quality_truncated` | The route may apply a declared static truncation |
| `quality_value_dependent` | Approximation quality can depend on tensor values |

Sequential forward, backward, and compilation work are normally additive. Peak
storage depends on temporary lifetimes and is not generally additive. Exactness
requires every relevant child route to be exact; truncation propagates when any
child route truncates.

Error estimates belong in qualified extensions until multiple operation
families share the same bound and composition rule. Comparison signals such as
`exp.rank_deficit` are not presented as output-error bounds.

### Extension attributes

Operation-specific facts use qualified extension names. They remain numeric so
the same formula interface can read common and operation-specific facts.

Examples include:

- `algebra.n` and `algebra.r`;
- `backend.cpu` and `backend.mps`;
- `dtype.bytes`;
- `layout.output_lanes`;
- `exp.retained_rank` and `exp.rank_deficit`;
- application facts such as `vendor.machine_score`.

Extension names must be dot-qualified. The defining operation or application
also defines their units and meaning. A policy that depends on a specialized
extension is portable only to candidates that publish that extension.

A fact belongs in the common set only when several operation families share its
meaning, units, and composition rule. Other facts should remain extensions.
This keeps the common policy vocabulary stable without closing the set of facts
available to new operators.

## Custom policies

`PlanningPolicy` is a protocol. A custom policy does not need to inherit from a
clifra base class.

```python
from dataclasses import dataclass

from clifra.core import PolicyEvaluation


@dataclass(frozen=True)
class MachinePolicy:
    work_scale: float

    def evaluate(self, candidate):
        facts = candidate.facts
        score = (
            facts["forward_work"] * self.work_scale
            + facts["compile_work"]
            + facts["peak_bytes"] / 4096.0
        )
        return PolicyEvaluation(score, "eligible")
```

The policy is fixed for the lifetime of its `AlgebraContext`. Construct another
algebra to use different policy settings. Evaluation must be deterministic for
a fixed candidate; runtime tensor values, randomness, clocks, and mutable
global state must not affect it.

Returning `PolicyEvaluation(None, reason)` rejects a candidate. Returning a
finite score accepts it.

## Adding a route or operation

A policy selects among candidates; it does not provide an implementation. A new
route must supply its executor or plan, capability checks, static facts, and
cache inputs before a policy can select it.

### PyTorch implementations

An operation implemented directly with PyTorch tensor primitives must publish
its own facts. clifra cannot infer work, storage, or approximation guarantees
from arbitrary tensor code.

For each route:

- construct `PlanFacts` from static contracts and options;
- pass route-specific numeric facts through the `extensions` mapping;
- state its exactness, truncation, and value dependence;
- publish a total error bound only when its upper-bound interpretation is valid;
- otherwise publish a clearly named operation-specific comparison proxy;
- include every static input that can change the route in the plan cache key;
- keep tensor values out of planning.

The executor must still satisfy the declared layouts, dtype and device behavior,
autograd requirements, and compilation contract.

### Operations composed from clifra APIs

Existing clifra calls plan and cache their own routes automatically. A function
that calls a product followed by a bivector exponential therefore receives the
selected child implementations without defining a new policy family.

clifra does not automatically combine those child facts into a parent cost. A
parent planner must compose them explicitly when it:

- compares more than one implementation of the composition;
- applies an aggregate resource budget;
- exposes an inspectable parent plan.

Action planning follows this pattern for its exponential and product children.
Optional analysis operations use `ResourceLimits` directly before executing.
Parent peak storage and error bounds must follow the actual lifetime and
numerical behavior of the composition rather than a generic sum.

## Caching

Product and bivector-exponential executors and policy-selected action plans are
cached per algebra using their static request data. The injected policy remains
fixed for that algebra's lifetime. Device or dtype moves clear policy-dependent
caches before replanning.
