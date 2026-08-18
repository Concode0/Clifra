# Why bivector coordinate fields work

An `InvertibleBivectorField` extends a Clifford-generated action from one
bivector to a field of local actions over a sample domain. At each composition
stage, a generator sampler evaluates a bivector for each persistent sample
identity, and `path_steps` specifies how many such stages are composed in order.

The construction works because a bivector induces a linear generator on the
grade-1 subspace. Exponentiating that generator gives an invertible local
action, while sampling and ordered composition turn those local actions into a
differentiable field.

## From a bivector to a local action

Let $B$ be a bivector in $Cl(p,q,r)$ and hold it fixed for one composition
stage. Define the rotor

\[
R(t)=\exp(-tB/2).
\]

Because bivector reversal gives $\widetilde{B}=-B$,

\[
\widetilde{R(t)}=\exp(tB/2).
\]

The sandwich action on a grade-1 value is therefore

\[
x(t)=R(t)x(0)\widetilde{R(t)}
    =\exp(-tB/2)x(0)\exp(tB/2).
\]

Differentiating the two exponential factors gives

\[
\frac{d x(t)}{dt}
=-\frac12\left(Bx(t)-x(t)B\right).
\]

The commutator with a bivector maps grade 1 back to grade 1. Consequently it
induces an ordinary linear operator $G(B)$ on the grade-1 coordinate lanes:

\[
G(B)x=-\frac12(Bx-xB),
\qquad
\frac{d x(t)}{dt}=G(B)x(t).
\]

Since $G(B)$ is constant during the stage, the solution at $t=1$ is

\[
x(1)=\exp(G(B))x(0).
\]

This connects learned bivector coefficients to invertible linear actions.
`InvertibleBivectorField` obtains the default grade-1 action from
`algebra.plan_versor_action(grade=2, ...)`; the planned executor constructs the
induced generator matrix and applies its matrix exponential. The equivalent
even-grade rotors can be inspected with `rotors_for_input()` or, for
shape-only samplers, `rotor_path()`.

## From a local action to an indexed field

Let $E$ embed coordinates into the selected grade-1 chart and let $D$ extract
ordinary coordinates afterward. For persistent identity $\xi_i$, a generator
sampler evaluates one bivector per composition stage:

\[
B_s^\Theta(\xi_i), \qquad s=0,\ldots,S-1.
\]

If the stored parameters use a lower-dimensional generator representation,
`GeneratorSubspace` supplies a fixed linear map

\[
B_s^\Theta(\xi_i)=M z_s^\Theta(\xi_i)
\]

from sampled latent coordinates $z$ to compact bivector coefficients. The
resulting ordered field map is

\[
\phi_\Theta(x_i,\xi_i)
=D\!\left[
\exp\!\left(G(B_{S-1}^\Theta(\xi_i))\right)
\cdots
\exp\!\left(G(B_0^\Theta(\xi_i))\right)
E(x_i)
\right].
\]

`CoordinateFieldInput` makes the two arguments explicit:

- `coordinates` contains the values $x_i$ transformed by the field;
- `sample_coordinates` contains the persistent identities $\xi_i$ used by the
  generator sampler.

The tensors share a prefix shape but may have different final dimensions. A
three-dimensional value can, for example, carry a one-dimensional identity.
`domain_shape` records the structured suffix of that prefix; it is topology
metadata, not another coordinate value.

If `sample_coordinates` is omitted, sampling falls back to `coordinates`.
Otherwise the identities remain fixed as transformed values change.
`with_coordinates()` replaces values while retaining identity and topology,
and `retain_sample_identity()` makes fallback identities explicit.

The sampler controls how $z_s^\Theta(\xi)$ or $B_s^\Theta(\xi)$ varies over the
domain. `BroadcastGeneratorSampler` shares one generator per stage,
`RegularGridGeneratorSampler` interpolates a structured control lattice, and
`RBFGeneratorSampler` evaluates controls at arbitrary
`sample_coordinates`. Sampling changes generator coefficients; it does not
interpolate or replace the coordinate values being transformed.

## Why composition order matters

`path_steps` counts ordered composition stages. It does not denote physical
time and is separate from batch and sampling-domain axes. In general,

\[
\exp(G(B_2))\exp(G(B_1))
\ne
\exp(G(B_1))\exp(G(B_2))
\]

when the induced generators do not commute. Reordering stages or replacing
them with a single exponential of the summed bivectors therefore changes the
field parameterization.

The implementation preserves this order from stage $0$ through $S-1$.
`rollout()` exposes intermediate forward and reverse composition states as a
`TransformationRollout`; its `forward_coordinates` and `inverse_coordinates`
are diagnostics of the declared composition, not a different field map.

## Differentiating the field

Every operation in the construction is differentiable: generator sampling,
the optional subspace map, the induced matrix exponential, ordered action
composition, and chart extraction. A generic objective can therefore be
written as

\[
\min_\Theta\;
\mathcal{L}\!\left(\mathsf{State}_\Theta(X,\xi),Y\right)
+\sum_k\lambda_k
\mathcal{P}_k\!\left(\mathsf{State}_\Theta(X,\xi),\Theta\right).
\]

`InvertibleBivectorField.state()` returns a `TransformationState`. It exposes
`input_coordinates`, `transformed_coordinates`, sampled `generator_weights`,
optional sampled `latent_coordinates`, topology metadata, and the
identity-preserving `field_input`. The loss terms determine what is fitted;
they are not part of the transformation-field representation itself.

For inspection, `latent_for_input()` evaluates sampled latent coordinates and
`weights_for_input()` evaluates the resulting bivector coefficients after the
`GeneratorSubspace` map.

## Indexed invertibility and global injectivity

For a fixed identity $\xi$, each local exponential is invertible:

\[
\exp(G(B_s(\xi)))^{-1}=\exp(-G(B_s(\xi))).
\]

The inverse of the composition applies the same sampled generators in reverse
stage order with opposite signs. Operationally, it starts with
$-B_{S-1}(\xi)$ and ends with $-B_0(\xi)$.

The same persistent identity must be supplied in both directions.
`TransformationState.inverse_input()` pairs the transformed values with the
original `sample_coordinates`:

```python
state = field.state(field_input)
reconstructed = field.inverse(state.inverse_input())
```

The equivalent explicit form is
`field.inverse(field_input.with_coordinates(state.transformed_coordinates))`.
Passing only the transformed values to a coordinate-driven sampler can sample
different generators and does not satisfy this indexed inverse contract.
`inverse_state()` returns the corresponding `TransformationState` for the
reverse evaluation.

This establishes local, identity-indexed invertibility. It does not by itself
establish global injectivity of the assembled field. Different identities may
select different invertible local actions whose outputs coincide, and a field
over a structured domain may fold or self-intersect even though every local
matrix exponential is invertible. Global injectivity requires a separate
property or validation of the assembled map. A custom injected action also
retains the inverse contract only when negating its generator implements the
local inverse.
