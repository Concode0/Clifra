# Bivector Exponential Methods

The exponential of a bivector $B$ is the even multivector

\[
\exp(B) = \sum_{k=0}^{\infty}\frac{B^k}{k!}.
\]

Its occupied grades and the cost of evaluating it depend on the dimension, signature, and algebraic structure of $B$. Clifra selects a static executor from the algebra specification, layouts, dtype, device, and exponential policy.

## Finite closures

### Simple closure

When $B^2=s$ is scalar, every power of $B$ lies in the span of $1$ and $B$:

\[
\exp(B)=C(s)+S(s)B,
\]

where

\[
C(s)=
\begin{cases}
\cos(\sqrt{-s}) & s<0,\\
1 & s=0,\\
\cosh(\sqrt{s}) & s>0,
\end{cases}
\qquad
S(s)=
\begin{cases}
\dfrac{\sin(\sqrt{-s})}{\sqrt{-s}} & s<0,\\
1 & s=0,\\
\dfrac{\sinh(\sqrt{s})}{\sqrt{s}} & s>0.
\end{cases}
\]

This covers all bivectors for $n\leq3$ and is implemented by
`closed_simple`.

### Biquadratic closure

For $4\leq n\leq5$, write

\[
B^2=s+K,
\]

where $s$ is scalar and $K$ has grade 4. In these dimensions $K^2$ is scalar,
so the exponential closes over

\[
1,\quad B,\quad K,\quad BK.
\]

`closed_biquadratic` evaluates the four scalar coefficients from the two roots
of the resulting quadratic relation. Real and complex root pairs use the same
finite closure with different scalar coefficient formulas.

## General representations

### Left-multiplication matrix

`left_matrix_exp` represents left multiplication by $B$ on the even
subalgebra. If $L_B$ is that operator and $e_0$ represents the scalar identity,
then

\[
\exp(B)=\exp(L_B)e_0.
\]

The executor applies `torch.matrix_exp` and maps the resulting column into the
requested output layout. The operator has $2^{n-1}$ lanes for a full even
algebra. `cpu_matrix_exp` uses the same construction on CPU for matrix cases
that are unavailable on MPS.

### Spectral-local representation

For an eligible definite signature, clifra maps $B$ to a skew generator $G$ on
the nondegenerate vector space. The symmetric problem based on $-G^2$ identifies
invariant plane pairs. The executor then:

1. selects up to four plane pairs;
2. reconstructs a simple bivector on each selected plane;
3. evaluates each simple exponential;
4. multiplies the commuting factors in a bounded local even algebra;
5. lifts the requested grades into the ambient output layout.

Supported degenerate signatures add a local null ideal to this construction.
The null-ideal dimension is capped at four, and its mixed and nilpotent terms
are evaluated inside the local algebra.

The cost of `spectral_local` depends on the retained local dimension rather
than the full ambient even algebra. It is exact when the retained planes and
null block contain the complete active structure. If additional planes are
present, it evaluates the exponential of the retained local component.

## Executor selection

| Family | Normal selection | Result |
| --- | --- | --- |
| `closed_simple` | $n\leq3$ | Exact scalar-bivector closure. |
| `closed_biquadratic` | $4\leq n\leq5$ | Exact scalar, bivector, and grade-4 closure. |
| `spectral_local` | Eligible higher-dimensional cases | Exact for a complete retained spectrum; otherwise truncated. |
| `left_matrix_exp` | Matrix fallback | Exponential of the represented full even operator. |
| `cpu_matrix_exp` | MPS matrix fallback | The matrix construction executed on CPU. |

The spectral-local eligibility rules are:

- the signature is definite on its nondegenerate part;
- the dtype is neither `float16` nor `bfloat16`;
- the nondegenerate space contains at least one plane;
- the null dimension does not exceed four;
- degenerate and truncated-degenerate handling are enabled when required;
- on devices other than MPS, the dimension reaches the policy transition,
  which defaults to 10.

Dimensions at or below five always use a finite closure. An ineligible
higher-dimensional case uses a matrix executor. A mixed positive/negative
signature is currently a matrix case.

The transition dimension and spectral limits are policy inputs. They determine
the executor family during planning; the executor does not change families from
runtime tensor values.

## Coefficient evaluation near repeated roots

The closed and spectral-local formulas contain removable limits and divided
differences. Direct evaluation near a repeated root can subtract nearly equal
values before dividing by a small invariant. Clifra uses a fixed Taylor
polynomial in that region.

Let $u=\operatorname{finfo}(\text{dtype}).\mathrm{eps}$. If the first omitted
Taylor term is approximately $x^m/D$ and direct roundoff is amplified as
$u/x^k$, the representations have comparable error near

\[
x_{\mathrm{cut}}=(uD)^{1/(m+k)}.
\]

The executor derives each cutoff from its dtype and coefficient formula.
Float32 therefore uses a wider Taylor interval than float64. The polynomial
degree and cutoff are fixed for a planned executor, so this treatment does not
introduce data-dependent iteration.

These cutoffs govern scalar coefficient evaluation only. Product roundoff,
eigenspace conditioning, backend kernels, and spectral truncation remain
separate sources of error in the complete result.

## Truncation and diagnostics

Let $|\theta_1|\geq\cdots\geq|\theta_M|$ be the plane-angle magnitudes and let
$k$ be the retained plane count. Clifra reports two complementary summaries:

\[
\operatorname{GVC}
=\frac{\sum_{i=1}^{k}\theta_i^2}
       {\sum_{i=1}^{M}\theta_i^2},
\qquad
T=\sum_{i=k+1}^{M}|\theta_i|.
\]

Geometric variance captured (GVC) measures relative spectral concentration.
The tail-angle sum $T$ measures the absolute size of the omitted plane angles.
Neither quantity alone is an error bound for an arbitrary downstream
calculation.

`spectral_exp_angle_diagnostics` computes these values from a supplied angle
spectrum. `spectral_exp_uniform_tail_stress` reports the static case in which a
fixed norm is distributed uniformly across all available planes. The former is
suited to measured generators; the latter is a capacity stress case.

Diagnostics should retain relevant batch, layer, or channel axes until the
desired aggregation is chosen. The diagnostic API detaches its input and is
intended for analysis rather than as a differentiable training objective.

## Backward behavior

The closed and matrix executors differentiate their tensor programs directly.
The spectral-local executor uses a filtered symmetric eigendecomposition. Its
backward replaces unstable inverse gaps near repeated eigenvalues with a finite
convention for the locally non-unique eigenspace.

When planes are truncated, backward differentiates the retained computation.
It does not add derivatives for omitted spectral components. Forward and
gradient comparisons with a matrix-exponential reference should therefore use
the same retained-spectrum assumptions.

## Operating principles

Clifra applies the following rules to bivector exponentiation:

1. **Plan statically.** Executor selection belongs to planning and is determined
   from structural inputs and policy, not inferred mathematical intent.
2. **Use finite closure when available.** Low-dimensional exact formulas avoid
   constructing a larger operator.
3. **Bound local work explicitly.** Spectral-local plane and null dimensions are
   planned limits, and truncation is part of that executor's result semantics.
4. **Separate numerical conditioning from method selection.** Dtype-derived
   coefficient cutoffs stabilize a selected formula without changing executor
   families.
5. **Differentiate the executed map.** Each backward path corresponds to the
   exact or truncated forward computation that produced the result.
6. **Measure approximation at the workload boundary.** When spectral truncation
   is enabled, inspect both retained spectral concentration and absolute tail
   magnitude on representative inputs.
