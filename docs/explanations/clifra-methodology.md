# Geometric Parameterization

clifra supports both individual Clifford operations and differentiable geometric models in PyTorch. A recurring pattern in clifra is to represent a geometric object's generating coordinates directly and construct its action with the algebra.

Under this pattern, a model may learn a bivector that generates a rotor, a
vector that determines a reflection, or coefficients in a declared mixture of
grades. The learned quantity can therefore be the geometric generator rather
than an unconstrained tensor representing the resulting transformation.

`VersorLayer` and `MultiVersorLayer` currently support `grade=1` reflection
parameters and `grade=2` rotor parameters. Other grade layouts remain available
to planned products and project-specific layers; they are not accepted as
versor-layer parameter grades.

## From coordinates to an action

A typical construction has four stages:

1. **Declare an algebra.** The signature states how basis directions square and
   therefore which geometry the products express.
2. **Declare a layout.** The selected grades define the coordinate space of the
   object being represented or learned.
3. **Generate an action.** Clifford products, exponentials, reverses, and
   sandwich actions turn those coordinates into a transformation.
4. **Learn through the action.** PyTorch autograd differentiates the loss through
   the planned tensor program and back to the coordinates.

For a bivector $B$, a rotor can be written

\[
R = \exp(-B/2), \qquad x' = R x \widetilde{R}.
\]

The parameter is $B$, rather than an arbitrary dense matrix or a stored rotor
whose constraints must be repaired after every update. The exponential and
sandwich product construct the action at each forward pass. The model therefore
learns the plane generator of the transformation in a fixed grade-2 coordinate
space.

In clifra, the signature, selected grades, and action are explicit modeling
choices rather than properties hidden inside a full multivector representation.

## Layout is part of the hypothesis

A layout does more than reduce storage. Selecting grade 1 places the
coefficients in the vector subspace of the chosen algebra. Selecting grade 2
places them in the bivector subspace of oriented plane elements. Selecting
several grades permits a mixed-grade multivector object. The meaning of those
coordinates then depends on the operation or layer applied to that layout.

For learned models, the layout therefore becomes part of the model hypothesis.
Narrowing it can make the representation interpretable and computationally
tractable, but it also excludes components. clifra keeps this choice explicit.
