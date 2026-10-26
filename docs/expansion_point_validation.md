# Expansion-point implementation and verification

The implemented shift is a **real nonnegative Laplace expansion point** `s0`.
It is not a substitution of `j*omega0` into a real factorization. The field
operator and mass form are

\[
a_{s_0}(u,v)=\int\sigma^{-1}\operatorname{curl}u\cdot\operatorname{curl}v
  +s_0\int\mu u\cdot v,\qquad m(u,v)=\int\mu u\cdot v.
\]

The same recurrence is used at zero and positive shifts:

\[
a_{s_0}(T_{2k+2},N)
=-m(H_{2k+1},N)/L_{2k+1}+a_{s_0}(T_{2k},N).
\]

The previous-stage term is already in the right-hand side. Do not add the
previous field again after the solve. There is an important boundary-lifting
detail: for current-drive Dirichlet data, the initial field has nonzero trace
but subsequent fields have homogeneous trace. At the first update the initial
field satisfies `a(T0,N)=0` for free test functions. Literally adding T0 to a
homogeneous correction would change the boundary data; the weak-form recurrence
above avoids that mistake.

Dirichlet boundary data are set with NGSolve's dual HCurl interpolation
(`GridFunction.Set(..., dual=True)`) in both the positive-shift CLN and the
direct reference. At exactly zero shift, `nograds=True` requires the reduced
space's default projection: dual interpolation gave nonfinite values there
with the tested NGSolve 6.2.2607.
The default projection can bias the imposed current in the full order-2 space;
increasing volume quadrature alone does not fix that boundary-lifting error.
The low-frequency Bessel regression requires relative error below `1e-4`.

## What changes at zero?

For positive `s0`, the mass term makes the operator coercive and the full HCurl
space is used. At zero, higher gradient modes are removed with `nograds=True`,
and an Omega correction enforces mass orthogonality to the remaining gradient
kernel. This changes the representation/gauge, not the recurrence.

- Voltage-drive Neumann: Omega is free on the boundary. One scalar value is
  pinned to remove the constant; no mass perturbation is added to the gauge.
- Current-drive Dirichlet: Omega vanishes on the prescribed lateral boundary,
  preserving its tangential trace.
- The corrected field remains `T+grad(Omega)` as a coefficient expression.
  Reinterpolating it into another HCurl space can reintroduce a gradient error.
- Solve, gauge, recurrence normalization and the direct reference use consistent
  explicit quadrature on curved elements. Default `integration_order=2*order+6`.
  Consistency does not replace quadrature/mesh convergence studies.

The supported benchmark is a connected straight circular conductor with the
documented current/voltage boundary data. General topology, ports and material
models require their own nullspace and boundary analysis.

## Convergence is checked, not inferred

Every primal solve checks `||P(b-A*x)||/||P*b||` on free DOFs after the final
update. `solve_diagnostics` records this value and the actual backend.
If a zero-shift ICCG result fails the requested original-system tolerance,
the implementation raises instead of returning CLN elements. It never
switches automatically to direct refinement. Select `solver="direct"`
explicitly to use regularized direct refinement, which still targets the
original unregularized operator and must pass the same true-residual check.
The inner ICCG tolerance is `min(tol*1e-2, 1e-12)`, preserving the historical
`1e-12` target at the default outer `tol=1e-10`. Diagnostics record both targets;
this requested margin does not replace measured convergence.

The previous implementation could return a resistance about 25 times the DC
benchmark under deliberately inadequate refinement (`eps_rel=1e3`, `maxiter=1`).
That case now fails explicitly. The gauge correction and consistent quadrature
also remove sources of incompatible residuals; changing the tolerance alone
would not correct them.

## Independent checks

`tests/test_shift_review.py` verifies:

- Neumann and Dirichlet shifted results against independent full FE solves,
  including the exact value at `s0` and the first moment `Z'(s0)=L1`.
- Complex responses on `s=j*2*pi*f`, separately from positive-real-axis checks.
- Stiffness and magnetic Gram off-diagonals at both `s0=0` and positive `s0`.
  Magnetic orthogonality is tested with **m(H_i,H_j)**, never `s0*m(H_i,H_j)`;
  multiplying by zero would make that test vacuous.
- Explicit rejection of unconverged solves and invalid parameters.

The cylinder regressions check zero-shift coefficients against analytic values
and require actual solve convergence. Export tests additionally compare the
same CLN with its state-space realization and LTspice Laplace circuits.

These checks establish the implementation on the tested cases. They do not
prove arbitrary-order accuracy or passivity across all frequencies. Truncation
error can be much larger away from the expansion point, and mesh error is
separate from reduction error. Inspect the exported poles and compare the
intended frequency band with `reference_impedance` before using a reduced model.

Element values can be much more sensitive than terminal impedance. At a real
shift of `2*pi*1e5`, radius 10 mm, conductivity `1e6` and permeability `mu0`,
a mesh of about 1 mm gave an R8 error exceeding 100% with order 2, compared
with about 7% with order 3. This is not evidence of an erroneous recurrence:
higher continued-fraction coefficients amplify discretization errors. Resolve
the field variation and check mesh/order convergence before interpreting later
elements. `tests/test_shift_elements.py` checks indices 0 through 4 at order 3
against independent coefficients of the analytic Bessel Taylor series.

See [LTspice and Simulink export](ltspice_simulink.md) for the exact handling of
the shift, port direction and proper/improper transfer functions.

## Ownership of retained fields

`initial()` and `step()` detach H and J from the mutable solver work vectors
using coefficient copies into new GridFunctions. The Omega correction remains
an exact coefficient expression over its privately retained operands; it is not
reinterpolated into HCurl. Reusing or clearing a solver work buffer therefore
cannot change previously returned fields. Returned GridFunctions are still
user-editable; this is ownership isolation, not general immutability of NGSolve
objects. Treat returned fields and retained modes as read-only once used in a
model. The ordinary solver already allocates a new work vector per stage; this
protection also covers future scratch-buffer reuse and interactive callers.

## Explicit positive-shift method selection

For `s0 > 0`, callers must pass `solver="direct"`. The default or an explicit
`solver="sparsesolv"` raises before mesh access and assembly. Direct means
Cholesky of the positive-shift operator itself; zero-shift direct refinement
and ICCG retain their existing definitions. Diagnostics record `requested_solver`
separately from the actual `backend` (`sparsecholesky` for positive shift).
No failure selects another method, and the original-system residual criterion
is unchanged.
