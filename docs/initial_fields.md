# Initial finite-element fields and CLN states

Initial conditions require retained field modes, not only R/L/C values. The transfer function alone does not determine the field corresponding to a state.

## Diffusion and heat

For M udot + K u = f F, retained even modes E reconstruct u=E x. Solve

    (E.T M E) x0 = E.T M u0.

This minimizes capacity-weighted error. The measured Gram matrix is used and checked against the circuit metric. Full, retained and residual quadratic storage and the Pythagorean residual are reported. These diagnose t=0; they are not future output-error bounds. For heat quadratic storage has units J K. Actual excess heat is a separate linear functional and need not be preserved.

```python
from cln.energy import physical_cln
model = physical_cln(K, M, load, nstages=3)
projection = model.project_initial(u0)  # original free-DOF ordering
u_projected = model.reconstruct(projection.x0)
lti = model.to_lti(initial_field=u0)
lti.write('output', 'initialized_heat')
```

Eliminate fixed DOFs and subtract the reference field first. K and M must be real SPD. The map supports the full capacitor-terminated physical RC model, not the legacy resistance termination.

For native NGSolve use `model = diffusion.energy_model(nstages=3)`, then pass a real GridFunction in `diffusion.fes` satisfying homogeneous essential boundary conditions. Integrals include extrusion depth once; no dense full-order matrix is formed. Temperature is excess above reference.

The MCP tool `cln_diffusion_physical_model` accepts optional `initial_field` in original free-DOF ordering and returns projection diagnostics, initialized A/B/C/D/x0, and MATLAB factory source. Matrix limit: 256 DOFs; use the native API for larger systems.

## Magnetic T–Omega CLN

```python
field_model = t_omega.field_model(nstages=3)
projection = field_model.project_initial(H0)
H_projected = field_model.reconstruct(projection.x0)
lti = field_model.to_lti(initial_field=H0)
lti.write('output', 'initialized_magnetic')
```

This builds **inductance-terminated admittance**: total port voltage input, current output, shunt-inductor current states. H0 is physical H in A/m on the same conductor mesh, not B, T alone or an ungauged vector potential. Supply physically admissible fields; the projection does not perform boundary lifting or gauge correction of input data.

Retained odd modes Q=[H1,H3,...] reconstruct H=Q x. Projection uses the full-volume mu-weighted Gram matrix. The certificate checks

    Q.T M_mu Q = diag(L_i * length)
    integral curl(Q_i).curl(Q_j)/sigma = -diag(L_i * length) A.

The second identity is physical Joule dissipation and includes cancellation of the expansion shift. Magnetic storage is in Joules. Zero and positive real expansion points are supported when the measured identities pass. Poor orthogonality or nonpositive dissipation causes rejection, not silent stabilization. Neumann and Dirichlet-generated bases are supported under the existing conductor/port assumptions.

R-terminated RL eliminates a terminal storage mode. Its Schur-complement dynamics do not have the same field metric as these odd modes. This API builds the L termination; it does not map an arbitrary existing R-terminated export. L-terminated impedance is improper and is not an ordinary LTI System model.

## MATLAB / Simulink and LTspice

The generated `[sys, data]` factory carries projected `data.x0`. `sys` alone does not carry initial conditions. Assign sys and x0 to the LTI System block, or A/B/C/D and x0 to State-Space. Bind variables explicitly to the model workspace, e.g. `SimulationInput.setVariable(..., 'Workspace', modelName)`.

Nonzero-state `.write()` emits only .m and .json. Laplace elements describe zero-state transfer and cannot encode this field; requesting a Laplace subcircuit with nonzero x0 raises an error. Existing Laplace files with the same name also cause rejection even with overwrite=True. `CLNLTI.with_initial_state(x0)` accepts already-known state coordinates without field projection.

A load-generated reduced space can miss initial modes not excited by the port. Lost-storage diagnostics expose this. Increasing stages may still not recover an unreachable subspace; adding initial-field vectors to the reduction basis is a separate extension.

### Physical magnetic port and admissibility

A finite-element boundary interpolation may give all retained modes a common
axial current alpha slightly different from 1 A. The model measures
`I_i = integral J_i,z / length`, checks that this normalization is common,
and uses an ideal-transformer scale alpha at the physical port: B is multiplied
by alpha, C by alpha, and admittance by alpha squared. Field state coordinates
and stored energy do not change. The measured normalization and residual are
included in the certificate. Non-common port errors cause rejection.

For Dirichlet-generated modes, the lateral trace is a scalar multiple of the
imposed current trace; an incompatible initial trace contributes to projection
error. For the zero-shift Neumann problem, the retained modes are mu-divergence
free with the corresponding normal-flux gauge condition. Inadmissible gradient
content also contributes to the residual; it is not automatically corrected.
The segment length must match the physical mesh extrusion. This is a geometry
precondition, not something the storage identity independently verifies.
