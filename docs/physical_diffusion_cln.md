# Energy-based physical CLN for diffusion

The implementation derives a modal RC circuit from material-weighted field
integrals, and checks that the field and circuit quadratic forms coincide.
It is not a fitted impedance with an assumed physical interpretation.

## Equation and scope

For a scalar reciprocal diffusion problem,

\[
-\nabla\cdot(a\nabla u)+b\dot u=fF,
\qquad Ku+M\dot u=\ell F,\quad y=\ell^T u,
\]

the assembled `K` and `M` must be real symmetric positive definite after fixed
degrees of freedom are removed. The source and observation must be collocated.
The matrix interface supports arbitrary such discretizations, including
coercive Robin problems. The native NGSolve helper uses homogeneous Dirichlet
boundaries (with natural zero flux on the remainder). Pure Neumann null modes,
advection, nonlinear materials, and multiple ports are not implemented.

The dense MCP tool accepts 1–256 full-order DOFs and 1–64 requested stages.
Use the native FE interface for larger spatial models. Supplying matrices
does not check their physical units or mesh convergence; these remain properties
of the caller's discretization.

## Material-normalized recurrence

Starting with `q=load`, `h_previous=0`, each stage computes

\[
Kt_i=q_i,\quad R_i=t_i^TKt_i,\quad e_i=t_i/R_i,
\quad h_i=h_{i-1}+e_i,\quad C_i=h_i^TMh_i,
\quad q_{i+1}=q_i-Mh_i/C_i.
\]

In the continuum, \(\widehat Q_i=-a\nabla t_i\), so
\(R_i=\int a^{-1}|\widehat Q_i|^2\) and
\(C_i=\int b h_i^2\). These are material-weighted norms of the flux and
potential modes. With `E=[e_i]`, `H=[h_i]`, and upper bidiagonal `B` having
diagonal +1 and superdiagonal −1,

\[
E=HB,\quad E^TKE=G=\operatorname{diag}(1/R_i),\quad
H^TMH=\operatorname{diag}(C_i),\quad
E^TME=M_c=B^T\operatorname{diag}(C_i)B,\quad E^T\ell=e_1.
\]

The volume-only flux norm above assumes no dissipative boundary form. For a
caller-assembled Robin problem, include its surface term in K; it contributes
to R and D in addition to the volume gradient integral.

The node potentials `v` and capacitor drops `w=Bv` are distinct coordinates:

\[
u=Ev=Hw,\qquad Q=\sum_i(v_i/R_i)\widehat Q_i,\qquad
M_c\dot v+Gv=e_1F,\quad y=v_1.
\]

The circuit has shunt resistors, capacitors between consecutive nodes, and
the final capacitor to reference. It is a modal network, not a conventional
series-resistor/shunt-capacitor thermal discretization. Retaining **every**
capacitor is necessary for these Galerkin identities. The older
`rc_ladder_impedance(..., termination="resistance")` drops the last capacitor;
use `termination="capacitance"` or `model.impedance(s)` for this model.

## What is conserved, and in what units?

\[
W=\tfrac12u^TMu=\tfrac12\sum_i C_iw_i^2,\quad
D=u^TKu=\sum_i v_i^2/R_i,\quad \dot W+D=Fy.
\]

For heat, `u` is temperature excess, `a` is conductivity, and `b` is volumetric
heat capacity. With a unit-integral continuum source, `F` is injected power,
`y` is source-weighted excess temperature, `R` is K/W, and `C` is J/K.
**W is J K, and D and Fy are W K.** This balance is a quadratic storage law,
not the first-law heat balance. Actual reconstructed excess heat is

\[
U=\int b u\,d\Omega=\left(E^Tg\right)^Tv,\qquad
g_j=\int b\phi_j\,d\Omega.
\]

Pass `storage_integral=g` to obtain `linear_storage_content` in joules.
After Dirichlet elimination, `M_free @ ones` is generally **not** this vector;
assemble the integral itself. Also, `sum(C_i*w_i)` is not generally U.
`W/T_ref` approximates near-equilibrium thermal availability for small
temperature deviations; it is not an exact thermodynamic identity.

The implementation also returns the rate-tested balance

\[
\frac{d}{dt}(\tfrac12u^TKu)+\dot u^TM\dot u=F\dot y.
\]

For magnetic vector-potential diffusion, this second identity can describe
magnetic energy and Joule loss. Physical energy must be identified using the
constitutive variables and units; neither metric is universally measured in J.

## MCP and Python usage

Install `pip install -e ".[energy]"` and launch `cln-mcp` through the MCP client's
stdio configuration. No solver process or arbitrary code execution is required
by the dense matrix tool. The tool `cln_diffusion_energy_formulation(physics="heat")`
returns the derivation and interpretation. Call `cln_diffusion_physical_model`
with, for example:

```json
{
  "stiffness": [[2, -1], [-1, 2]],
  "capacity": [[2, 0], [0, 1]],
  "load": [1, 0],
  "nstages": 2,
  "physics": "heat",
  "storage_integral": [2, 1],
  "nodes": [1, 0],
  "drive": 0.5
}
```

This illustrative algebraic system returns R/C, G/M/B, both bases, a netlist,
certification residuals, stage diagnostics, and the optional state balance.
The load is used **as supplied**, including signed zero-sum loads. To make `F`
total injected heat, assemble the load from a unit-integral continuum source;
the sum of free-DOF load entries is not a reliable continuum normalization.
The example integral weights are illustrative, not an inferred physical mesh.

The equivalent Python entry point is `cln.physical_cln(K, M, load, nstages)`.
It needs NumPy but does not import NGSolve. For native finite elements:

```python
from ngsolve import Mesh, x, y
from netgen.geom2d import unit_square
from cln import DiffusionCLN

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
model = DiffusionCLN(mesh, a=1+x, b=2+y, source=1+x,
                     order=4, length=2.5).energy_model(nstages=3)
state = model.balance([1., 0., 0.], drive=0.5)
temperature = model.reconstruct([1., 0., 0.])
flux = model.reconstruct_flux([1., 0., 0.])
```

The native helper normalizes a source with nonzero continuum integral and
provides the heat-content weights automatically. `length` is a constant
extrusion-depth multiplier. Solves, normalization, and certificate integrals
use the same quadrature; `integration_order` can be specified explicitly.
Quadrature consistency is necessary but does not establish quadrature accuracy.
Positive, real scalar material coefficients are caller preconditions; global
material positivity is not validated by the native helper's subspace audit.

## Certification, approximation, and PRIMA

The certificate reports dimensionless residuals for the two orthogonalities,
field/circuit capacity, and collocated port. The default tolerance is 1e-8.
If extra stages lose orthogonality, the largest passing prefix is retained;
`effective_stages`, `stop_reason`, and `rejected_larger_prefixes` expose this.
The MCP response also includes a top-level stage summary. Plain recurrence can
lose orthogonality early on ordinary FE systems: an independently checked
200-element slab retained 3 of 8 requested stages at tolerance 1e-8. The method
does not yet re-orthogonalize; always inspect the retained count and response
accuracy. Optional `nodes` must have that effective length.
The matrix interface also reports operator condition estimates and stops at
an exhausted invariant subspace. The native recurrence can raise an error
if its solve or normalization fails before the audit; it does not regularize
a singular operator. No failed model is presented as certified.

Certification concerns identities **on the retained subspace**, not uniform
full-order response accuracy, positivity of reconstructed temperature, or
representation of arbitrary initial fields. Check independent full-order
frequency/transient solutions and spatial convergence for each application.

PRIMA can preserve passivity and quadratic storage by congruence projection.
For the same Krylov subspace, the full capacitor-terminated CLN and a PRIMA-type
congruence have the same transfer function. The added value here is explicit
material-normalized alternating field bases, positive R/C elements, circuit
topology, reconstruction, and measured field/circuit identities. This is not
a claim of greater accuracy merely from changing coordinates.

Tests check independent circuit assembly, full-order solutions, the first 2N
moments of an SPD system, congruence equivalence, midpoint discrete
energy balance, variable-coefficient native FE integrals, and invalid inputs.

## References

- Tanimoto, Sugahara, Takahashi, Matsuo, “A Fundamental Study of the Formulation
  of a CLN Method with a Clear Physical Interpretation for Heat Conduction
  Analysis,” IEEJ SA-25-014/RM-25-014 (2025).
- Odabasioglu, Celik, Pileggi, “PRIMA: Passive Reduced-Order Interconnect
  Macromodeling Algorithm,” IEEE TCAD 17(8), 645–654 (1998),
  [doi:10.1109/43.712097](https://doi.org/10.1109/43.712097).
