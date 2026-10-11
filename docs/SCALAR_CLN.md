# V/Phi CLN with compatible high-order local recovery

The new `CLN_VPhi` route replaces all global physical A-T vector solves.
Only scalar electric/magnetic facet potentials are globally factored. Both
the physical flux and the cell potential are eliminated and recovered on
each element. Compatible element curl reconstruction supplies the particular
potentials needed by the next scalar step. Original A-T remains a separately
executed comparison reference; its solver and vector-condensation constructor
are guarded against calls throughout the scalar route.

## Relation to Bíró's formulation

[Bíró and Köster, Compumag 2021, Sections III-IV](https://www.compumag.org/Proceedings/2021_Cancun/www.conftool.pro/compumag2021/index.php/OA1-2-299ecbe.pdf?filename=OA1-2-299.pdf&form_id=299&form_version=final&ismobile=false&page=downloadPaper)
introduce particular vector potentials for the known current and flux, then
solve scalar V/Phi Poisson equations. Their nodal realization does not by
itself make the computed flux an exactly compatible discrete curl. The source
is an accessible conference manuscript; the cited Bíró/Gyimesi journal full
text remains unavailable, and no unverified exact one-pass theorem is assumed.

This implementation keeps those scalar PDEs but uses a **hybrid-mixed
discretization**, rather than claiming to reproduce the paper's nodal code.
The [NGSolve hybrid-method tutorial](https://docu.ngsolve.org/latest/i-tutorials/unit-2.7-hybrid/hybrid.html)
provides the established discontinuous-flux, cell-scalar, facet-scalar pattern.
NGSolve owns orientation, Piola mappings, curved geometry, quadrature and
element elimination. No proprietary curl orientation or geometry code is used.

## Equations and boundary conditions

For a particular field P, find q, u and its facet trace uhat satisfying

    (q,v) - sum_e (u,div v)_e + sum_e <uhat,v.n>_boundary(e) = (P,v),
    (div q,w)_e = 0,
    sum_e <q.n,what>_boundary(e) = 0.

These are the mixed weak equations q = P - grad u and div q = 0.
The last equation enforces normal continuity and any prescribed zero normal
boundary flux. Within each element the flux/cell block is invertible; its
elimination gives a scalar facet Schur complement. We factor its negative
with `sparsecholesky`. A single scalar gauge fixes the pure-Neumann Phi
problem; no mass perturbation is added. The full unanchored physical equations
are subsequently checked by an independent matrix-free application.

At HCurl order p >= 2, use discontinuous HDiv order p-1, L2 order p-2 and
FacetFESpace order p-1. These degrees match the divergence-free curl image of
the existing full HCurl space. Boundary and interior normal moments agree to
the scalar solve tolerance. The element inverse from notebook 20 can therefore
recover a conforming potential, instead of attempting to invert incompatible
nodal fluxes. Its repeated shared-DOF corrections still require multiple sweeps.

The present constant-material wire permits unit-weight scalar projections;
physical current and flux are then J = sigma*q and B = mu*q. This rescaling
does not extend to variable material coefficients without changing the forms.
The boundary conditions intentionally match the existing internal-wire A-T
case, not a new air/terminal model:

* Initial electric particular field is e_z, with zero scalar correction on
  the full boundary. The total voltage lift is V0 = -z plus that correction.
* Magnetic particular field is R*T, with zero normal B on the full boundary
  and a pure-Neumann Phi correction.
* Electric increment has particular field -A/L, with zero V correction on
  the full boundary. The particular A has zero tangential trace.

After each magnetic/electric increment, accumulate B/J exactly as in A-T,
compute L = integral(B.B/mu) and R = 1/integral(J.J/sigma), and reconstruct
the potential of the cumulative field for the following step.

## Why this replaces the vector solve

On the simply connected wire, the divergence-free HDiv spaces (with or
without zero boundary normal flux) are the corresponding discrete curl
images. Integrating by parts gives

    (B_increment/mu, curl N) = (R*T, curl N) = (R*J, N),
    (J_increment/sigma, curl W) = (-A/L, curl W) = (-B/L, W).

The boundary terms vanish under the matched conditions. Thus each scalar
mixed projection satisfies the original A-T physical equation, while its
global unknown is scalar. No original vector solution supplies any scalar
right-hand side. HCurl spaces in this route serve only element curl recovery
and independent weak residual tests; no global curl-curl matrix is assembled.
There is still a global scalar solve and an iterative local curl inverse.

## Evidence and operating limits

Notebook 21 reads native `scalar_cln_study.json` and `scalar_nodal_control.json`.
Four cohorts cover p=2/3/4, maxh 3 mm and p=3 at 2 mm, curved geometry order
3, quadrature bonus 8, zero shift and three full CLN stages. Original ICCG,
condensed A-T and scalar V/Phi are run separately on the identical mesh.
Four complete runs retain a first observation and three warmed observations;
route order alternates. NGSolve uses four threads and BLAS is fixed to one.
Setup and all checks are included; mesh generation and cross-comparison are
excluded. Raw times are retained; one-host medians are not scaling guarantees.

Every scalar system must pass its original uncondensed residual <= 1e-6.
The original A-T weak physical equation and the equation after curl recovery
must independently pass the same limit. The relative element divergence is
scaled by the declared 10 mm wire length and must be <= 1e-8; local curl fit
is requested at 1e-10 and integrated recovery error must be <= 1e-8. Coefficient
and volume-vector-field parity limits are 1e-6. Field energies and 78-point
vector samples provide additional saved-data checks. Nodal Phi controls keep
their correctly solved scalar systems separate from incompatible recovered
fluxes; incompatibility is never silently accepted or repaired by A-T.

This is a verified replacement for these linear internal-wire cases. It is
not yet a validated general-domain, air, variable-material, cohomology,
positive-shift or nonlinear scalar implementation. No speedup is presumed.

## Reproduction

Native NGSolve/Radia/NumPy/SciPy/threadpoolctl environment:

    python validation/scalar_cln_study.py --output docs/data/scalar_cln_study.json
    python validation/scalar_nodal_control.py --output docs/data/scalar_nodal_control.json

The scalar route alone, without executing an A-T reference:

    python -m validation.cln3d.CLN_VPhi --maxh .003 --order 3 --stages 3 --output scalar-result.json

Portable saved-evidence gates, without NGSolve:

    python validation/validate_scalar_cln.py
    python validation/validate_notebooks.py
