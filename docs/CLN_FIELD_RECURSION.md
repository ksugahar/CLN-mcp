# Physical excitation, field recursion, and circuit elements

For a linear reciprocal diffusion model, let `K` be the positive definite magnetic
energy matrix after boundary/gauge constraints, `M` the positive semidefinite
conductivity matrix, and `f` the actual source per unit port current. Solve
`(K+s M) A = f I`. These matrices include the physical volume measure.
The induced port impedance is `Z(s)=s f^T (K+s M)^(-1) f`.
Winding resistance is a separate circuit element when it is included.

The current-driven, magnetic-first Type1 recursion at zero expansion point is

```text
a_0 = K^-1 f; e_-1 = 0
L_j = a_j^T K a_j
e_j = e_(j-1) + a_j/L_j
R_j = 1/(e_j^T M e_j)
a_(j+1) = a_j - R_j K^-1 M e_j
```

`L_j` is a magnetic-energy norm (twice the energy at unit modal current).
`1/R_j` is a Joule-loss norm at unit modal voltage. This normalization fixes the
elements in henry and ohm. The electric modes are mutually `M`-orthogonal and
the magnetic modes mutually `K`-orthogonal in exact arithmetic. A finite ladder
starts with a shunt `s L_0`, continues through series `R_0` and shunt `s L_1`,
and terminates at the last series resistance in this example.

[The Wolfram proof](../mathematica/derive_energy_recursion.wls) constructs the
elements for an exhausted two-state SPD system (`n=N=2`) and proves exact
impedance equality. `K=I` and a source along an axis are without loss of
generality by congruence and rotation. That symbolic part does not establish
the general truncated `n<N` case or semidefinite `M`. The script separately
checks two exact rational truncated instances with non-identity `K` and
semidefinite `M`, proving ladder/Galerkin equality and `2n` matched moments of
`Z/s` for those instances. These checks are not a general theorem.
[TEAM 28](11_team28_force.ipynb) additionally checks the field volume integrals,
energy orthogonality, circuit/field impedance and electrical loss balance in FEM.
Its six magnetic states contain twelve physical elements; these counts are not
interchangeable. This example does not demonstrate shifted or Type2 recursion.

For prescribed coil current, the induced impedance tends to zero as `s` tends
to zero. The first shunt `s L_0` therefore gives the magnetic-first Type1 form;
excluded winding resistance would be a separate series element.

Both CLN and snapshot/POD methods can use the actual physical excitation.
Their distinguishing features are the field recursion, normalization, sampling
information and error properties. An Arnoldi basis alone is not a demonstrated
physical ladder, even when its span matches the CLN span. Terminal equivalence
also does not establish local field or force accuracy: reconstruct and test
those observables separately.

## Implementation checks

* Save modes in independent arrays (`a.copy()`) or new `GridFunction` storage.
  `saved = gf` aliases mutable storage and does not save a field snapshot.
* Check energy/loss orthogonality and positive finite elements at every run.
  A nearly exhausted field space can cause cancellation; an unrepresentable
  stage should fail explicitly instead of silently changing the algorithm.
* Check gauge/load compatibility before solving singular formulations. A gauge
  projection cannot justify removing a physically incompatible source.
* Boundary conditions and field spaces may differ between the initial driven
  field and subsequent homogeneous corrections. This can be deliberate design;
  an assignment of a boundary label alone is not evidence of a bug.
* Distinguish reduction error on one mesh from mesh, quadrature, outer-domain,
  material and experimental uncertainties. A failed implementation is not a
  proof that three-dimensional CLN is structurally impossible.

In TEAM 28 the regular axisymmetric variable is `w=A_theta/r`. Finite `w` on
the axis gives `A_theta=r w`, `B_r=-r d_z w`, and `B_z=2w+r d_r w` without
singular integration weights. The finite outer-air boundary has `w=0`; axis
regularity does not mean imposing `w=0` there.

## General 3D voltage port and Type2

[The notched-conductor experiment](12_general_3d_cln.ipynb) uses a one-volt
harmonic scalar lift `e0=-grad(phi0)`, side insulation and equipotential end
contacts. Its total terminal current is `I=G0-s*f.T*a`; a reaction transfer
alone omits the DC conduction port. The magnetic boundary `n x A=0` encloses
the conductor, with no exterior return-path energy.

For admissible currents, let `r(u,v)=integral(u.v/sigma)` and
`ell(u,v)=integral(B[u].B[v]/mu)`. Define `F_s0` by the same continuity-constrained
field problem at real `s0`. The resistor-first physical recurrence is

```text
j0 = F_s0(e0)
Rhat[2k] = 1 / (r(jk,jk) + s0*ell(jk,jk))
xk = x_previous + Rhat[2k]*jk
L[2k+1] = ell(xk,xk)
j_next = jk - F_s0(A[xk])/L[2k+1]
```

At zero shift, the resistor norm is pure Joule loss. At positive shift the
magnetic term is essential; store both physical integrals rather than call
the composite norm pure loss. The electric modes are orthogonal in
`r+s0*ell`, while the accumulated magnetic modes are orthogonal in `ell`.
An `m`-electric-mode ladder ending on its `qL` branch (`q=s-s0`) has contact
order `2m` in the tested discrete examples. This statement is checked through
the full-model Taylor coefficients, not solely at `s=s0`.

The T-port construction is `J=I*j_dc+curl(T)` with a free scalar `I` and
tangential `T=0` only on the insulating side, including contact rims. The
correction currents have zero net terminal flux; imposing tangential `T=0`
everywhere would remove the current port. Shared lifts must be disclosed:
agreement forced by a common lift is not independent convergence evidence.
In the notched example, both T forms also share zero-shift R2 because the
common lift and current space give the same first correction load.

Surface enrichment can supplement insufficient current distributions, but
later-stage disagreement should first be separated into truncation and full
FE-space/port discrepancies. Adding modes need not produce a simple physical
R–L ladder; test the coupled energies and port response before asserting that
representation. The current notched benchmark contains no surface enrichment.

## Surrounding air and an explicit return

For an enclosure example, conductivity and Joule norms remain conductor-only,
while the magnetic inverse and energy extend over conductor plus air. An
inductive forcing A[x] is restricted to the conductor before assembly; its
magnetic response is solved globally. Thus ell=ell_conductor+ell_air in every
recursion step. An external series inductance generally changes all shifted
Cauer elements, not only the first inductance.

Zero air conductivity adds gradient null directions to the global magnetic
operator. Include the complete global gradient map, verify the scalar
restriction identities and original-equation residuals, and check physical
fields under gauge-weight changes. A tiny artificial air conductivity changes
the physical problem and is not a gauge fix. The return geometry and source
break are part of the impedance definition; a conductor bonded to the same
perfect return on both terminals would short the intended voltage drive.

The surrounding-air A–T comparison uses conductor currents and the shared
global A inverse. A single-valued T–Omega air scalar cannot supply nonzero
circulation around the conductor; a cut or normalized topological source is
needed before claiming an equivalent T–Omega model.

A large external series inductance can make high-order impedance coefficients
sensitive to cancellation when an admittance series is inverted. Compute
Taylor coefficients directly from a unit-current constrained full system,
with work-conjugate voltage as output, and compare with an independently
constructed reduced system. This is a verification method, not a replacement
of the field recursion. NumPy longdouble is platform-dependent; record its
epsilon rather than assume it provides more precision than float64.

See notebook 13 for the candidate implementation and its review status.

## Surface trace restriction inside the field recursion

Air can be eliminated through a physical harmonic extension on the gradient
quotient. Its interface DtN is symmetric and nonnegative; its trace kernel
must remain explicit. A Schur complement of a gauge-penalised matrix is an
algebraic reproduction control, not automatically the physical air energy.

For stable truncation, restrict the admissible tangential trace to a nested
Steklov space and clamp omitted coefficients. At fixed current load, magnetic
compliance then approaches the full value from below. Split energies, driven
impedance and individual Cauer elements need not be monotone. Replacing the
DtN by a low-rank matrix on unrestricted traces can instead create zero-cost
physical directions; test coercivity and source coupling.

A protected actual DC port trace makes the first inductance exact by
construction. Report unprotected controls and dynamic response convergence
rather than treating that identity as independent accuracy evidence. The
physical Type2 recursion and total-energy normalization remain unchanged;
see notebook 14 for the numerical candidate and its review status.
