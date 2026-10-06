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
elements from a two-state SPD system and proves exact impedance equality.
[TEAM 28](11_team28_force.ipynb) additionally checks the field volume integrals,
energy orthogonality, circuit/field impedance and electrical loss balance in FEM.
Its six magnetic states contain twelve physical elements; these counts are not
interchangeable. This example does not demonstrate shifted or Type2 recursion.

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
