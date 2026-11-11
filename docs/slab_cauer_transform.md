# Slab E-mode: finite Cauer conversion and AC-axis accuracy

For the homogeneous symmetric slab, the finite full-order conversion preserves
the port impedance and reconstructed fields. Truncating the transformed circuit
changes its accuracy band. The result depends on the order of the original
finite circuit.

## Normalization and port

The slab occupies -a <= z <= a, has constant positive mu and sigma, and carries
unit sheet current K. End and edge effects are neglected. Define x=z/a,
tau=mu*sigma*a^2, u=s*tau and Rdc=1/(2*a*sigma). The numerical tables use
w=omega*tau and s=j*omega.

With k=sqrt(u), the exact normalized solution is

- Z/Rdc = k*coth(k).
- J/(K/(2*a)) = k*cosh(k*x)/sinh(k).
- H/(K/2) = sinh(k*x)/sinh(k).
- A/(mu*a*K/2) = (cosh(k)-cosh(k*x))/(k*sinh(k)).

Coordinates and vector directions are chosen so that dH/dx=J and dA/dx=-H
in the normalized fields. A vanishes at both surfaces. Errors in A use this
same gauge. Removable limits at w=0 are J=1, H=x and A=(1-x^2)/2.

## Independently constructed finite model

Expand current in P_(2k)(x), k=0,...,N-1. Integrate each polynomial once to
obtain the odd magnetic basis and again from the surface to obtain A.
The normalized resistance Gram matrix is diagonal, R_kk=1/(4k+1);
L_ij=integral_0^1 H_i H_j dx is tridiagonal. With b=(1,0,...),
Y_N=b^T(R+u L)^(-1)b and Z_N=1/Y_N. Scaling the solved coefficients to
b^T c=1 enforces the sheet-current port.

An exact L-orthogonal Krylov transformation starts with L^(-1)b and applies
L^(-1)R. It makes L diagonal and R tridiagonal. The leading m-dimensional
blocks give the high-frequency truncations.

The executable verification checks the following routes.

1. The accumulated magnetic/electric energy recurrence applied to the reversed
   pencil K=L, N=R.
2. Rational L-orthogonal Lanczos/Krylov projection.
3. Independent rational extraction of series inductance and shunt resistance
   at infinity.

All eight prefix impedances from these three routes agree exactly. The
ordinary low-frequency ladder synthesized from R_k=4k+1 and L_k=1/(4k+3)
agrees exactly with the finite Legendre model. A separate two-stage unit-element
example with shunt L, series R, shunt L and series R gives the transformed
elements 2R, 4L/5, 2R/25, L/5, and its full impedance agrees exactly.

These checks establish agreement between the conversion routes above once
port, scale, topology and termination are aligned. Here I denotes the
low-frequency representation and II the reversed-pencil representation; other
texts may assign these labels differently. The comparison uses a complete
finite model with N resistors and N inductors.

## AC-axis 1 percent impedance bands

The recorded model has N=8. Error is abs(Z_m(j*w)/Z_exact(j*w)-1).
The logarithmic scan contains 1801 points over 10^-4 <= w <= 10^5,
with 40 bisection iterations at each detected crossing. The intervals below
are scan-detected bands. A left endpoint of 10^-4 is the scan boundary.
The scan does not constitute a proof excluding narrower missed intervals.

| Retained stages m | I | II |
| --- | --- | --- |
| 1 | [0.0001, 0.6834] | none detected |
| 2 | [0.0001, 8.302] | none detected |
| 3 | [0.0001, 35.51] | [571.10, 1353.90] |
| 4 | [0.0001, 100.83] | [208.84, 1357.50] |
| 5 | [0.0001, 229.83] | [97.83, 1357.52] |
| 6 | [0.0001, 454.98] | [52.50, 1357.52] |
| 7 | [0.0001, 815.52] | [30.42, 1357.52] |
| 8 | [0.0001, 1357.52] | [0.0001, 1357.52] |

The upper limit remains finite because the parent eight-stage model
approximates the infinite-dimensional slab. For any chosen material and
thickness, f=w/(2*pi*mu*sigma*a^2).

## Field reconstruction

Errors are sampled relative complex l2 norms at 201 equally spaced positions.
The exact reference uses the hyperbolic solution above. The transform is
applied to the original polynomial field basis before reconstruction.

| w | Retained model | J error | A error | H error |
| --- | --- | --- | --- | --- |
| 10 | I, 3 stages | 1.262% | 0.0994% | 0.4107% |
| 10 | II, 3 stages | 148.1% | 80.05% | 79.96% |
| 100 | I, 3 stages | 33.38% | 12.48% | 25.84% |
| 100 | II, 3 stages | 46.51% | 54.74% | 45.52% |
| 1000 | I, 3 stages | 75.01% | 103.3% | 98.89% |
| 1000 | II, 3 stages | 7.372% | 6.932% | 6.760% |
| 1000 | either, 8 stages | 7.718% | 1.297% | 5.285% |

Thus a 1 percent port-impedance criterion can coexist with larger local-field
errors. At w=1000 the transformed three-stage approximation has slightly
smaller field error than its full parent, through cancellation of approximation
errors. This single sample does not establish monotonic convergence.
Both full-order reconstructions coincide within the 25-digit arithmetic used.
Very small printed zero errors reflect that numerical precision.

## Reproduction and scope

Run Mathematica with one kernel thread:

    wolframscript -file mathematica/verify_slab_cauer_transform.wls

The output includes named exact checks, impedance scans and field errors.
The stored portable record is [slab_cauer_transform.json](data/slab_cauer_transform.json).
Mathematica 14.2.1 can emit arbitrary-precision zero as 0.e-N; the saved record
normalizes that spelling to JSON 0.0e-N and converts numeric data to binary64.
The producer hash identifies the executed script. Its private raw output and
resource evidence are retained separately.

A 70-digit mpmath check builds the matrices directly from Legendre
orthogonality, then recomputes all recorded impedance samples and field-error
values. For the transformed circuit it uses the same L-orthogonal Krylov
method as the Mathematica script, in a separate implementation. It is
therefore a cross-implementation check of one method, not an independent
second method. The documentation validation CI runs it:

    python validation/validate_slab_cauer_transform.py docs/data/slab_cauer_transform.json

The mpmath check reproduced the stored impedance-error samples within
2.35e-14 and the field-error values within 1.82e-16 in absolute difference.
For each reported 1 percent band it checks that the error equals 1 percent at
each refined endpoint, does not exceed 1 percent at the geometric midpoint and
at the scan points next to each edge, and exceeds 1 percent at the adjacent
scan point outside each edge. Every recorded impedance sample must lie inside
a band exactly when its error is at most 1 percent. Edges at the scan limits
10^-4 and 10^5 are not crossings and have no outside check.

The scope is a homogeneous linear slab and its finite polynomial model.
Nonlinear permeability, hysteresis, arbitrary geometry and finite-element
discretization require their own verification.

The public derivation is independently written. Background on the Legendre
field expansion is given by Shindo, Miyazaki and Matsuo,
[Cauer circuit representation, IEEE Transactions on Magnetics 52(3), 6300504](https://doi.org/10.1109/TMAG.2015.2478139).
