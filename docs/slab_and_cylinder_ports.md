# Slab and cylinder modes: geometry, ports and polynomial families

These analytic definitions apply to homogeneous conductors with no end effects,
an unshifted recurrence and the stated symmetric drive. They do not rank
general three-dimensional A–T, A–phi or T–Omega discretizations.

## Symmetric slab

Let the thickness be `2a`, with equal imposed tangential electric field at
`x=-a` and `x=a`. Then `E_z` is even, `H_y` is odd, and

```text
kappa = sqrt(s*mu*sigma)
E_z(x) = E_surface*cosh(kappa*x)/cosh(kappa*a)
H_y(x) = E_surface*sigma*sinh(kappa*x)/(kappa*cosh(kappa*a))
K = integral_-a^a sigma*E_z dx = 2*H_y(a)       [A/m]
Z_sheet = E_surface/K = kappa*coth(kappa*a)/(2*sigma) [ohm]
```

The DC limit is removable: `Rdc=1/(2*a*sigma)` and `Ldc=mu*a/6`.
For `z=s*mu*sigma*a^2`, `Z_sheet/Rdc=sqrt(z)*coth(sqrt(z))`.
A rectangular segment has `Z_terminal=(length/width)*Z_sheet`, subject to the
assumption that edge and end effects can be neglected.

In normalized coordinates with `a=mu=sigma=1`, the unit-surface electric modes
are `E_(2k)=P_(2k)(x)` and the unit-sheet-current magnetic modes are
`H_(2k+1)=P_(2k+1)(x)/2`. The Legendre norm
`integral_-1^1 P_n(x)^2 dx=2/(2*n+1)` yields

```text
R_(2k)/Rdc = 4*k+1                    -> 1, 5, 9, 13, 17
L_(2k+1)/Ldc = 3/(4*k+3)              -> 1, 3/7, 3/11, 1/5
```

The derivative identity
`(P_(n+1)-P_(n-1))'=(2*n+1)*P_n` also gives the integrated Maxwell recurrence:
the previous odd magnetic mode cancels the lower-degree antiderivative term.
For the electric update, integration from the surface preserves the prescribed
unit trace. This is why the accumulated field and the homogeneous correction
must be distinguished. The potential maps are `H=-A'` with homogeneous A at
both surfaces, and `T'=E` with odd T. An arbitrary constant in H violates the
chosen odd symmetry.

See the Legendre row in [NIST DLMF Table 18.3.1](https://dlmf.nist.gov/18.3)
and the [polynomial relations](https://dlmf.nist.gov/18.9).

## Circular conductor

For radius `a`, the total current is `I=2*pi*a*H_theta(a)` [A]. Consequently
`Z_per_length=E_z(a)/I` [ohm/m]. Omitting `a` in the circumference changes
the quantity being computed; setting `a=1` can hide that error.
The correct regular analytic expression is

```text
Z_per_length = kappa*I0(kappa*a)/(2*pi*a*sigma*I1(kappa*a))
Rdc = 1/(pi*sigma*a^2)                [ohm/m]
Ldc = mu/(8*pi)                      [henry/m]
```

With `rho=r/a`, the normalized modes are radial Zernike polynomials:

```text
E_(2k)(rho) = R_(2k)^0(rho) = P_k(2*rho^2-1)
H_(2k+1)(rho) = R_(2k+1)^1(rho)/(2*pi*a)
R_(m+2k)^m(rho) = rho^m*P_k^(0,m)(2*rho^2-1)
```

Here `P_k` is Legendre, `P_k^(alpha,beta)` is Jacobi, and `R_n^m(1)=1`.
For example `R_2^0=2*rho^2-1`, `R_4^0=6*rho^4-6*rho^2+1`, and
`R_3^1=rho*(3*rho^2-2)`. The label `m=1` belongs to the magnetic radial
polynomial; an axisymmetric azimuthal vector has sine/cosine Cartesian
components and is not a nonaxisymmetric scalar excitation.

The radial norm `integral_0^1 rho*(R_n^m)^2 drho=1/(2*(n+1))`, combined
with the physical disk measure `2*pi*a^2*rho drho`, produces
`R_(2k)/Rdc=2*k+1` and `L_(2k+1)/Ldc=1/(k+1)`.
The circular and slab coefficient sequences therefore use different measures
even though both admit classical orthogonal-polynomial descriptions.

The function is implemented as `ZernikeR[n,m,rho]` in
[Wolfram Language](https://reference.wolfram.com/language/ref/ZernikeR.html).
This supplies a useful independent symbolic reference for a Maxwell-integrated
mode construction, instead of maintaining another production implementation.

## Verification boundary

The formulas above are analytic derivations and definitions. New Mathematica
execution of the slab and built-in Zernike comparisons has not been performed.
They do not imply an all-frequency finite-ladder error bound, a timing result,
or finite-element convergence. Any future symbolic run must report the
normalization, actual runtime, exact identities checked and failure outcomes.
No saved third-party notebook output is treated as an independently executed
validation result. Public source code and images from supplied reference
material are not reproduced here.
