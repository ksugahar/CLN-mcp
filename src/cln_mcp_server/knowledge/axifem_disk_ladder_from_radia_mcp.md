# Disk/sphere ladder time constants (removed from radia-mcp axifem knowledge)

Source: ksugahar/Radia packages/radia-mcp/src/radia_mcp/radia_ngsolve/knowledge/axifem.py (branch release/radia-mcp-2.0.0 before this change).

* Cross-validating a reduced eddy-current model (Hiruma 3-term Lanczos) against BEM.

---

* `research/hiruma_disk/test_hiruma_disk_q2.py` — full disk Cu disk
  eddy-current Hiruma 3-term, expects τ₁ ≈ 223.7 µs.

---

## Lanczos stage time constants vs BEM (Phase 3-(3))

The eddy-current comparison endpoint is the set of per-stage time constants
of the reduced tridiagonal (Lanczos) model of the disk:

    tau_pair[k] = lambda_{2k+1} * lambda_{2k+2}

Two independent routes produce the same quantities:

  (A) BEM-Foster reference: Mathematica computes 50
      Foster eigenvalues and amplitudes on a 1920-element ring mesh, 20
      moments alpha_n are derived, and a 50-digit mpmath continued-fraction
      (Lanczos-equivalent) extraction reads off the stage constants.  We do
      NOT implement the verified-interval-arithmetic part, so the values are
      high-precision floats, not interval-rigorous bounds.

  (B) Differential-equation Henrotte FE + Hiruma 3-term Lanczos:
      C++ axihenrotte at order=1 / order=2, with the 3-term recurrence
      reading off the tridiagonal coefficients directly:
          lambda_{2k+1} = w_{2k+1}^T K w_{2k+1}   (conductance-like)
          lambda_{2k+2} = w_{2k+2}^T M w_{2k+2}   (inductance-like)

The Foster-amplitude normalisation differs between BEM and FE, so absolute
coefficients differ by a common scale factor; tau_pair[k] is
normalisation-invariant and is the comparison endpoint:

```
k   BEM ref      p=2 fine    p=1 very-fine    p=2/BEM gap    p=1/BEM gap
0   219.32 us    218.71      218.05          -0.28 %        -0.58 %
1    78.65       78.12        77.77          -0.68 %        -1.12 %
2    40.04       39.54        39.37          -1.24 %        -1.66 %
3    23.74       23.16        23.14          -2.46 %        -2.54 %
4    17.07       16.07        16.06          -5.86 %        -5.91 %
5    14.70       13.12        13.01         -10.77 %       -11.50 %
```

axihenrotte p=2 beats axihenrotte p=1 at every k.  The high-mode (k >= 4)
divergence is the combined effect of FE basis-order error at higher modes
plus the numerical conditioning of high-stage moment extraction (BEM itself
starts producing negative tau for k >= 6).

## Moment convention matters

The Hiruma 3-term Lanczos recurrence works on the impedance-form generating
function
  f_H(s) = bT (K - sM)^-1 b,        alpha_n^H = bT (K^-1 M)^n K^-1 b
(Krylov-Pade impedance).  The susceptibility form
  f_S(s) = uT M (sM - K)^-1 M u,    alpha_n^S = uT M (K^-1 M)^n u,  u = M^-1 b
is the one that matches the analytical Stoll Bessel spectrum of the sphere
and the Stoll-equivalent BEM-Foster reference.

Sphere ground truth (Cu a=10 mm, B0=1 T uniform; Mathematica Hankel-Pade
240 digits on the Stoll spectrum):
  k=0  susceptibility form tau = 694.142 us   impedance form tau = 728.85  (+5.0%)
  k=1  susceptibility form tau = 154.604 us   impedance form tau = 171.51  (+10.9%)
  k=2  susceptibility form tau =  64.075 us   impedance form tau =  71.68  (+11.9%)

Implication for axifem: the tau_1 ~ 223.7 us cylinder reference (Hiruma) is
in the impedance convention and must not be mixed with BEM-Foster values
(Stoll/susceptibility convention) without switching the moment formula.

---

Cross-validation reference (Cu disk R=10 mm, t=2 mm,
sigma=5.8e7, B0=1 T, leading stage time constant tau_pair[0]):

| method                              | tau_pair[0] [us] | gap to BEM |
|-------------------------------------|------------------|------------|
| BEM-Foster reference               | 219.32 (ref)     | --         |
| axihenrotte p=2 (Hiruma 3-term)     | 218.71           | -0.28 %    |
| axihenrotte p=1 very-fine (Hiruma)  | 218.05           | -0.58 %    |
| 3D HCurl (NGSolve + Kelvin)         | ~ 218.7          | < 1 %      |
| Cylinder axisym VIM (144 cells)     | 211.85           | -3.4 %     |

The Cylinder VIM is run at a single coarse grid as a sanity check, not for
high accuracy; the axifem `p=2` Q-element remains the recommended
production path for axisym Cu eddy-current reduced-model extraction.

The same workflow on Cu sphere R=10 mm (Stoll Bessel ground truth, tau_1 = mu0 sigma R^2 / pi^2 = 738.48 us):

| method                              | tau_pair[0] [us] | gap to Stoll |
|-------------------------------------|------------------|--------------|
| Stoll analytical (mu0 sigma R^2/pi^2) | 738.48 (ref)   | --           |
| **axifem p=2 + z-offset Kelvin**    | **738.47**       | **-0.001 %** |
| axifem p=2 + Kelvin + Curve(2)      | 738.69           | +0.028 %     |
| 3D HCurl (NGSolve + Kelvin)         | ~ 694            | 0.027 % at L1 (Stoll tau=694 convention) |
| Sphere axisym VIM (480 cells)       | 708.4            | +2.07 %      |

The axifem + Kelvin result is the closest to Stoll across all available
axisym/3D methods on this benchmark (machine-precision agreement on the
leading time constant). See `axifem_documentation(topic="kelvin")` for the
full canonical recipe (Phase B3, commit 81f6415f).

---

  research/hiruma_disk/test_hiruma_disk_q1.py   # disk Hiruma 3-term, p=1

---

  research/hiruma_disk/test_hiruma_disk_q2.py   # disk Hiruma 3-term, p=2
