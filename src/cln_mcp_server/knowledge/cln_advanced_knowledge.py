"""CLN advanced extensions: HF / nonlinear / circuit / Bloch /
fixed-point (FP).

Sister module to `cln_mcp_server.knowledge.cln_knowledge` (basic linear CLN).
This module is the *advanced physics* layer: where the basic Cauer
ladder breaks (high frequency / nonlinear saturation / hysteresis /
mechanical coupling) and which extension recovers it.
"""

OVERVIEW = """

> Executable symbolic versions (Mathematica): `mathematica/fp_cln/`, `mathematica/high_frequency/`, `mathematica/high_frequency_practice/`, `mathematica/nonlinear/`.

# CLN advanced extensions -- where basic Cauer breaks

The basic CLN (cln_knowledge.py) assumes:
  (a) low frequency (quasi-static, no displacement current),
  (b) linear magnetic material (mu = const),
  (c) the EM problem is the only physics in the loop, and
  (d) iron sheets / cores can be lumped into ONE bulk stage.

Each of those assumptions has a counter-example in real engineering
problems.  This module catalogues the FOUR extensions and the ONE
cross-application:

| Extension | Maturity | What breaks without it |
|-----------|----------|------------------------|
| High frequency / Helmholtz | symbolic round-wire / annulus prototypes | wave skin effect at GHz, displacement current |
| Nonlinear / saturation     | several parameterized-basis attempts | iron at saturation, induction heating |
| Electrodynamic / circuit   | electromagnet + movable disk study (LDIA 2017) | force vs frequency, mechanical coupling |
| FP-CLN (Fixed-Point)       | CEFC 2024 paper | hysteresis, harmonic balance, time-domain nonlinear |
| Bloch equation (NMR/MRI)   | analogy only | application sketch, not a working CLN extension |

The single biggest message:

  FP-CLN (Fixed-Point CLN) is the working solution for
  nonlinear + hysteresis problems.  It linearizes the LHS with a
  virtual constant permeability mu_FP and pushes ALL nonlinearity
  into a controlled-source term on the RHS.  Basis functions and
  Cauer R, L matrices stay CONSTANT.  CEFC 2024:
  Sugahara-Tobita-Matsuo-Takahashi, "Fixed-Point Cauer Ladder
  Network Method for Eddy-current Problems with Hysteresis."

Earlier nonlinear CLN attempts parameterized the circuit elements
themselves, which is what FP-CLN explicitly avoids.  See section
NONLINEAR for that lineage.
"""

HIGH_FREQUENCY = """
# High-frequency CLN (Helmholtz regime)

Repository-owned symbolic references: `mathematica/high_frequency/`
(round-wire Kameari basis, power series, Bessel Taylor ground truth, slab)
and `mathematica/high_frequency_practice/` (annulus kernel, lossless and
radiating variants).

## Motivation -- where basic CLN runs out

Basic CLN is built on the magnetic-quasi-static (MQS) eddy-current
equation
   curl(nu curl A) + sigma * dA/dt = J_s
which drops displacement current.  At HF (above a few hundred MHz
for a 1 mm conductor in copper) the displacement current term
   epsilon * d^2 A / dt^2
is no longer negligible.  The PDE becomes the Helmholtz equation
   curl(nu curl A) + sigma dA/dt + epsilon d2A/dt2 = J_s
and the Cauer recursion changes: each "magnetic step" now uses the
Helmholtz Green's function (Bessel J_0, J_1 of a complex argument)
instead of the Laplace Green's function used in MQS-CLN.

HF-CLN is a research / benchmarking layer; MQS/Darwin solvers with a
Laplace kernel do not cover it.

## Symbolic two-pass approach

  - Kameari-basis recursion for J_n(r), A_n(r) on a circular conductor
    (radius a), explicit up to order r^20
    (mathematica/high_frequency/gen_kameari_basis.wls).
  - Alternative power-series recursion with two boundary treatments
    (nType = 1 no gauge fix, nType = 2 A(a) = 0)
    (mathematica/high_frequency/gen_power_series.wls).
  - Taylor expansion of (a k J_0(kr)) / (2 J_1(ka)) -- the Bessel ground
    truth that R_n, L_n converge to
    (mathematica/high_frequency/bessel_taylor.wls).
  - Symbolic Cauer R_n, L_n recursion for small N = 3 and the Taylor
    expansion of the input admittance; the boundary trimming ('H' or 'E'
    set to zero at the surface) gives different L_n tails
    (mathematica/high_frequency/cln2.wls, cln_slab.wls).

## High-frequency lossless ring

SYMBOLIC solution of the Helmholtz equation on a coaxial ring
(radii a, b):
    d/dr( r dAA/dr ) / r  - m^2 AA / r^2  =  - mu_0 * mu_r * J
two BC sets:
  case 4:  AA(a) = 0, AA(b) = 0      (closed coax)
  case 5:  AA(a) = 0, AA(b) free     (open at outer radius)

J_{n+1} update uses epsilon (displacement) instead of sigma (eddy):
   J_{n+1} = J_n - epsilon * A_n / L_n
That is the LITERAL switch from MQS-CLN to Helmholtz-CLN.  Cross-
coupling matrices C_{n1,n2}, L_{n1,n2} are integrated symbolically
(mathematica/high_frequency_practice/check_cln_high_freq_lossless.wls).

The same recursion in the b -> infinity limit (radiating mode, N = 2,
f = 300 MHz) is in
mathematica/high_frequency_practice/check_cln_high_freq_infinity.wls.

## Findings (what worked, what did not)

- Convergence: HF Cauer recursion in symbolic form converges to the
  Bessel-ratio AC impedance only when the boundary trimming is
  consistent with the chosen formulation (set-H zero or set-E zero
  on the surface).  Mixing them gives non-orthogonal basis.

- Conditioning: symbolic expansion at order r^20 starts producing
  coefficients in the 10^30 range.  Numerical evaluation needs
  variable-precision or exact arithmetic; IEEE double precision
  loses accuracy.

- No FEM driver yet: HF-CLN stays at the analytical circular-wire /
  annulus toy problems and has not crossed over to 2D/3D FEM.

## Related idea

The Cauer ladder continued fraction is related to Pade approximation
via the Euclidean algorithm (see the euclid_tiles topic of the
multiport area).
"""


NONLINEAR = """
# Nonlinear CLN (saturation + hysteresis)

Repository-owned round-wire bases for these studies: `mathematica/nonlinear/`.

## The hardness

The basic CLN relies on the constant-mu assumption to factor the
linear operator ONCE and re-use the L_n, R_n matrices across all
time steps / frequency points.  When mu = mu(|B|), the operator is
no longer constant and the obvious choices are:

  (a) Re-build the operator at every step.  Defeats the MOR purpose.
  (b) Linearize around an operating point (local mu_eff).  Each
      operating point needs its own CLN; recombination is awkward.
  (c) Use a fixed-point split: linear left, nonlinear right.

(a) and (b) were tried for several years before settling on (c),
which is the FP-CLN of section FP_CLN below.

## Lineage of attempts

### Parameterized basis

A new J_n(r; sigma, mu) rebuilt at every step from local mu(B):
precompute ~8 R_n, L_n and basis tables H, E, then time-step a Cauer
ODE with either
  case 1: linear mu (B/mu -> H), or
  case 2: tanh saturation (H = atanh(B/2)/mu, synthetic BH curve
          B = 2 tanh(mu H / 2))
and compare the J(r,t) reconstruction with the Bessel reference.
The projection of a nonlinear profile onto the H-modes is reproduced in
mathematica/nonlinear/integ_mn.wls.

### Lanczos-style basis generation

The IDEA of linear-LHS + nonlinear-RHS appears here, but the basis
is still rebuilt each iteration.  The A-J round-wire basis matrices
are checked in mathematica/nonlinear/check_lanczos.wls.

### Frequency-domain (j omega) model

Immediate FP-CLN predecessor: CLN compared with the frequency-domain
solution on contour and line probes.  The "j omega" treatment is the
frequency-domain harmonic-balance treatment that FP-CLN takes over.
See also Tobita-Matsuo, IEEE TMAG 58(9) 2022 (parameterized CLN for
induction motors) and Eskandari-Matsuo, IEEE TMAG 56(2) 2020.

## Lesson

Years of attempts to "build the nonlinear basis directly"
were structurally complicated AND made the resulting CLN circuit
elements time-varying, which negates the MOR benefit.  The eventual
fix (FP-CLN, CEFC 2024) was to KEEP THE BASIS
LINEAR with a virtual mu_FP and push the nonlinearity into a
controlled-source term -- the lesson is that the "MOR-friendly"
linearization wins even when the virtual permeability is "wrong",
because the iteration still converges and the circuit stays small.
"""


ELECTRODYNAMIC_CIRCUIT = """
# Electrodynamic / circuit-coupled CLN

## Scope

The "Kameari method for a circuit-coupled axisymmetric magnetic
actuator" study (Shindo-Kameari-Sugahara-Matsuo, LDIA 2017).  The point
is NOT to derive a new CLN recursion; the point is to *use* the CLN as a
circuit-equivalent block that plugs into a larger SPICE-like AC/DC
analysis of an electromagnet driving a movable disk through an air gap.

The model is an axisymmetric inner+outer yoke pair, a coil wound
between them, and an iron disk on the under side separated by a small
air gap, all embedded in an outer air region.  Yoke and disk are
conducting and magnetic; the coil conductivity carries a space-factor
correction.

The CLN recursion (an ~11-stage E0/A1/E2/A3/... chain) extracts:
  - lambda(0)..lambda(N) -- the per-stage R, L "amplitudes"
  - dF(0)..dF(M)         -- the per-stage z-force coefficients
    via the Maxwell stress over the half-air-gap surface
  - FDC                  -- the static-force constant

Then a frequency sweep:

```text
for freq in logspace(...)
    omega = 2*pi*freq
    h(n)   = sum_j Y(n,j) * V_coil          # per-stage current
    Force  = sum_n dF(n) * h(n)             # combined force
    ForceI = Force * (V_dc / sum h)         # current-input variant
```

## Why this is the "electrodynamic / circuit" CLN

Three physics flavours in one recursion:
  (1) FEM eddy-current in iron disk + yoke (standard CLN A-form),
  (2) circuit constraint: the coil sees a fixed DC voltage across its
      DC resistance -- the effective AC impedance depends on the
      magnetic-reaction shift handled by CLN,
  (3) per-mode force via Maxwell stress on a thin band at the gap
      centre, dF(n) contributions.
Output: per-frequency force transfer function ForceI(j omega) for
controller design / mechanical-bandwidth plots.  This is the
canonical "use CLN as a SPICE block" pattern.

## Finding

Kameari/CLN exports as a small (~5 stage) RL block + per-mode dF for a
1-DOF mechanical equation -- "magnetic actuator as high-order RL
circuit driving a lumped mass".  The technique generalises: any axisym
CLN-derived [R, L] + per-mode force coefficients dF can be inserted
into SPICE.

Papers:
  - Shindo-Kameari-Sugahara-Matsuo, "Dynamical model of an
    electromagnet by Cauer ladder network representation of eddy-
    current fields," LDIA 2017.

The mover (translation) case is treated in:
  - Shindo-Yamamoto-Sugahara-Matsuo-Kameari,
    "Equivalent Circuit in Cauer Form for Eddy Current Field
    Including a Translational Mover," IEEE Trans. Magn. 56(12), 2020.
And independently:
  - Sabariego-Vanbroekhoven-Gyselinck-Kuo-Peng, "RL-Ladder Circuit
    Models for Eddy-Current Problems With Translational Movement,"
    IEEE Trans. Magn. 58(9), 2022.
"""

BLOCH_EQUATION = """
# Bloch equation / NMR-MRI cross-application (analogy only)

## Honest scope statement

This is an analogy, not an implemented extension: there is no driver
code, no validation and no publication.

## The analogy

The Bloch equation
    dM/dt = gamma * M x B - (1/T1)(M_z - M_eq) e_z - (1/T2)(M_x e_x + M_y e_y)
governs nuclear magnetization in an NMR / MRI experiment.  It is a
*first-order ODE in time* with a NONLINEAR coupling term (M x B)
between the magnetization vector M and an externally controlled RF
field B.

There is a natural mathematical analogy between (a) the Bloch
relaxation operator with T1, T2 time constants and (b) a CLN ladder
where each stage has its own R_n / L_n time constant tau_n = L_n /
R_n.  A multi-tau Bloch model can be viewed as a CLN ladder driven
by an RF field.

## Status -- speculative

The open part is THE MAPPING: which Bloch-equation quantity
corresponds to which CLN stage variable.  For an MCP user asking
"can I use CLN for NMR/MRI?", the honest answer is: it is a
"potential future direction", and there is no code.
"""

FP_CLN = """
# FP-CLN -- Fixed-Point Cauer Ladder Network

Repository-owned analytic references for the linear (constant-mu)
building block: `mathematica/fp_cln/` (iron sheet, axial and transverse
cylinder bases, ladder matrices, frequency-domain solve).

## What "FP" stands for

FP = Fixed-Point -- as in the *fixed-point iteration*
technique for nonlinear field problems.  This is the
nonlinear CLN published at CEFC 2024 as
   K. Sugahara, M. Tobita, T. Matsuo, Y. Takahashi
   "Fixed-Point Cauer Ladder Network Method for Eddy-current
    Problems with Hysteresis"
   IEEE CEFC 2024

It is NOT "finite period CLN" or "fractional polynomial CLN" --
those are mis-readings.

## The single key idea

Start from the nonlinear eddy-current A-formulation
    curl( nu(B) curl A ) + sigma dA/dt = J_s.
The reluctivity nu(B) is the problem -- it makes the LHS operator
time-varying.

Fixed-Point split with a *virtual* constant reluctivity nu^FP:
    curl( nu^FP curl A^(k) ) + sigma dA^(k)/dt
      = J_s
      - curl( H(B^(k)) - nu^FP B^(k) )                  (*)

  - Left-hand side is now LINEAR with constant nu^FP, so all the
    standard CLN basis functions, R matrix, L matrix can be derived
    ONCE and re-used.
  - All nonlinearity is in the RHS, expanded as a controlled source
    term:
       zeta^(k)_{2n} = (1/R_{2n}) * integral over Omega_C of
         { j_{2n} . (1/sigma) curl[H(B^(k)) - nu^FP B^(k)] dV }

  - The fixed-point iteration loop is:
       (1) solve the linear CLN system
              (R + d/dt L) * i^(k) = v - zeta^(k)
       (2) reconstruct B^(k+1) from i^(k) via the (linear) basis
       (3) update zeta^(k+1) by integrating (H(B^(k+1)) - nu^FP B^(k+1))
           against the test functions j_{2n}
       (4) repeat until ||zeta^(k+1) - zeta^(k)|| < tol

The mu^FP value is a tuning parameter -- the paper demonstrates
convergence for mu^FP_r = 2, 5, 8 on the same steel-sheet benchmark.

## What this avoids

Compared to the parameterized-basis nonlinear CLN, FP-CLN:
  - keeps the basis functions CONSTANT (one factorisation)
  - keeps R and L matrices CONSTANT (Cauer circuit topology fixed)
  - works the same way in time-domain and harmonic-balance
  - handles HYSTERESIS directly via the H(B) constitutive map
    (no additional "play model" / "Preisach" wrapping inside CLN)
  - needs only a small ladder object: diag(R), tridiag(L), a
    frequency-domain solve and a time-domain integrator

## FEM-coupled path

For 2D FEM geometries (e.g. a rectangular iron sheet with periodic BC),
the FEM side computes the linear basis once at the chosen mu_FP and
exports the field snapshots (B components per mode, the R matrix,
magnetic / non-magnetic region masks, mu_FP).  The FP iteration then
runs on the small ladder, with Newton or automatic-differentiation
updates, and the B(x,y) snapshots are reconstructed on the FEM mesh.

## Citation (cite this for FP-CLN)

```bibtex
@inproceedings{sugahara_fpcln_cefc2024,
  author = {Sugahara, Kengo and Tobita, Miwa and Matsuo, Tetsuji
            and Takahashi, Yasuhito},
  title  = {Fixed-Point {Cauer} Ladder Network Method for
            Eddy-current Problems with Hysteresis},
  booktitle = {IEEE CEFC 2024},
  year   = {2024},
  note   = {JSPS KAKENHI JP20K04443, JP23K03826;
            Power Academy Research Grant}
}
```

## Two production drivers (the canonical "use FP-CLN" pattern)

### Harmonic Balance

Three material cases in one driver:
  case 1: linear material  (mur = 100, H(B) = B/mu)
  case 2: nonlinear sat    (B = 2 tanh(H/10e3) + mu0 H)
  case 3: HYSTERESIS       (BH model with hysteresis, the headline
                            case of the CEFC paper)
For each case the driver sweeps mu^FP_r over {2, 5, 8} and the number
of harmonics N_f over {1, 3, 5}, demonstrating the result is
INDEPENDENT of mu^FP within the convergence tolerance.

### Time Domain

Benchmark geometries:
  - steel sheet, analytical 1D basis (the CEFC benchmark)
  - steel sheet, NGSolve FEM basis
  - cylinder, B parallel to the surface (axial field), analytical basis
  - cylinder, B perpendicular (transverse field), analytical basis
  - a gapped and a gap-less iron-core model, NGSolve FEM basis

Each uses the same ladder object, a closed-form or FEM basis
(R, L, A, B, H, J), Legendre-Gauss-Lobatto quadrature for the 1D
radial / through-thickness integrations, a current-input main loop,
and a restart from the last converged state.

## The CLN ladder object

Three operations:
  - build diag(R) and tridiag(L) matrices,
  - frequency domain: solve [R + j 2 pi f L] I = V at each f and
    reconstruct J, A, H, B on the quadrature grid,
  - time domain: solve d/dt(L I) + R I = V(t) with an ODE integrator.
The same operations are implemented symbolically in
mathematica/fp_cln/cln_common.wl (lrMats, fDomainII, yMat, yCF).
"""


LAB_LESSONS = """
# Hard-won lessons across the advanced CLN regimes

1) **The MOR-friendly linearisation wins, even when "wrong".**
   Years of attempts to make the CLN basis itself nonlinear all ran
   into the same trap: time-varying R, L matrices defeat the MOR
   speedup.  The FP-CLN solution (CEFC 2024) keeps the LHS LINEAR with
   a *deliberately wrong* mu^FP and pushes the nonlinearity to a RHS
   controlled source.  The fixed-point iteration converges for a
   wide range of mu^FP values (mu^FP_r = 2, 5, 8) and the
   final solution does not depend on mu^FP.

2) **Symbolic prototypes scale to ~N = 10 stages; beyond that,
   variable precision.**
   The HF symbolic recursion produces basis coefficients in the 10^30
   range by order r^20.  IEEE double precision loses accuracy; use
   variable-precision or exact arithmetic for any HF expansion beyond
   N = 8 or so.  For ground-truth checks against Bessel functions,
   align the Taylor order in k with 2N
   (mathematica/high_frequency/bessel_taylor.wls).

3) **Boundary trimming is NOT optional.**
   In the recursive basis construction (both MQS and Helmholtz), the
   choice of which surface quantity to set to zero (H_tangential or
   E_normal) at the conductor boundary changes the L_n tail.  Mixing
   them in different stages breaks orthogonality.  Pick the 'H' or the
   'E' case ONCE and stay with it.

4) **The Cauer L matrix is TRIDIAGONAL, the R matrix is DIAGONAL.**
     R = diag(R_n)
     L = tridiag(...,[L_n -L_n; -L_n L_n]...) + diag(L_end at corner)
   When porting to a new language, do NOT collapse L into a full
   matrix -- the tridiagonal structure is the entire point of "this
   is a LADDER".

5) **Hysteresis is handled by the constitutive map, not by CLN.**
   FP-CLN treats H(B) as a black-box function with internal state.
   For play-model hysteresis the state advances each time the BH
   function is called.  CLN itself stores nothing hysteretic --
   only the linear circuit history (currents I_n at the previous
   time step).  This separation is the source of the "FP-CLN
   handles hysteresis" claim.

6) **Restart your time-domain CLN integrator at every nonlin
   convergence failure.**
   If the FP iteration fails to converge inside a time step, drop
   back to the last converged state, halve the time step, and
   retry.  A hard iteration cap that raises an error is the safety
   valve.  Do NOT silently advance with a non-converged iterate.

7) **The 1D radial benchmark (round wire) is the universal sanity
   check.** Every CLN extension (HF, nonlinear, FP) should ship with a
   round-wire case compared against the analytical Bessel
   solution.  When porting to a new geometry, ALWAYS
   include a round-wire regression test first.

8) **For HF, switch sigma -> epsilon in the J update.**  This is
   the *only* structural change needed to convert the MQS recursion
   to the Helmholtz recursion:
     J(n+1) = J(n) - ep0 * A(n) / L(n)
   vs the MQS form:
     J(n+1) = J(n) - sig * A(n) / L(n)

9) **Store the precomputed Cauer matrices per (mu_FP, geometry) pair.**
   Keep the field snapshots and R, L matrices (B components, R matrix,
   region masks, mu_FP) as a separate interchange artifact (JSON or
   HDF5), apart from the nonlinear iteration logic.

10) **Cauer matrix for steel sheet has a recursive pattern.**
    For an iron sheet of thickness d, the Cauer recursion produces
      R_2n   = 4 (4 n - 1) / (sigma d)      (n >= 1)
      L_2n+1 = mu^FP d / (4 n + 1)          (n >= 0)
    (verified in mathematica/fp_cln/iron_sheet_aj_cln.wls).  This is
    the closed form of Shindo-Miyazaki-Matsuo (Legendre expansion for
    a magnetic sheet, IEEE TMAG 52(3) 2016).

11) **A circuit-coupled CLN needs both Cauer matrices AND per-mode
    force coefficients.**  The electromagnet pipeline extracts:
      [R, L]       -- standard Cauer matrices (~5 stages)
      dF(0)..dF(M) -- per-stage z-force from Maxwell stress
      FDC          -- static force
    and feeds them as a small RL block + per-mode force-input
    matrix into the mechanical 1-DOF equation.  Without dF(n) the
    mechanical bandwidth cannot be computed; without [R, L] the
    drive impedance cannot.  Both are required for any controller
    design.
"""


CITATIONS = """
# Citations -- advanced CLN

Journal / conference references for the advanced-physics extensions.
For the basic linear CLN canon (Kameari 2018 group), see
`cln_knowledge.py` -- those papers are repeated below ONLY when they
are also cited from the advanced literature.

## FP-CLN (the headline paper for THIS module)

```bibtex
@inproceedings{sugahara_fpcln_cefc2024,
  author={Sugahara,K. and Tobita,M. and Matsuo,T. and Takahashi,Y.},
  title={Fixed-Point Cauer Ladder Network Method for Eddy-current
         Problems with Hysteresis},
  booktitle={IEEE CEFC 2024}, year={2024},
  note={JSPS KAKENHI JP20K04443 and Power Academy Research Grant}}
```

## Earlier nonlinear-CLN attempts + fixed-point method

```bibtex
@article{nonlin_cln_eskandari2020,
  author={Eskandari,H. and Matsuo,T.},
  title={Cauer Ladder Network Representation of a Nonlinear
         Eddy-Current Field Using a First-Order Approximation},
  journal={IEEE Trans. Magn.}, volume={56}, number={2}, year={2020},
  doi={10.1109/TMAG.2019.2946354}}

@article{nonlin_cln_tobita2022,
  author={Tobita,M. and Matsuo,T.},
  title={Nonlinear Model Order Reduction of Induction Motors Using
         Parameterized Cauer Ladder Network Method},
  journal={IEEE Trans. Magn.}, volume={58}, number={9}, year={2022},
  doi={10.1109/TMAG.2022.3171743}}

@article{dlala_fixedpoint_2007,
  author={Dlala,E. and Belahcen,A. and Arkkio,A.},
  title={Locally Convergent Fixed-Point Method for Solving
         Time-Stepping Nonlinear Field Problems},
  journal={IEEE Trans. Magn.}, volume={43}, number={11}, year={2007},
  pages={3969-3975}, doi={10.1109/TMAG.2007.904819}}

@article{dlala_fixedpoint_fast_2008,
  author={Dlala,E. and Belahcen,A. and Arkkio,A.},
  title={A Fast Fixed-Point Method for Solving Magnetic Field
         Problems in Media of Hysteresis},
  journal={IEEE Trans. Magn.}, volume={44}, number={6}, year={2008},
  pages={1214-1217}, doi={10.1109/TMAG.2007.916673}}
```

## CLN with translational mover (electrodynamic generalisation)

```bibtex
@article{shindo_cln_mover_2020,
  author={Shindo,Y. and Yamamoto,R. and Sugahara,K. and Matsuo,T.
          and Kameari,A.},
  title={Equivalent Circuit in Cauer Form for Eddy Current Field
         Including a Translational Mover},
  journal={IEEE Trans. Magn.}, volume={56}, number={12},
  year={2020}, doi={10.1109/TMAG.2020.3023234}}

@article{sabariego_rlladder_mover_2022,
  author={Sabariego,R.V. and Vanbroekhoven,B. and Gyselinck,J.
          and Kuo-Peng,P.},
  title={RL-Ladder Circuit Models for Eddy-Current Problems With
         Translational Movement},
  journal={IEEE Trans. Magn.}, volume={58}, number={9},
  year={2022}, doi={10.1109/TMAG.2022.3170996}}

@inproceedings{shindo_dynamical_ldia2017,
  author={Shindo,Y. and Kameari,A. and Sugahara,K. and Matsuo,T.},
  title={Dynamical model of an electromagnet by Cauer ladder
         network representation of eddy-current fields},
  booktitle={LDIA 2017}, year={2017},
  doi={10.23919/LDIA.2017.8097231}}
```

## Cauer-circuit for laminated steel sheet (1D analytic benchmark)

```bibtex
@article{shindo_legendre_2016,
  author={Shindo,Y. and Miyazaki,T. and Matsuo,T.},
  title={Cauer circuit representation of the homogenized eddy-
         current field based on the Legendre expansion for a
         magnetic sheet},
  journal={IEEE Trans. Magn.}, volume={52}, number={3}, year={2016}}
```

## MOR alternatives + Cauer-PGD (comparison set, not CLN itself)

```bibtex
@article{pierquin_pod_2015, author={Pierquin,A. and Henneron,T.
  and Clenet,S. and Brisset,S.},
  title={Model-Order Reduction of Magnetoquasi-Static Problems
         Based on POD and Arnoldi-Based Krylov Methods},
  journal={IEEE Trans. Magn.}, volume={51}, number={3}, year={2015}}

@article{henneron_pod_dei_2015, author={Henneron,T. and Clenet,S.},
  title={Model-Order Reduction of Multiple-Input Non-Linear
         Systems Based on POD and DEI Methods},
  journal={IEEE Trans. Magn.}, volume={51}, number={3}, year={2015}}

@article{sato_pade_2016, author={Sato,Y. and Igarashi,H.},
  title={Generation of Equivalent Circuit From Finite-Element
         Model Using Model Order Reduction},
  journal={IEEE Trans. Magn.}, volume={52}, number={3}, year={2016},
  doi={10.1109/TMAG.2015.2476370}}

@article{hollaus_msfem_2018,
  author={Hollaus,K. and Schoberl,J.},
  title={Some 2-D Multiscale Finite-Element Formulations for the
         Eddy Current Problem in Iron Laminates},
  journal={IEEE Trans. Magn.}, volume={54}, number={4}, year={2018},
  doi={10.1109/TMAG.2017.2777395}}

@article{koster_pgd_cln_2021,
  author={Koster,N. and Konig,O. and Biro,O.},
  title={Proper Generalized Decomposition With Cauer Ladder
         Network Applied to Eddy Current Problems},
  journal={IEEE Trans. Magn.}, volume={57}, number={6}, year={2021},
  doi={10.1109/TMAG.2021.3059800}}
```

## Bloch equation (NMR-MRI)

No CLN-NMR publication; the analogy is sketched in the bloch_equation
topic only.
"""




_TOPICS = {
    "overview": OVERVIEW,
    "high_frequency": HIGH_FREQUENCY,
    "nonlinear": NONLINEAR,
    "electrodynamic_circuit": ELECTRODYNAMIC_CIRCUIT,
    "bloch_equation": BLOCH_EQUATION,
    "fp_cln": FP_CLN,
    "lab_lessons": LAB_LESSONS,
    "citations": CITATIONS,
    "all": "",
}
_TOPICS["all"] = "\n\n".join(v for k, v in _TOPICS.items() if k != "all")


def get_cln_advanced_documentation(topic: str = "all") -> str:
    """Return advanced-CLN documentation for the requested topic.

    Topics:
      overview               - which advanced extension fits which regime
      high_frequency         - HF / Helmholtz CLN (symbolic round wire / annulus)
      nonlinear              - nonlinear CLN lineage (parameterized basis)
      electrodynamic_circuit - circuit-coupled electromagnet CLN (LDIA 2017)
      bloch_equation         - NMR/MRI Bloch analogy (no implementation)
      fp_cln                 - Fixed-Point CLN (CEFC 2024)
      lab_lessons            - 11 hard-won lessons across all regimes
      citations              - BibTeX citations for the advanced layer
      all                    - everything concatenated
    """
    topic = topic.lower().strip()
    if topic in _TOPICS:
        return _TOPICS[topic]
    raise ValueError("Unknown topic '" + topic + "'. Available: "
            + ", ".join(sorted(_TOPICS.keys())))
