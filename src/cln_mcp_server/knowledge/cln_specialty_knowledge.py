"""CLN specialty methods: termination, Hiruma, BEM+FEM, verified error bounds,
series expansion.

These are refinements of the canonical Kameari-Ebrahimi-Sugahara-Shindo-
Matsuo 2018 IEEE TMAG recursion that are not part of its standard
extensions.

The specialty topics, in order of increasing scope:

  termination      - what to put at the end of a finite ladder
  Hiruma method    - Lanczos-type algorithm giving CLN directly from the
                     algebraic (G + s C) form
  BEM + FEM        - coupling Kameari-CLN on the conductor with a
                     BEM exterior, multi-mode multi-port ladder
  error theory     - verified bounds on R_n, L_n and a mesh-adequacy
                     diagnostic for their finite-element discretisation

Companion module: cln_mcp_server.knowledge.cln_knowledge (the canonical Kameari-
Ebrahimi-Sugahara-Shindo-Matsuo recursion, multi-expansion-point
extension, applications).
"""

# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------

OVERVIEW = """

> Executable symbolic versions (Mathematica): `mathematica/bem_fem/`, `mathematica/hiruma/`, `mathematica/termination/`, `mathematica/series_expansion/`.

# CLN specialty methods (Group D)

The canonical CLN paper (Kameari-Ebrahimi-Sugahara-Shindo-Matsuo, IEEE
TMAG 54(3):7201804, 2018) defines a recursion that produces a ladder
of (R, L) stages.  Four research threads refine or extend that
recursion:

| Topic | Origin | Status |
|-------|--------|--------|
| Ladder TERMINATION impedance | magnetic-sheet analytic case | method + empirical finding |
| Hiruma's Lanczos-type algorithm -> CLN | Hiruma, Igarashi | published |
| Coupling open-boundary BEM with conductor-CLN | round-conductor 2D | multi-mode multi-port; ongoing |
| Verified error bound on R_n, L_n | Nagamine et al., CEFC 2022 | published + mesh rule of thumb |

Why these are "specialty" and not "core":

  * Termination - The canonical paper computes Z(s) by direct
    continued-fraction evaluation.  At a finite ladder length N, the
    response loses accuracy at high frequency.  Replacing the open or
    short termination with a frequency-dependent Z_term(s) recovers
    accuracy without enlarging N.

  * Hiruma - The canonical CLN recursion needs the differential
    operators (curl curl, etc.) of the original PDE.  Hiruma derived
    a Lanczos-type algorithm that ingests the ALGEBRAIC
    system (G + s C) x = b directly and produces the same Cauer
    ladder constants.  This makes CLN reachable from any black-box
    FEM matrix exporter.  It also unifies CLN with PVL / SyPVL /
    PRIMA (the PEEC MOR family) on the algebraic side.

  * BEM+FEM - The canonical CLN is interior-domain only.
    Coupling with an exterior BEM (eddy current in a finite
    conductor, free space outside) requires changing the conductor
    boundary condition from Dirichlet to Robin and re-deriving the
    Kameari recursion.  Result: a MULTI-PORT, MULTI-MODE ladder
    where each H_m mode contributes one port.

  * Error theory - The canonical paper proves CONVERGENCE of the
    ladder as N -> infinity but does not bound the FE discretisation
    error in any individual R_n, L_n.  Verified-arithmetic computation
    of the exact R_n, L_n (e.g. for the square cross-section) and a
    cut-off-frequency / characteristic-length diagnostic close this gap.

All four threads share the same starting point: the basic Kameari
recursion in cln_knowledge.py.  They diverge in WHERE they cut into
that recursion (termination at the tail, algebraic ingestion at the
front, BEM coupling at the side, error bound underneath).
"""

# ---------------------------------------------------------------------------
# TERMINATION
# ---------------------------------------------------------------------------

TERMINATION_THEORY = """
# Ladder TERMINATION

Repository-owned checks: `mathematica/termination/` (tail fitting of a
truncated Lambert continued fraction, two-ladder coupling).

## The problem

The Kameari recursion at stage N has computed R_0, L_1, R_2, ..., L_{2N-1}
(or up to R_{2N}, depending on whether the ladder ends on R or L).  The
truncated impedance is then evaluated by

   Z_N(s) = R_0 + 1 / (s L_1 + 1 / (R_2 + 1 / (s L_3 + ...))).

For a 1-D magnetic sheet in the H-mode (Shindo-Kameari analytic case),
the exact impedance is

   Z(s) = (1/sigma d) * tanh(sqrt(s sigma mu) d / 2)
                       / (sqrt(s sigma mu) d / 2)

and the truncated Z_N(s) ALTERNATES between under- and over-shoot of
the exact value as N grows (the "tail-flapping" pattern).

## The fix: terminate with a frequency-dependent impedance

Instead of stopping the ladder open (Z_end = infinity) or shorted
(Z_end = 0), insert a small impedance Z_term(s) at the tail.  If
Z_term(s) is chosen to approximate the rest of the ladder (the
infinite tail from stage N+1 onward), then Z_N(s) becomes accurate
over a much wider frequency range.

Two terminators:

   Z_term^E(s) = R_{2N} + s L_{2N+1}    (E-mode, R-then-L terminator)
   Z_term^H(s) = s L_{2N+1} + R_{2N+2}  (H-mode, L-then-R terminator)

where R_{2N+2}, L_{2N+3} are taken from the ANALYTIC asymptotic form
of R_n, L_n (here, R_n = sigma d / (2n+1), L_n = mu d / (2(2n+1))
for the magnetic sheet) -- they are NOT computed by FEM.

## Empirical findings

For a round-trip 2-conductor model:

  * Without termination, Z_N requires N >= 30 for 1 % accuracy over
    1 kHz to 1 MHz.

  * With ideal R-termination, Z_6 has the same accuracy as Z_{N>15}
    in the mid-frequency band; at high frequency the high-stage Z_N
    still wins because the truncated termination has its own error.

  * Conclusion: termination IMPROVES the lower-mid band but is NOT a
    substitute for adding more stages at high frequency.  Adding
    stages is more effective than optimising the terminator.

Use terminator-optimisation as a POLISH after fixing N, not as a way
to reduce N.

## Open question

Is there a closed-form Z_term(s) that makes Z_N(s) exact for s in some
finite band [s_min, s_max]?  The infinite-ladder limit is known:
Z_term^infinity(s) = (1/sigma d) tanh(...)/(...) -- but this requires
the analytic solution, which CLN is trying to avoid.  Looking for a
RATIONAL approximation of Z_term^infinity matched to N is an open
direction.
"""


# ---------------------------------------------------------------------------
# HIRUMA METHOD
# ---------------------------------------------------------------------------

HIRUMA_METHOD = """
# Hiruma method

Repository-owned checks: `mathematica/hiruma/hiruma_check0.wls` (two-unknown
exact test) and `mathematica/hiruma/sym_lcn.wls` (symbolic R-L ladder,
shifted to an expansion point s0).

## Origin

The method is due to Shingo Hiruma and Hajime Igarashi (Hokkaido
University): a Lanczos-type algorithm that produces the Cauer
continued fraction of a transfer function from the discretised
algebraic system.

## The core idea

Canonical Kameari CLN starts from the PDE (curl(nu curl A) + s sigma A
= J_0) and builds basis fields H_n, E_n by recursive static FEM
solves.  Each step needs the differential operators of the PDE.

Hiruma starts from the discretised algebraic system

   (G + s C) x = B u                                          (1)
   y = c^T x                                                  (2)

with G the FE conductance matrix and C the FE capacitance/mass matrix
(or, in the magnetic case, K + s N: K the stiffness, N the conductivity
mass).  The transfer function is

   H(s) = c^T (G + s C)^{-1} B u
        = l^T (I - s A)^{-1} r                                (3)

where  A = -G^{-1} C,  l = c,  r = G^{-1} B u.

The Lanczos-type recursion expands H(s) in a continued fraction
directly from (3) without ever touching the PDE.

## The recursion (scalar input/output)

Let w_1 = G^{-1} l = G^{-1} r (initial vector).  Set lambda_1 = w_1^T G w_1.
Then for n = 1, 2, ...:

   w_{n+1}^new = w_{n-1} - (A w_n) / lambda_n     if n odd  (or even - mirror)
   lambda_{n+1} = w_{n+1}^T M_{n+1} w_{n+1}       (M alternates G and C)

After N steps the recursion gives lambda_1, lambda_2, ..., lambda_{2N+1}
and these are EXACTLY the ladder constants R_0 = 1/lambda_1,
L_1 = lambda_2, R_2 = 1/lambda_3, L_3 = lambda_4, ...

The non-symmetric form (l != r) handles input-different-from-output
transfer functions; the same Krylov subspace K(A, u_0) is generated
twice as K(A, u_0), K(A^T, v_0), but this SHORTENS back to one
Krylov subspace K(A, u_0) when G is SPD by using G-orthogonality:

   (u_i, v_j)_G = u_i^T G v_j = 0   for  0 <= i < j <= N-1.

This is the key insight: the symmetric inner product (.,.)_G replaces
the two-sided non-symmetric inner product of vanilla non-symmetric
Lanczos.  The result is equivalent to SyPVL (Symmetric Pade via
Lanczos) but generates the CAUER (continued fraction) form instead of
the FOSTER (partial fraction) form of PVL.

## What Hiruma's method gains over canonical CLN

  1. Black-box FEM: any FEM solver that exports (G, C) works.  No
     need to expose H_n, E_n basis fields or curl operators.

  2. Unifies CLN with the PEEC MOR family (PVL, SyPVL, PRIMA, Arnoldi):
     the Kameari basis fields are a Lanczos sequence in disguise.  This
     is the bridge between the FEM-CLN literature and the PEEC-MOR
     literature.

  3. Multi-port extension is immediate: if b is replaced by a matrix
     [b1, b2, ...], the recursion runs block-wise and gives a multi-
     port Cauer network.

  4. Eddy-current loss in a coil layer (unit-cell sub-problem) can be
     CLN-expanded directly without re-deriving the PDE in the cell.

## What is LOST relative to canonical CLN

  * The basis fields H_n, E_n no longer have a physical
    interpretation -- they are abstract Krylov vectors.  Visualising
    the eddy-current modal pattern is harder.

  * The algorithm is conditional on a starting vector w_1 = G^{-1} l.
    Different starting vectors give different ladder constants;
    this is the "expansion point" question (next section).

  * Lanczos breakdown (lambda_n = 0 for some n) is possible and must
    be checked at each step.

## Hiruma + expansion point

Restart the Krylov sequence at s0 != 0: define K_{s0} = G + s0 C,
w_1 = K_{s0}^{-1} l, lambda_n alternating between K_{s0} and C
inner products.  Rewriting (G + s C) x = b as (K_{s0} + (s-s0) C) x = b
makes the truncated ladder exact at s = s0 and accurate over a wider
band than the s0 = 0 ladder -- the algebraic counterpart of the
Kuriyama 2019 multi-expansion-point CLN.

Empirical finding: imaginary-axis shifts (s0 = j omega_target) widen
the accurate frequency band; real-axis shifts improve the
matched-frequency accuracy.  The algorithm also RUNS when K_{s0} is not
SPD -- an open theoretical question.

## Hiruma + unit-cell problem

Apply the algebraic Cauer expansion to a UNIT CELL of a wire layer (a
single conductor cross-section in periodic BC).  Given a uniform
external dB/dt, the cell's (K + s N) system is Hiruma-expanded, and the
truncated ladder gives an effective mu_eff(s) for the layer.  This is
the "homogenisation" application that motivates the algebraic
approach: each unit cell shape (round, rectangular, litz) gets its CLN
without re-deriving the Kameari PDE recursion.  It is the algebraic
counterpart of the analytical-sheet Cauer circuit (Shindo-Kameari-Matsuo,
COMPEL 2017), ready for any cell geometry.
"""


# ---------------------------------------------------------------------------
# BEM + FEM
# ---------------------------------------------------------------------------

BEM_FEM_COUPLED = """
# BEM + FEM coupled CLN

Repository-owned check: `mathematica/bem_fem/check_basis_analytic.wls`
(Robin-boundary m = 1 round-conductor basis, exact in mu_r).

## The motivation

Canonical Kameari CLN puts the eddy-current conductor inside a closed
FEM domain with Dirichlet A=0 on its faces.  For a round-conductor
2-D problem the H_m mode current is then

   J_m(r) = -p_m * sigma * j omega * J_m(k r) * cos(m theta)    (1)

with J_m the m-th Bessel function and k = sqrt(-j omega mu sigma);
Kameari recursion with Dirichlet recovers (1).

For an OPEN-BOUNDARY problem (conductor in infinite air), only the
conductor is meshed; the air is handled by BEM (surface Green's
function).  The conductor boundary is now ROBIN (tangential H and
normal B continuity), giving

   J_m(r) = J_m(k r) * (m/k - mu*(J_{m-1} - J_{m+1})/2)^{-1}
            * (n . (H_inc + Sum_n p_n H_n^scat))   (2)

Same r-dependence J_m(k r) but a different (s-dependent, complex)
prefactor: the integration-by-parts BOUNDARY TERM picks up a different
contribution under Robin than under Dirichlet, so the lambda_n acquire
s-dependence -- no longer the clean integer sequence of equation (1).

## The structure of the coupled system

Let the conductor occupy domain Omega_c with surface Gamma; the air
is the unbounded exterior.

  FEM part   (interior, in Omega_c):
     A_z + Robin BC on Gamma:    n . curl A = f(A|_Gamma, H_inc)
     -> for each mode H_m, one Kameari recursion -> one port

  BEM part   (exterior, on Gamma):
     Surface integral equation for the secondary current/magnetisation
     -> a matrix P_BEM(s) coupling all surface DoFs

  Coupling:
     Continuity of A_z and (1/mu) dA_z/dn across Gamma
     -> a multi-mode multi-port Cauer ladder with mode index m and
        port index k (one port per conductor / per mode).

The BEM part uses a standard 2-D Biot-Savart kernel; the FEM part can
be done with any 2-D FEM code.

The result is a MULTI-PORT Hiruma-style Cauer ladder: each H_m mode
contributes one port, the BEM coupling between conductors contributes
the off-diagonal blocks of the multi-port Y matrix.

## Sub-experiment: TSVD basis reduction

Direct multi-port CLN with M conductors and N_m modes is M*N_m ports.
For M = 6, N_m = 36 (cosine modes on a circular conductor) this is
216 ports -- intractable.  TSVD (Truncated SVD) builds the BEM
"interaction matrix" Az between all boundary surface points and a
single source point, does SVD, and keeps the top K = 5-15 left singular
vectors as a reduced port basis.  The continuum analog (KL expansion of
the smooth BEM kernel has exponentially decaying singular values)
justifies the truncation.

## State of the work

  * H_1 mode: ladder converges to MoM reference at N=10, K=5.
  * H_2 mode: ladder converges at N=15, K=8.
  * Higher modes: BEM-side discretisation dominates; refine the BEM
    surface before refining N or K.

Open extensions (not yet done): 3-D version, coupling with a PEEC
PRIMA solver, validation against ngsolve.bem on a 3-D test case.

## Sub-experiments

  * PM hoverboard over a conducting plate, scalar (1-port lumped) vs
    matrix (multi-port H_m-mode) BEM+FEM CLN.  Matrix form gives 5-10x
    B-field accuracy improvement at the cost of a larger Hiruma
    recursion.

  * Coil-symmetry benchmark recipes (one / two / six coils; dipole,
    monopole, quadrupole, sextupole) against the Bessel-mode MoM
    reference: the Hiruma BEM+FEM CLN converges to the analytic MoM as
    N (stages) and K (TSVD modes) grow.
"""


# ---------------------------------------------------------------------------
# ERROR THEORY
# ---------------------------------------------------------------------------

NAGAMINE_ERROR_THEORY = """
# Error theory: verified R_n, L_n and mesh adequacy

## Published basis

  * H. Nagamine et al., "Convergence analysis of the Cauer ladder
    network method using eigenfunction expansion", 2022 IEEE 20th
    Biennial Conference on Electromagnetic Field Computation
    (CEFC 2022).

The approach applies VERIFIED NUMERICAL COMPUTATION (interval
arithmetic, affine arithmetic) to the CLN method to give rigorous
bounds on the circuit constants R_n, L_n.

## Verified exact constants for a square cross-section

For an infinite square conductor the EXACT R_n, L_n can be computed by
interval arithmetic (IA) and affine arithmetic (AA) via the Stieltjes
transform + quotient-difference algorithm -- analytic, not FEM.
Affine arithmetic (correlation tracking) tightens the intervals
by a factor of a few vs plain IA.  Lanczos needs SPD G; Arnoldi handles
non-symmetric G at higher cost.

## FE accuracy of the circuit constants

Comparing FEM-computed R_n, L_n at increasing mesh quality against the
verified exact values shows that the relative error is NOT
monotonically decreasing in n: beyond roughly stage 8-9 the FEM values
diverge from the exact ones for a fixed mesh.  The convergent-in-N CLN
ladder is NOT convergent for any finite mesh.

The reason: each stage's static FEM solve depends on the previous
stage's basis functions, so FEM error in A_n is amplified through the
recursion.  The recursion in 2D A-formulation is

   solve  (nu curl w, curl dA) = (sigma w, E_n) / G_n,   dA = 0 on the wall
   A_{n+1} = A_n + dA
   L_{n+1} = (nu curl A_{n+1}, curl A_{n+1})
   E_{n+1} = E_n - A_{n+1} / L_{n+1}
   G_{n+1} = (sigma E_{n+1}, E_{n+1})

## Characteristic frequency and length

Define, from the ladder constants already computed:

  * characteristic frequency f_n = max_i (R_{2i} / (2 pi L_{2i-1}))
    over the n cut-off frequencies of the Foster equivalent of the
    n-stage Cauer ladder (skin-depth analog for multi-frequency CLN)
  * characteristic length delta_n = 1 / sqrt(pi f_n sigma mu)

For the 1-D magnetic-sheet E-mode problem (Shindo-Kameari-Matsuo):
R_{2n} = sigma d / (4n+1), L_{2n+1} = mu d / (4*(4n+3)).
Numerically, f_n ~ n^{3.8-3.9}, consistent with an O(n^4)
effective-frequency-range scaling.

Empirical relation between the relative error of R_n, L_n and the
ratio delta_n / Delta_x (mesh size):

   delta_n / Delta_x = 100  ->  relative error ~0.001 %
   delta_n / Delta_x =  10  ->  relative error ~0.1 %
   delta_n / Delta_x =   1  ->  relative error several %

i.e. delta_n shrinks 10x -> relative error grows 100x.  The same scaling
holds for the cylindrical conductor (R_{2n} = (2n+1)/(sigma pi a^2),
L_{2n+1} = mu/(8(n+1)pi)).

PRACTICAL RULE: trust only stages with Delta_x <= delta_n / 10.

Anomaly: for very small Delta_x (around 1 um) the scaling breaks down --
relative error grows without delta_n shrinking, probably from
floating-point round-off in the recursive FE solve.  Open question.

## Why this matters

Without such a diagnostic, CLN is used with the IMPLICIT assumption
that all R_n, L_n are correct as long as the mesh is "reasonable".
Instead:

  1. Even at a fine mesh, high stages (8+) can have >10 % error in R, L
     for the square cross-section.

  2. There is a quantitative DESIGN RULE: pick the maximum stage n
     such that delta_n >= 10 * Delta_x.  Beyond that the FEM
     discretisation dominates.

  3. The characteristic length delta_n is a CHEAP proxy: it is
     computed from the (already-available) Cauer ladder constants,
     no extra FEM run is needed.

  4. The Foster equivalent (partial-fraction form) is more numerically
     stable than the Cauer (continued-fraction form) at high stage
     numbers, because the cut-off frequencies f_{ni} converge to
     fixed values as n grows.

This addresses the practical question "how many stages should I trust
for my mesh?", which convergence of the ladder alone does not answer.
"""

# ---------------------------------------------------------------------------
# LESSONS
# ---------------------------------------------------------------------------

LAB_LESSONS = """
# Lessons across the specialty methods

1. TERMINATION is a polish, not a substitute for stages.
   Optimising the terminator gives about the same accuracy as adding
   2-3 stages.  Add stages first; polish later.

2. HIRUMA is the right entry point for a black-box FEM.
   Given (K, N) matrices from any solver (NGSolve, another FEM code,
   a PEEC exporter), the algebraic Lanczos produces the Cauer
   ladder without re-deriving Kameari's PDE recursion per cell shape.

3. HIRUMA EXPANSION-POINT shifts: imaginary-axis first.
   Real-axis s0 improves accuracy at the matched relaxation frequency
   but degrades it on the imaginary (physical omega) axis.
   Imaginary-axis s0 = j omega_target widens accuracy over a band of
   physical frequencies -- the default choice.

4. Non-SPD shifted system "just works" for Hiruma -- empirically.
   K_{s0} = G + s0 C may not be SPD for s0 != 0, yet the
   Lanczos still converges.  Open theoretical question; treat as a
   useful surprise, not a guarantee.

5. BEM+FEM needs Robin BC, not Dirichlet.
   Canonical Kameari assumes closed Dirichlet A=0.  BEM coupling
   requires Robin (tangential H + normal B continuity); lambda_n
   change and acquire s-dependence.  Do NOT reuse a Dirichlet driver.

6. TSVD truncation makes multi-port BEM+FEM tractable.
   M conductors x N_m modes -> SVD-truncated to top K = 5-15
   effective ports.  Beyond K the BEM discretisation error dominates.

7. ERROR THEORY: trust only delta_n >= 10 * Delta_x.
   delta_n is computed in O(N) from already-available R_n, L_n
   (Foster cut-off).  Do NOT trust the first n stages just because
   they look converged.

8. CLN ladder is NOT monotone-convergent in n for a finite mesh.
   Both R and L can grow in error vs n for fixed Delta_x.  Do NOT use
   "the error stopped decreasing" as a stopping criterion; use
   delta_n >= 10 * Delta_x.

9. Foster (partial fraction) is more stable than Cauer (continued
   fraction) at high n.  Foster cut-off frequencies converge; the
   continued-fraction tail does not.  Use Foster for diagnostics
   (NOT for SPICE export -- Cauer is the canonical export form).

10. The delta_n definition is one CHOICE.  Other proxies
    (highest pole of Z, last-stage R_n/L_n) oscillate with n;
    Foster cut-off converges.  State the definition explicitly.
"""

# ---------------------------------------------------------------------------
# CITATIONS
# ---------------------------------------------------------------------------

CITATIONS = """
# Citations (BibTeX-ish)

## Hiruma method

@article{Hiruma2022CLNerror,
  author  = {Shingo Hiruma and others},
  title   = {Error estimator for Cauer ladder network representation},
  journal = {IEEE Transactions on Magnetics},
  volume  = {58}, number = {9}, year = {2022}, pages = {1--4},
  note    = {Art no. 7500904},
  doi     = {10.1109/TMAG.2022.3165943}
}

## Verified error theory

@inproceedings{Nagamine2022CEFC,
  author    = {Hideaki Nagamine and others},
  title     = {Convergence analysis of the Cauer ladder network method
               using eigenfunction expansion},
  booktitle = {Proc. 2022 IEEE 20th Biennial CEFC},
  year      = {2022}
}

## Magnetic-sheet Cauer circuit (termination and error-theory test problem)

@article{Shindo2017COMPEL,
  author  = {Yuji Shindo and Akihisa Kameari and Tetsuji Matsuo},
  title   = {Efficient circuit representation of eddy-current fields},
  journal = {COMPEL},
  year    = {2017},
  doi     = {10.1108/COMPEL-02-2017-0084},
  note    = {Cauer ladder for magnetic sheet (E and H mode) from
             continued fraction of tanh(kd/2) AND from Legendre-
             polynomial expansion of H_y}
}

## Base CLN paper (parent module: cln_knowledge.py)

@article{Kameari2018CLN,
  author  = {Akihisa Kameari and S. Ebrahimi and Ken Sugahara and
             Yuji Shindo and Tetsuji Matsuo},
  title   = {Cauer ladder network representation of eddy-current fields
             for model order reduction using finite-element method},
  journal = {IEEE Transactions on Magnetics},
  volume  = {54}, number = {3}, year = {2018}, pages = {1--4},
  note    = {Art no. 7201804},
  doi     = {10.1109/TMAG.2017.2743224}
}
"""

# ---------------------------------------------------------------------------
# SERIES EXPANSION
# ---------------------------------------------------------------------------

SERIES_EXPANSION = """
# Series (Neumann) expansion and the Kameari basis

The Kameari method can be derived as a Gram-Schmidt process applied to the
Neumann series of the quasi-static operator (checked on the round wire and
the infinite plate).

Split the quasi-static operator into the static part L and the induction part s M:
    (L + s M) x = f   ->   x = (1 + s L^-1 M)^-1 L^-1 f = sum_k (-s L^-1 M)^k x_static.
Each term costs one STATIC solve.  For the round wire the terms reproduce the Bessel series of
I0(sqrt(z) r)/I0(sqrt(z)) term by term (z = mu sigma a^2 s); for the plate, cosh(sqrt(z) x)/cosh(sqrt(z)).
The power-series basis is not orthogonal, which makes it awkward to use directly.  Orthogonalising it
(Gram-Schmidt in the Joule / magnetic-energy inner products) leaves the span unchanged (it is the Krylov
space) and gives the Kameari basis; the ladder elements are the continued-fraction coefficients of the
same moment sequence:
    wire  R_2k/R_dc = 2k+1, L_2k+1/(tau R_dc) = 1/(8(k+1))
          (L_1 = mu/(8 pi), tau = mu sigma a^2, R_dc = 1/(pi sigma a^2);
           element values 1, 1/8, 3, 1/16, 5, 1/24, 7)
    plate R_2k/R_dc = 4k+1, L_2k+1/(tau R_dc) = 1/(4k+3)      (Lambert's continued fraction of sqrt(z) coth sqrt(z))
With an expansion point s0 the static operator becomes L + s0 M; each Taylor term at s0 is one shifted
static solve (L + s0 M) e_k = -M e_{k-1}.  This is the expansion-point recursion
(docs/expansion_point_validation.md).
Kameari's isolated-system condition (no energy leaves the domain except through the port) is what makes
the orthogonalisation collapse to the two-term ladder recursion.
Executable check: mathematica/series_expansion/series_expansion.wls (11 PASS).
"""



_TOPICS = {
    "overview": OVERVIEW,
    "termination_theory": TERMINATION_THEORY,
    "hiruma_method": HIRUMA_METHOD,
    "bem_fem_coupled": BEM_FEM_COUPLED,
    "nagamine_error_theory": NAGAMINE_ERROR_THEORY,
    "series_expansion": SERIES_EXPANSION,
    "lab_lessons": LAB_LESSONS,
    "citations": CITATIONS,
    "all": "",
}
_TOPICS["all"] = "\n\n".join(v for k, v in _TOPICS.items() if k != "all")


def get_cln_specialty_documentation(topic: str = "all") -> str:
    """Dispatch by topic.

    Topics:
      "all"                    - all sections concatenated
      "overview"               - big picture across the 4 specialty methods
      "termination_theory"     - why terminate, what to terminate with
      "hiruma_method"          - Hiruma's algebraic Lanczos-type recursion
      "bem_fem_coupled"        - BEM exterior + FEM interior coupled CLN
      "nagamine_error_theory"  - verified arithmetic, characteristic
                                 frequency / length, delta_n >= 10 Delta_x rule
      "series_expansion"       - Neumann-series view; Kameari basis = orthogonalised series
      "lab_lessons"            - 10 cross-cutting lessons
      "citations"              - BibTeX-ish references
    """
    topic = topic.lower().strip()
    if topic in _TOPICS:
        return _TOPICS[topic]
    raise ValueError("Unknown topic '" + topic + "'. Available: "
            + ", ".join(sorted(_TOPICS.keys())))
