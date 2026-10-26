"""
Cauer Ladder Network (CLN) — an MOR method for eddy-current FEM
problems: overview, recursion, expansion points, applications.
"""

CLN_OVERVIEW = """
# Cauer Ladder Network (CLN) — MOR for eddy-current FEM

The Cauer Ladder Network method (Kameari-Ebrahimi-Sugahara-Shindo-Matsuo
2018) is a Model Order Reduction technique that replaces a large eddy-
current FEM system by an equivalent electric LADDER NETWORK with a
small number of stages, while preserving the impedance response.

## The big picture

```
Eddy-current FEM (millions of DoFs)
        ↓ (MOR via CLN)
Cauer ladder circuit (N stages, e.g. N=10-60)
        ↓ (can be solved instantly for any input signal)
Time-domain response (transient simulation in real time)
```

A finite-element transient eddy-current simulation that takes ~1-7 HOURS
for a single frequency point becomes a Cauer network simulation that
takes ~MILLISECONDS for any input waveform.

This is essential for:
- Real-time control of converters/inverters
- Online power-loss calculation
- Hardware-in-the-loop simulation of electrical machines
- System-level co-simulation (SPICE coupling)

## The Sugahara Lab's role

Sugahara is a primary author/co-author on the foundational CLN papers:
- 2018a: Kameari-Ebrahimi-**Sugahara**-Shindo-Matsuo
  "Cauer Ladder Network Representation of Eddy-Current Fields for
   Model Order Reduction Using Finite-Element Method"
  IEEE Trans. Mag. 54(3):7201804
- 2018b: **Sugahara**-Kameari-Ebrahimi-Shindo-Matsuo
  "Finite-Element Analysis of Unbounded Eddy-Current Problems Using
   Cauer Ladder Network Method"
  IEEE Trans. Mag. 54(3):7200704
- 2018c: Matsuo-Kameari-**Sugahara**-Shindo
  "Matrix Formulation of the Cauer Ladder Network Method for
   Efficient Eddy-Current Analysis"
  IEEE Trans. Mag. 54(11):7205805
- 2019: Kuriyama-Kameari-Ebrahimi-Fujiwara-**Sugahara**-Shindo-Matsuo
  "Cauer Ladder Network With Multiple Expansion Points for Efficient
   Model Order Reduction of Eddy-Current Field"
  IEEE Trans. Mag. 55(6):7203404
- 2020: Ebrahimi-**Sugahara**-Matsuo-Kaimori-Kameari
  "Modal Decomposition of 3-D Quasi-Static Maxwell Equations by
   Cauer Ladder Network Representation"
  IEEE Trans. Mag. 56(3):7513004

## Connection to PRIMA / Lanczos

PRIMA / Lanczos reduction of PEEC circuits and CLN are sibling MOR
methods (both use Krylov/Cauer expansion).  The CLN framework is the
natural extension of PEEC-circuit Krylov reduction to FEM eddy-current
problems.
"""

CLN_BASIC_RECURSION = """
# Basic CLN recursion (A-formulation)

## Setup

Eddy current problem in time-harmonic form (1 conductor + air):
    curl(ν curl A) + jω σ A = j_0   in Ω
with appropriate boundary conditions.

Space-time separation:
    E(t, x) = Σ_n v_n(t) E_n(x)
    H(t, x) = Σ_n i_n(t) H_n(x)

where {E_n} and {H_n} are orthogonal basis fields:
    ∫_{Ω_c} σ E_n · E_m dx = 0   for m ≠ n
    ∫_Ω    μ H_n · H_m dx = 0   for m ≠ n

The time coefficients v_n(t), i_n(t) are the node voltages and
branch currents of a CAUER LADDER CIRCUIT (next section).

## Recursive computation of basis fields

Starting from E_{-1} = H_{-1} = 0 and L_{-1} = 0:

For n = 0, 1, 2, ...:
    curl(E_n − E_{n−1}) = −(L_{n−1})⁻¹ B_{n−1}    (electric step)
    curl(H_n − H_{n−1}) = R_n J_n                 (magnetic step)

    R_n = [∫_Ω σ E_n · E_n dx]⁻¹
    L_n = ∫_Ω μ H_n · H_n dx

Each step is a STATIC FEM problem (no time dependence) → solvable
fast with standard ICCG/PARDISO.

## The resulting Cauer circuit

    vT(t) ──[R_0]──┬──[R_1]──┬─ ... ──[R_n]──
                   |          |
                  [L_0]      [L_1]    [L_n]
                   |          |          |
                   └──────────┴─────────┘

Total impedance Z(jω) computed via continued fraction:
    Z(jω) = R_0 + 1 / (jωL_0 + 1/(R_1 + 1/(jωL_1 + ...)))

After truncating at stage N, the impedance has the EXACT eddy-current
behavior at expansion point + asymptotic correctness.

## Convergence properties

- Resistance R: monotonically convergent
- Inductance L: monotonically convergent
- For N ≥ 30-60, error < 1% over the entire eddy-current frequency
  range (kHz to MHz for typical industrial inductors)

## Where to truncate (Köster-König-Birò 2021)

Use the IMPEDANCE INCREMENT as stopping criterion:
    |Z_n(f) − Z_{n−1}(f)| / |Z_n(f)| < tolerance

This does NOT need a reference FEM impedance value — purely
internal convergence diagnostic.
"""

CLN_MULTIPLE_EXPANSION = """
# Multiple expansion points (Kuriyama et al 2019)

## Limitation of single-expansion CLN

The basic CLN expands around frequency f_0 = 0 (DC limit).  Near f_0,
convergence is fast (few stages).  Far from f_0 (e.g. at hundreds of
kHz for a problem expanded at DC), convergence is slow — may need
N > 100 stages.

## The fix: change expansion point at stage M+1

Instead of always expanding around f_0 = 0, change to expansion point
f_1 (e.g. f_1 = the target operating frequency) at the M+1-th stage:

    Stages 0 to M:    expand around f_0 (typically 0)
    Stages M+1 to N:  expand around f_1 (operating frequency)

The orthogonality conditions need extra terms p_{2i-1} to maintain
the basis structure:
    p_{2i-1} = -1/λ_{2i-1} · a_{2M-1}^T K_0 a_{2i-1}

And one additional inductor L* appears in the circuit:
    L* = -λ_{2M-1} / [(s_1 - s_0) λ_{2M-2} (λ_{2M-1} + p_{2M-3}λ_{2M-3})]

## Results (Kuriyama 2019)

For a wireless power transfer (WPT) system at 85 kHz operating
frequency:
- Single expansion at f_0 = 0: not accurate even after 100 stages
- Single expansion at f_0 = 85 kHz: accurate around 85 kHz with few stages
- Two expansion points (0, 85 kHz), change at stage 2: ACCURATE from
  DC to 1 MHz with only 6 stages

Speedup vs. direct FEM:
- FEM: 31 min for 13 frequency points
- CLN (6 stages): 2.5 min to generate, then trivial cost per frequency

## When to use multiple expansion points

- The signal/load has a known nominal operating frequency
- The impedance Z(f) must be accurate over a wide frequency range
- Computational efficiency matters (real-time / HIL simulation)

## A vs T formulations

Same recursion structure can be applied to:
- A-formulation: vector potential, conductive + non-conductive domain
- T-formulation: current vector potential, conductive domain only

T-formulation gives a simpler circuit but limited applicability
(needs to be extended to T-Ω for general use, per Kuriyama 2019
conclusion).
"""

CLN_NONLINEAR = """
# CLN for nonlinear materials (Sato-Shimotani-Igarashi 2017)

The basic CLN assumes linear materials (μ const, σ const).  For
ferromagnetic media with B-H saturation, the recursion (5) must be
adapted.

## Approach: linearize around an operating point

For a given excitation level, identify the operating point in
B-H space.  Use the LOCAL slope μ_local = dB/dH at that point.
Build the CLN with μ_local.

## Limitations

- Each operating point requires its own CLN
- For varying excitation, multiple CLNs need to be combined
- Hysteresis is not naturally captured

## Extension to hysteresis (Sato-Clemens-Igarashi 2016)

The "adaptive subdomain MOR with discrete empirical interpolation"
method handles nonlinear magneto-quasi-static problems.  Idea:
identify the saturated regions adaptively and use POD / discrete
empirical interpolation on each subdomain.

This is more general than pure CLN but loses the closed-form ladder
circuit interpretation.

## Connection to the Hantila polarization method

The Hantila polarization method for nonlinear magnetic materials
also splits the constitutive
relation into linear + residual parts and factors the linear
operator once.  This is the same "factor once, iterate on residual"
philosophy as CLN.

A natural future direction: combine CLN-style modal expansion with
Hantila-style nonlinear residual iteration.  This would give a
NONLINEAR CLN that handles full B-H curves without ad-hoc local
linearization.
"""

CLN_APPLICATIONS = """
# CLN applications

## 1. Industrial inductors (Köster-König-Birò 2021)

Geometry: 2 cylindrical ferrite cores with 12 air gaps, aluminum
windings, height 260 mm.  Operating frequency 1 kHz - 1 MHz.

Direct FEM: 1-7 hours per frequency point.
CLN (N=60 stages, mesh m3.2): ~5 min generation, then trivial.
Accuracy: ΔR < 3%, ΔL < 0.3% at 750 kHz.

Different meshes are optimal for different goals:
- Mesh for IMPEDANCE: coarse, just enough to capture conductor topology
- Mesh for FIELD reconstruction: must resolve skin depth at the
  target frequencies

A "winding mesh" (fine in conductor, coarse outside) is the best
trade-off in 2021 author's experience.

## 2. Wireless Power Transfer (Kuriyama 2019)

Two coils (primary 10 turns, secondary open) at 85 kHz operating
frequency.  Multi-expansion-point CLN with f_0=0 and f_1=85 kHz
gives accurate impedance from DC to 1 MHz with only 6 stages.

Useful for system-level WPT design where the magnetic system
appears as a circuit element in a larger SPICE-like simulation.

## 3. Hybrid Twin (digital twin with CLN backend)

Recent work (~2022-2024) combines CLN with machine-learning-
enhanced reduced order models for magnetic bearings.  The CLN
provides the physics-based skeleton; ML corrects the residual.

## 4. TEAM 28 levitation force (reproduction of a published result)

An NGSolve reproduction of the PUBLISHED CLN-on-TEAM-28 levitation (NOT a
first -- see prior art below), compared with the PUBLISHED measured
levitation height.

Setup (axisymmetric): Al disk R=65mm, t=3mm, sigma=3.4e7; two counter-wound
coils (960 turns / +20 A at r=41mm, 576 turns / -20 A at r=87.5mm) at 50 Hz.
The coil-driven eddy problem is `(K + s*N) X = F` (K = s-independent
magnetostatic mixed phi-B operator, N = conductivity term `v*sigma*u/r`,
F = coil source `v*Jz`).  The CLN / Cauer reduction is the Krylov subspace
generated from the COIL SOURCE:
    V_0     = K^{-1} F
    V_{k+1} = orthonormalise( K^{-1} (N V_k) )
and the levitation Lorentz force is evaluated from the N-stage reduced field.

Result: a 6-stage CLN reproduces the full-FEM levitation force vs height to
max |CLN-full| = 5e-4 N (stage 1 = 97.8% err DC, stage 3 = 0.14%, stage 5 =
0.000%).  The PHYSICAL levitation equilibrium (lift == disk weight 1.055 N,
using F_z/2 -- the verbatim TEAM 28 surface integral Re[B_r J_t] is exactly 2x
the physical time-averaged force) lands at absolute disk-bottom z = 11.0 mm,
matching the PUBLISHED measured steady-state levitation height z = 11.5 mm
(Karl-Fetzer-Kurz-Lehner-Rucker, official TEAM 28 definition) to 4%.
(Pitfall: balancing the 2x integral against the 1x weight gives a spurious
14.9 mm; comparing with the published 11.5 mm exposes it.)

Prior art (this REPRODUCES it, NOT a first): K. Sugahara, N. Tanimoto,
Y. Takahashi, T. Matsuo, "Cauer Ladder Network Representation with Constant
Basis Functions for Eddy Current Problems Involving Conductor Movement",
COMPUMAG 2023 (Paper ID 324) -- the full motion-coupled transient levitation
height z(t) vs measurement + the conventional method, 4-stage CLN, ~7 s vs
~8 h; it introduced the constant-basis `A_s(zgap) = sum_n a_2n i_2n` expansion.

## 5. Connection to PEEC

PEEC = Partial Element Equivalent Circuit (Radia/Loop-Star method).
Both PEEC and CLN produce circuit-level abstractions of EM problems,
but they differ in:
- PEEC: spatially-distributed circuit elements derived from geometry
  (no MOR initially; MOR via PRIMA-Lanczos comes later)
- CLN: time-evolution-modal expansion, lumped circuit elements

For LARGE problems (e.g. industrial-scale induction heating),
combining PEEC-coil + CLN-workpiece can give the best of both.

## Implementation in this repository

- `cln` (src/cln/): the CLN method on NGSolve -- T-Omega ladder
  (src/cln/t_omega.py), ladder evaluation (src/cln/ladder.py) and
  SPICE / LTI export (src/cln/spice.py, src/cln/export.py)
- Large H(curl) systems in the basis-field computation need an
  HX/AMS-type or ICCG preconditioner
"""


CLN_PGD_POSITIONING = """
# CLN's positioning within the broader MOR family: CLN as PGD

## One-line claim (Köster-König-Bíró 2021, IEEE TMag 57(6))

The Cauer Ladder Network method is a **specific instance of Proper
Generalized Decomposition (PGD)** — a separated-variable a-priori
model order reduction family — in which the temporal-factor recursion
is replaced by the Cauer ladder network equations.

## Why this matters

When discussing CLN in cross-community settings (MOR theorists,
POD / Reduced Basis / PGD community), the natural question is:
"How does CLN relate to the established MOR families?"  Köster et
al. 2021 answer this explicitly: CLN ⊂ PGD.  This is not a weakening
of CLN — it is a **proper positioning** that grants CLN the
generality theorems of PGD (a-priori convergence, separated-variable
structure, enrichment-based extension) while preserving its
distinctive Cauer-circuit realization.

## The exact mathematical statement (Köster 2021 §III.A)

PGD-class MOR methods are defined by separated representations
    u(x_1, ..., x_M) ≈ Σ_{i=1}^{Q} F_i^1(x_1) ··· F_i^M(x_M),
with orthogonal function sets {F_i^k} and recursions for each
factor F_i^k (the factor recursions may be coupled).

CLN realizes this with M=2, variables (t, x), as
    u(t, x) ≈ Σ_{i=1}^{N} T_i(t) X_i(x),
where:
  - X_i(x) are the static FE basis fields (recursively computed
    from K_0 + sources)
  - T_i(t) are the temporal factors

The crucial CLN-specific twist: instead of solving a recursion for
T_i(t) in the time domain directly, the network of coupled ODEs for
{T_i} is **realized as a Cauer ladder circuit** whose passive
elements {L_n, R_n} are the integrals of X_i against the basis
field operators.

The X_i recursion remains classical PGD enrichment (greedy in
modal hierarchy); the T_i "recursion" is the Cauer circuit, which
is therefore reading position #i in a longer (in principle infinite)
ladder.  This **PGD-in-space + Cauer-circuit-in-time** split is
what makes CLN distinctive in the PGD family.

## Implications for a sqrt(s) Schur augmentation

1. **A single-DOF √s Schur augmentation is base-agnostic.**
   Its uniqueness argument depends only on the rational nature of
   the base ROM (its $Y_R \\in \\mathbb{Q}(s)$ tail), not on whether
   the base was constructed as canonical Cauer recursion (single
   expansion point at $s\\!=\\!0$), Kuriyama multi-expansion-point
   Krylov, or general PGD.  The Schur block z(s) = (s+d)/(K_SIBC √s)
   is the structurally minimal asymptote-preserving augmentation
   regardless of which rational base sits above.

2. **The PGD framing legitimises the multi-expansion-point and
   POD-style extensions.**  Kuriyama's 2019 Multi-K is in PGD
   language a *multi-snapshot* PGD with snapshots at multiple
   expansion frequencies, $K_0$-MGS orthogonalised.  General POD
   (Sirovich-Berkooz-Holmes-Lumley snapshots) is a sibling.  All
   three (canonical CLN, Multi-K, POD) are PGD instances at
   different levels of basis enrichment, and the augmentation
   argument applies to all of them.

3. **Cauer-CLN naming convention is retained for continuity with the
   eddy-current literature.**  Köster's PGD positioning does NOT
   require renaming CLN to "PGD-with-Cauer".  The eddy-current CLN
   literature (Kameari 2018, Kuriyama 2019, Hiruma 2020) uses the
   CLN/Cauer terminology; the PGD relation needs one sentence and
   one citation.

## Citation (canonical bibtex)

    @article{Koester2021PGDCLN,
      author={K{\\"o}ster, Niels and K{\\"o}nig, Oliver and
              B{\\'i}r{\\'o}, Oszk{\\'a}r},
      journal={IEEE Trans. Magn.},
      title={Proper Generalized Decomposition With {Cauer} Ladder
             Network Applied to Eddy Current Problems},
      year={2021},
      volume={57},
      number={6},
      pages={Art.\\\\ no.\\\\ 6300904},
      doi={10.1109/TMAG.2021.3059800}
    }

## Related papers

The same author group (Köster, König, Bíró, TU Graz) has further
CLN-extension work on scalar-potential CLN, stranded wires via
homogenization + CLN, and FE implementation of the iterative scalar
potential method; these build on Köster 2021.

## Rule of thumb for AI sessions

When discussing CLN in the MOR / POD / PGD community context, cite
Köster 2021 once to position CLN ⊂ PGD.  Do not over-cite — the
PGD positioning is well-established and a single reference suffices.
For CLN's distinctive Cauer-circuit realization, cite Kameari 2018.
For multi-expansion-point variants, cite Kuriyama 2019.
"""


CLN_SURFACE_ENRICHMENT = """
# Surface enrichment: CLN bulk and a surface basis

## The defect this fixes

A finite Cauer ladder is a RATIONAL function of s. Rational functions have
INTEGER asymptotic slopes. The eddy-current admittance of a conductor decays as
f^(-1/2) at deep skin, so no ladder of any length produces the right tail. This
is structural, not a shortage of rungs, and adding expansion points (Kuriyama
2019) does not help either: every finite expansion point yields another rational
basis, so the extra points relocate poles and the tail still wags.

## Why s = infinity is different

On the negative real axis the bulk contributes POLES at s = -lambda_n (the
diffusion eigenvalues, accumulating toward -infinity). The square-root tail is
not a pole at all. 1/sqrt(s) has a BRANCH POINT: go once around s = 0 and
sqrt(s) returns with the opposite sign, so no Laurent expansion exists and no
integer m makes s^m/sqrt(s) finite and nonzero. In Stieltjes form

    1/sqrt(s) = (1/pi) * integral_0^inf  xi^(-1/2) / (s + xi)  d xi

which is a CONTINUUM of first-order poles each carrying infinitesimal weight.
A pole is a point with finite residue; the cut is a density with none. A finite
sum of point masses cannot equal a continuous density -- that is the whole
argument.

Caveat worth stating before someone asks: a BOUNDED conductor's exact Y(s) is
meromorphic and has only poles. The cut is the half-space (unbounded) model,
used as the asymptotic description of where those poles ACCUMULATE. Bulk basis
resolves the head of the spectrum, surface basis resolves the tail.

## The construction

    v(r, s) ~ sum_k xi^b_k phi_k(r)  +  sum_k xi^s_k psi_k(r, s)

  phi_k : bulk CLN Krylov modes, frequency independent, built once
  psi_k : surface envelopes ~ exp(-d(r)/delta(s)), which carry the sqrt(s)

Read as a two-point Pade in sqrt(s): one expansion point at s = 0 (the ladder),
one at s = infinity (the surface). Multiple expansion points enrich the trial space. The surface contribution
represents the high-frequency asymptote with a frequency-dependent basis;
this alone does not define a mixed finite-element formulation.

Adding a rung adds one term of the Taylor expansion at s = 0. Adding a Senior
curvature term (gamma_1 = -H, gamma_2 = (K - H^2)/2) adds one term of the
asymptotic expansion at s = infinity. Symmetric.

## Schur complement: what it is for

Eliminating the bulk block gives

    S(s) = K_ss(s) - K_sb(s) K_bb(s)^-1 K_bs(s)

The CODE DOES NOT NEED THIS -- the block system is at most 6x6, solve it and
stop. Write it down for three reasons instead:
  1. admittance is a PORT quantity, so writing it explicitly means eliminating
     the interior; S(s) is the discrete Steklov-Poincare (DtN) map and the
     surface-port admittance, three names for one object;
  2. it exhibits the surface amplitude as computed rather than fitted, which is
     what removes the hand-chosen crossover of a Warburg termination;
  3. -K_sb K_bb^-1 K_bs IS Gram-Schmidt in the K inner product, so its size
     measures how much the two families overlap in the finite space.

## Measured, with the metric stated

Metric: abs(Y_exact - Y_enriched) / abs(Y_exact), the COMPLEX difference. Do not
take abs() of each side and then subtract because that discards phase error.
This knowledge text deliberately does not quote error values: report them
only from a run whose metric, frequency band and reference are stated.

## The polyhedron floor needs an independent edge amplitude

The tensor envelope psi = f(x) f(y) f(z) already has the right asymptotics
everywhere: exp(-d t) - 1 on a face, r^2 sin(2 theta) at a 90-degree edge (the
Wiener-Hopf wedge), and x y z ~ r^3 Y_{3,0} at an octant corner. What one
tensor-product DOF cannot do is give the correct quantitative WEIGHT to all
three boundary classes at once. A naive split into one separable DOF per face,
edge, and corner class is asymptotically rank-deficient. Conforming local
edge/corner bumps instead show that an independent edge amplitude removes the
cube floor. A fixed high-frequency edge/corner ratio is
adequate only asymptotically; through the transition band those amplitudes must
remain independent because they couple to the interior modes.

SIBC corner and edge treatments exist independently (Deeley 1990 IEEE TMag
26(2):712; Yuferev-Proekt-Ida 2001 IEEE TMag 37(5):3465) and are not yet
imported into this trial space.

## Time domain

The Senior tower is half-integer powers of s, i.e. fractional integrals, each
realisable as a finite RC ladder. Re-entrant edges give an angle-dependent
non-half-integer exponent instead. The production realization uses a passive
real-pole Foster fit seeded by AAA and completed with diffusive poles and
non-negative least squares. Direct use of the one-sided complex AAA poles is a
negative result, not the time-domain model.

## Reporting rule

Keep one numerical artifact per study and let documentation READ it rather
than recompute it; recomputing in several places is how a metric forks.
"""


CLN_SAME_CIRCUIT = r"""
# A法・T法から同じ回路定数が得られる理由

結論: 確認した丸線では、既存の規格化や境界条件の設定を修正したのではない。
異なる未知量から物理場を復元し、規格化を正しく追うと、同じTypeの回路定数が一致する。
「境界条件の間違いが原因」「規格化を直して一致させた」と説明しない。
境界条件は証明に必要だが、不具合の原因と確認されたわけではない。

## Type2
x=r/a, z=mu*sigma*a^2*s とする。単位電圧勾配の解析場を e,h とし、
Y=2*pi*h(1), W=2*pi*integral(x*h^2)= -dY/dz とする。
単位電流の場 (e,h)/Y の全エネルギー積分は1/Y。
その値で割る既存の規格化により、単位電圧の場(e,h)と一致する。
R0=1/Y, L1=W/Y^2, c=R0/L1=Y/W とすると、
E2=e+c*de/dz, H2=h+c*dh/dz は E2(1)=1 と H2(1)=0 を同時に満たす。
したがって、この場合A法DirichletとT法Dirichletの組合せを変える必要はない。
A法の補正未知量とT法の今段の磁界は異なるため、SolveStageの見た目は異なる。
復元後の場が一致するので、エネルギー積分から得る R0,L1,R2,L3 も一致する。

## Type1
L[-1]=1/(zY), E0=e, R0=1/P(e), P(e)=2*pi*integral(x*e^2)。
共通の補正場は E1=R0*z*de/dz, H1=R0*(h+z*dh/dz)。
A法: -laplacian(u)+z*u=R0*e, u(1)=0; E1=-z*u, H1=-u'。
T法: H1=t, E1=(x*t)'/x-R0*e; 自然境界条件E1(1)=0。
印加項R0*eを落とさずに復元すると同じ場となり、L[-1],R0,L1,R2が一致する。
Type1とType2の要素値そのものが同じという意味ではない。

## 検証範囲と証拠
均質丸線・軸方向不変・正の実数展開点・各Typeの4要素。
Type2: Wolframの記号チェック12件、数値照合4条件、独立Opusレビュー済み。
Type1: 記号チェック13件、数値照合4条件。A/T別のNGSolve有限要素計算で収束確認。
最細メッシュの要素相対誤差はA約2.1e-6、T約1.1e-6、相互差約3.2e-6。
radia-ngsolveのPython環境でも同じスクリプトを再実行した。
これは別環境での再現であり、独立した別実装と呼ばない。
Type1の新規Opusレビューは未実施。任意段数・一般3D・複素展開点・s0=0は未証明。
同じ端子伝達関数だけを根拠に、内部場が同じだと断定してはいけない。
有限の連分数は展開点付近の近似であり、全周波数で厳密Zと一致するわけではない。

公開証拠:
https://github.com/ksugahar/CLN-mcp/blob/main/docs/type2_same_circuit.ipynb
https://github.com/ksugahar/CLN-mcp/blob/main/docs/type1_same_circuit.ipynb
"""


def get_cln_documentation(topic: str = "all") -> str:
    """Dispatch by topic.

    Topics:
      "all"
      "same_circuit"     - Verified A/T field reconstruction and normalization
      "overview"         - CLN big picture, Sugahara Lab's role
      "recursion"        - Basic A-formulation recursion + Cauer circuit
      "multiple"         - Multiple expansion points (Kuriyama 2019)
      "nonlinear"        - Extension to nonlinear ferromagnetic materials
      "applications"     - Industrial inductors, WPT, hybrid twin
      "pgd_positioning"  - CLN as a PGD instance (Köster 2021), and
                           implications for base-agnostic Theorem 1
      "surface_enrichment" - The ladder cannot make the f^-1/2 tail; bulk CLN
                           plus an s-dependent surface basis, Schur complement,
                           and the measured errors with their metric
    """
    topic = topic.lower().strip()
    if topic in ("same_circuit", "a_t_equivalence", "normalization"):
        return CLN_SAME_CIRCUIT
    if topic in ("overview", "intro"):
        return CLN_OVERVIEW
    if topic in ("recursion", "basic", "a_formulation"):
        return CLN_BASIC_RECURSION
    if topic in ("multiple", "multi_expansion", "expansion_points"):
        return CLN_MULTIPLE_EXPANSION
    if topic in ("nonlinear", "hysteresis", "saturation"):
        return CLN_NONLINEAR
    if topic in ("applications", "examples", "inductor", "wpt"):
        return CLN_APPLICATIONS
    if topic in ("pgd_positioning", "pgd", "koester", "köster",
                 "positioning", "mor_family"):
        return CLN_PGD_POSITIONING
    if topic == "surface_enrichment":
        return CLN_SURFACE_ENRICHMENT
    if topic == "all":
        return "\n\n".join([
            CLN_OVERVIEW, CLN_BASIC_RECURSION, CLN_MULTIPLE_EXPANSION,
            CLN_NONLINEAR, CLN_APPLICATIONS, CLN_PGD_POSITIONING,
            CLN_SURFACE_ENRICHMENT, CLN_SAME_CIRCUIT,
        ])
    raise ValueError(
        f"Unknown topic '{topic}'. Available: "
        "all, overview, recursion, multiple, nonlinear, applications, "
        "pgd_positioning, surface_enrichment, same_circuit."
    )
