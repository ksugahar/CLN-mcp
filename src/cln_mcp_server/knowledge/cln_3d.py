"""3D Cauer Ladder Network (CLN) knowledge base.

Captures Tanimoto-Kameari iterative CLN methods for 3D eddy current
analysis on NGSolve: A-T, T-Omega and A-Phi formulations, a 2D
reference, gauge/penalty variants, and solver variants.

Each formulation produces a Cauer-II ladder {R_n, L_n} via Kameari's
iterative orthogonalization on impressed J source.
"""

CLN_3D_OVERVIEW = """
# 3D Cauer Ladder Network (CLN) — Tanimoto-Kameari Methods

## Mathematical Foundation

The Cauer Ladder Network reduces a 3D eddy current problem to a series
of (R_n, L_n) circuit elements via iterative orthogonalization in the
σ-weighted L^2 inner product on the conductor:

  R_n = 1 / ⟨J_n, J_n/σ⟩_cond
  L_n = R_n × ⟨J_n, A_pot_n⟩_cond     (accumulated potential)
  J_{n+1} = J_n - σ·A_pot_n / L_n     (Schmidt orthogonalization)

The first Cauer admittance Y_zz(s) = -μ_0·s·Σ a_n/(1+sτ_n) is
reconstructed from the ladder; Foster modes τ_n correspond to physical
eddy current decay times.

## Three Equivalent Formulations (equal in physics, differ in FE space)

### (1) A-T Formulation — Most Common
Two HCurl spaces:
  fesA = HCurl(mesh, order, nograds=True, dirichlet="<conductor BCs>")
  fesT = HCurl(mesh, order, nograds=True)         # auxiliary T field

Curl-curl in conductor with boundary E source:
  a(T,W) = ∫_cond (1/σ) curl T · curl W dx
  f(W) = -∫_∂cond (E_s × n) · W ds                # boundary E flux
  J = curl T          # current via curl of auxiliary

Then iterate Kameari on (R_n, L_n) using J and A:
  a(A,N) = ∫ (1/μ) curl A · curl N dx
  f(N) = ∫ R·J · N dx in conductor

### (2) T-Ω Formulation
Hybrid HCurl × H1(conductor-only):
  fesT = HCurl(mesh, order, nograds=True)
  fesΩ = H1(mesh, order, definedon=mesh.Materials("conductor"))

Scalar potential Ω confines current source to conductor; coupled
solve for (T, Ω) with Lagrange multiplier or coupled bilinear form.

Distinctive feature: Ω is conductor-only, reducing global DoF.  The
conductor-only T-Omega CLN with an expansion point is implemented in
this repository as cln.TOmegaCLN (src/cln/t_omega.py).

### (3) A-Φ Formulation
HCurl + H1 with Φ in conductor:
  fesA = HCurl(mesh, order, dirichlet=...)
  fesΦ = H1(mesh, order, definedon="conductor",
            dirichlet="in|out")

Body current J = σ∇Φ (implicitly through weak form).  A higher-order
variant (order=3) uses the mixed space `V = HCurl × H1`.

## Iterative Algorithm (Common to All Formulations)

```python
# Initial impressed J_0 from boundary E or applied A_ext
J = sigma * A_ext         # or: J from boundary integral
Apot = 0
for n in range(N_STAGES):
    # Solve curl-curl A_n = J_n with chosen formulation
    solve(a, f_with_J_source, gfA_n)
    # R_n = 1 / <J, J/sigma>
    R_n = 1 / Integrate(J*J/sigma * dx, mesh)
    Apot += R_n * gfA_n
    # L_n = R_n × <J, Apot>
    L_n = R_n * Integrate(J * Apot * dx("conductor"), mesh)
    # Schmidt-orthogonalize J for next stage
    J = J - sigma * Apot / L_n
```

## Validation Cases (Cylinder, r×h)

For a cylindrical conductor with TM modes, analytical formulas exist:
  R_theory[n] = (2n+1) / (π·r²·σ·h)
  L_theory[n] = μ_0·(2n²+2n+1) / (8·(n+1)·π·h)

(Bessel-mode decomposition.)

## Constraint/Gauge Variants

### A-Penalty
Add small penalty term to stabilize near-singular curl-curl:
  a_penalty(A, N) = (1/μ) curl A · curl N + ε/μ A · N    # ε ~ 1e-6
Avoids explicit gauge correction; loses ∇·A=0 exactly but in practice
gives same Cauer ladder values within tolerance.

### A-Gauge (Coulomb explicit)
Two-step:
  1. Solve A via HCurl(nograds=True): gives A with arbitrary gradient
  2. Project to Coulomb gauge: solve H1 problem ∇²φ = ∇·A, then A := A - ∇φ
Maintains ∇·A = 0 to machine precision but adds H1 solve per stage.

## Solver Variants

| Solver | Distinctive |
|---|---|
| Direct sparse | Reference baseline |
| NGSolve CG + local preconditioner | Standard Krylov |
| ICCG (shifted incomplete Cholesky) | Robust for the semidefinite curl-curl |
| accelerated ICCG | Acceleration-parameter tuning |

Recommendation: A + ICCG with an inline gauge correction.

## Common Boilerplate

```python
from netgen.occ import Cylinder, Z, OCCGeometry, Pnt
from ngsolve import (Mesh, HCurl, H1, BilinearForm, LinearForm,
                     GridFunction, CoefficientFunction, curl, dx, ds,
                     Integrate, TaskManager, Cross, specialcf)
import scipy.sparse as sp

# Geometry
cyl = Cylinder((0,0,0), Z, r=0.01, h=0.01)
cyl.faces.Max(Z).name = "out"
cyl.faces.Min(Z).name = "in"
cyl.faces.name = "conductorBND"
cyl.mat("sig")
cyl.maxh = 0.001
mesh = Mesh(OCCGeometry(cyl).GenerateMesh(maxh=0.001)).Curve(3)

sigma = 1e6   # or 5.8e7 for Cu
mu = 4*pi*1e-7

# Per-stage solve (A-T form)
fesA = HCurl(mesh, order=1, nograds=True, dirichlet="in|out|conductorBND")
fesT = HCurl(mesh, order=1, nograds=True)
A, N = fesA.TnT()
T, W = fesT.TnT()

Es = (0, 0, 1)  # impressed E along z
n = specialcf.normal(mesh.dim)
a_T = BilinearForm(fesT)
a_T += 1/sigma * curl(T) * curl(W) * dx
f_T = LinearForm(fesT)
f_T += -Cross(Es, W.Trace()) * n * ds("conductorBND")
```
"""


CLN_3D_NOTEBOOK_INDEX = """
## Example scripts shipped with this package (cln_notebook tool; by N. Tanimoto)

- AT          : A-T formulation, primary 3D Kameari (10-stage)
- T_Omega     : T-Ω with H1 confined to conductor
- APhi        : A-Φ with HCurl + H1(conductor)
- 2D          : 2D scalar reference for validation
- production  : A + ICCG with inline gauge correction

Variants worth reproducing when testing a new implementation:
penalty stabilization (single stage), explicit Coulomb gauge (4 stages),
gauge-fix diagnostics, nograds=True without penalty, penalty 1e-6,
direct vs CG vs ICCG back-ends, order / stage / type-1 sweeps, a
TEAM-28-size mesh, a mixed HCurl×H1 order-3 A-Φ, and the T-from-A
(curl T = A) reconstruction check.

## CLN + Kelvin transformation

Kelvin transformation in NGSolve is an accurate open-boundary technique
when properly formulated (accuracy independent of the Kelvin radius
with adaptive refinement).

**Canonical reductions**:

| Method      | Inner          | Outer (Kelvin)          | Use case         |
|-------------|----------------|-------------------------|------------------|
| Omega-Omega_r | H=-grad(Omega_t) | H=-grad(Omega_r)+H_s | Linear magnetic |
| A-Omega_r   | B=curl(A_t)    | H=-grad(Omega_r)+H_s    | Recommended      |
| A-A_r       | B=curl(A_t)    | B=curl(A_r+A_s)         | Vector source    |
| A-phi-A_r   | A,phi in cond  | A_r in Kelvin           | Eddy (TEAM7)     |

**Source injection**: the source field enters via a BOUNDARY INTEGRAL at
the inner-Kelvin interface (dOmega), NOT via a (nu - nu_0) bulk volume
term.  The bulk form is a common pitfall.

**Implication for CLN + Kelvin**:
- A-T or A-Phi: use A-Omega_r boundary-integral coupling, NOT bulk
- T-Omega: structurally cleanest (T in conductor only, Omega_r outside;
  no A_s pullback required in the conductor region). RECOMMENDED.
- A-phi-A_r: appropriate for eddy current with applied A_s from coil

**Recommended path** for applied uniform B + CLN:
- Omega-Omega_r or A-Omega_r with boundary-integral source
- H_s enters at dOmega interface: f += -omega * (B_s.n) * ds("kelvin_int")

**Recommended path** for PEEC coil source + CLN:
- A-formulation with A-A_r (decaying coil field, no singularity issue)
- Direct boundary form, NOT (nu - nu_0) bulk
"""


CLN_3D_KEY_FORMULAS = """
## Key Formulas Quick Reference

### Cauer-II Ladder Synthesis (from Kameari iteration)
After N stages, the admittance Y_Foster(s) = Σ a_n/(1+sτ_n) is
reconstructed from {R_n, L_n} via Cauer-II continued fraction at s=0:
  Y(s) = 1/(s·L_0 + 1/(R_0 + 1/(s·L_1 + 1/(R_1 + ...))))

### Foster Pole Identification
The dominant Foster pole τ_lead corresponds to MAX τ_n where τ_n=L_n/R_n,
not necessarily the first Cauer stage. For closed PEC cuboid 5×2×1:
  Analytical TE_z(1,1,0) τ = μ_0·σ·a²·b² / (π²·(a²+b²))
  Kameari max τ_n over 12 stages = 25.33 μs (vs analytical 25.46 μs,
                                              0.5% match)

### Schmidt Orthogonality Drift Diagnostic
  drift_n = max_{m<n} |⟨J_n, J_m/σ⟩_cond| / ⟨J_n, J_n/σ⟩_cond

For 3D HCurl order=3 with bonus_intorder=8 (closed PEC):
  N ≤ 11: drift ≤ 1e-12 (machine precision)
  N = 12: drift = 3e-12 (exponential growth onset, ~×6/stage)
  N = 25: drift = 5.5% (1% breakdown threshold)
  N ≥ 26: corrupted regime

### CRITICAL LINT: NGSolve GridFunction is mutable — snapshot before
### embedding in CoefficientFunction expressions across iteration stages

In the Kameari accumulation iteration (and any 3D CLN iterative scheme),
the accumulator `Apot_acc` is a GridFunction whose `.vec.data` is updated
each stage. If you build a CoefficientFunction that *references*
Apot_acc (e.g., `A_acc_proj_cf = Apot_acc - grad(gf_phi_acc)`), then store
that CF inside `J_n_cf` (`J_n_cf = J_n_cf - sigma_cf * A_acc_proj_cf / L_n`),
the CF holds a *reference* — not a copy — to Apot_acc. When Apot_acc is
updated next stage, every previous J_n_cf silently re-evaluates with the
new (wrong) accumulator value.

Symptom: stage 0 matches the analytical Cauer-I to 0.027 %,
but stage 1 returns the wrong sign (e.g. τ_1 = −796 μs
when the analytical Stoll χ-Foster gives +154.6 μs), then stages 2+
diverge. H-H projection of A_acc, ORDER_PHI matching, ORDER bumps,
Schmidt orthogonalization — all fail to fix the symptom because they
all keep the mutable-Apot_acc reference alive across stages.

Fix (checked on a Cu sphere a=10mm, B0=1T): snapshot Apot_acc
into a fresh GridFunction *each stage* and reference the snapshot in
the CF expression:

    Apot_acc.vec.data += R_n * gfAphi.vec
    snap_acc = GridFunction(fesA, name=f"snap_acc_{nStage}")
    snap_acc.vec.data = Apot_acc.vec       # explicit copy
    A_acc_proj_cf = snap_acc - grad(gf_phi_acc)   # frozen refs
    J_n_cf = J_n_cf - sigma_cf * A_acc_proj_cf / L_n

After this fix the sphere 3D Kameari + Kelvin gives stage 0/1/2/3 with
−0.03 / −0.09 / −1.0 / −4.1 % gap vs the analytical Stoll Cauer-I ladder
(vs a 7-stage axisym H1 run reaching machine precision).  The FP64
precision wall remains (~k = 4 onwards).

### bonus_intorder Critical Setting
For order ≥ 3 HCurl with quadratic A_ext source, default integration
order is too low. Use `bonus_intorder=8` in all `dx()` calls to keep
Schmidt drift at machine precision through 11+ stages.

### IMPORTANT: Hiruma 3-term Lanczos vs Kameari accumulation —
### they extract DIFFERENT Cauer-I sequences

Hiruma & Igarashi (IEEE TMag 56(3) 2020) and the Kameari accumulation
BOTH produce Cauer-I ladders {R_n, L_n, τ_n} from the same FE matrices
(K, M, b), but they expand DIFFERENT generating functions, so the
resulting τ_n sequences are MATHEMATICALLY DIFFERENT (typically 5–12%
offset in low stages):

- Hiruma 3-term Lanczos: Cauer-I of f_H(s) = bᵀ·(K - sM)⁻¹·b
  (Krylov-Padé impedance representation, Lanczos-style three-term
   recurrence on (K, M))
- Kameari accumulation: Cauer-I of f_K(s) = uᵀ·M·(sM - K)⁻¹·M·u,
  with u = M⁻¹·b   (susceptibility-like representation)

Sphere (Cu a=10 mm, B₀=1 T uniform) comparison vs the analytical Stoll
Bessel Cauer-I (240-digit Hankel-Padé):

  k │ Stoll χ-Foster Cauer-I │ Hiruma 3-term  │ gap
 ───┼───────────────────────┼────────────────┼────────
  0 │      694.142 μs        │   728.85       │ +5.0%
  1 │      154.604 μs        │   171.51       │ +10.9%
  2 │       64.075 μs        │    71.68       │ +11.9%

  k │ Stoll Cauer-I │ Kameari accumulation NGSolve │ gap
 ───┼──────────────┼──────────────────────────────┼────────
  0 │  694.142 μs   │           694.142            │  0.000%
  1 │  154.604 μs   │           154.604            │  0.000%
  2 │   64.075 μs   │            64.075            │ −0.000%
  4 │   21.400 μs   │            21.398            │ −0.014%

Conclusion: Kameari accumulation = exact analytical Cauer-I
(susceptibility). Hiruma 3-term = a different Cauer-I convention
(Krylov-Padé impedance) which is not directly comparable to
Stoll/χ-Foster spectra. To convert Hiruma matrices to the
Kameari/Stoll Cauer-I, use:
  α_n = uᵀ M (K⁻¹M)ⁿ u   with u = M⁻¹·b
then Hankel-Padé continued-fraction extract.

FP64 NGSolve: both methods reach k=6 reliably (precision wall at
k=7). For higher stages, use mpmath ≥60 digits offline.

### CRITICAL: BEM-Foster CLN extracted from g_k² τ_k moments is the
### Hiruma convention, NOT the Kameari convention

When extracting Cauer-I from a BEM-Foster spectrum (eigenmodes v_k of the
boundary integral operator with eigenvalues τ_k), the moment definition
chooses the convention. Two distinct conventions appear in the literature:

  (Hiruma / Krylov-Padé impedance):
    α_n^Hiruma = (-1)^n Σ_k g_k^2 · τ_k^(n+1)
    where g_k = Re(v_k · a_ext) is the Foster amplitude
    matches Hiruma 3-term Lanczos on (K, M, b)

  (Kameari / χ-Foster susceptibility):
    α_n^Kameari = (-1)^n Σ_k a_k · τ_k^(n+1)
    where a_k is the χ-susceptibility amplitude (sums to 1; for sphere
    a_n = 6/(n²π²))
    matches Kameari accumulation iteration

Example: cylinder R=10mm t=2mm Cu disk
  cylinder BEM-Foster (Hiruma conv, axisym moments + Hankel-Padé):
    τ_pair = [219.3, 78.6, 40.0, 23.6, 16.8, 14.0] μs
  cylinder NGSolve axisym Kameari accumulation:
    τ_pair = [208.4, 67.5, 32.6, 20.4, 15.7, 11.3] μs
  cylinder NGSolve 3D Kameari + snapshot fix:
    τ_pair = [208.3, 67.3, 31.8, 19.8, 15.8, 12.9] μs

Cylinder axisym Kameari and 3D Kameari agree to 0.02-3 % (algorithm
verified), both differ from BEM-Hiruma by 5-15 % (convention difference,
not algorithm error). Confusing the two leads to phantom validation
gaps that no h- or p-refinement can close.

**Sphere is special**: Stoll Bessel χ(s) = -1 + Σ(6/(n²π²))/(1+sτ_n) is
a χ-susceptibility (Kameari) representation, so the Hankel-Padé extract
from a_n = 6/(n²π²) IS the Kameari Cauer-I. This is why sphere Kameari
NGSolve matches Stoll to 0.000 % — same convention.

**Lesson**: when validating a 3D Kameari implementation against an
"axisym BEM Cauer reference", first verify the BEM extraction convention.
If the BEM script uses g_k² · τ_k moments, expect a 5-15 % offset that
is structural, not numerical. To get an apples-to-apples Kameari ground
truth, run Kameari accumulation on the axisym mesh.

### POLICY (CRITICAL): always compare in Cauer-rung convention
### never mix Foster-pole values with Cauer-rung values

When comparing CLN-related τ values across methods, always extract
Cauer-I rungs τ_pair[k] = L_{2k+1} / R_{2k} via Hankel-Padé and compare
those. NEVER substitute "Foster pole" τ_n (= leading eigenvalue of the
diffusion operator, or = max|g|²·τ entry of a Foster expansion, or
= leading time-constant of a time-domain exponential fit) in place
of the Cauer rung. They are different physical quantities.

  Foster pole τ_n:    individual eddy-current eigenmode time constant
  Cauer rung τ_pair[k]: ladder-stage time constant in {R_n, L_n} synthesis

They differ systematically:
  Sphere    : Foster τ_1 = 738 μs ≠ Cauer rung τ_pair[0] = 694 μs (-6%)
  Cylinder  : Foster τ_1 = 224 μs ≠ Cauer rung τ_pair[0] = 209 μs (-7%)
  Cuboid    : leading Foster pole ~10.9 μs ≠ Cauer rung ~10 μs
              (analogous -8% offset)

Forbidden:
  - Comparing a volume-integral solver's "leading τ" (a Foster pole)
    directly with a Cauer rung
  - Comparing a time-domain leading exponential decay τ directly with
    a Cauer rung (the exponential is a Foster pole)
  - Comparing BEM-Foster spectrum eigenvalues τ_k directly with a Cauer
    rung (use kameari_cauer_from_bem() below to extract Cauer-I first)

Required:
  - Pass every method's spectrum through Hankel-Padé to get Cauer rungs
  - Use the BEM-to-Kameari converter (kameari_cauer_from_bem below)
    to convert Foster amplitudes {τ_k, g_k²} into Cauer
    rungs in the Kameari (susceptibility) convention
  - When reporting "the leading τ", state explicitly whether it is the
    Foster pole τ_1 or the Cauer rung τ_pair[0]

Failure mode: a cuboid 5×2×1 NGSolve 3D Kameari run with the snapshot
fix gave τ_pair[0] = 9.95 μs (Cauer rung); comparing it with
Foster-pole references (~10.9-11.5 μs) produces a phantom -10..-14 %
gap.  After the -8 % Foster→Cauer offset the 3D Kameari result is
consistent with the Foster-pole references.

### CONVERSION: derive Kameari Cauer-I from BEM-Foster {τ_k, g_k}

The BEM Foster spectrum (eigenvalues τ_k and amplitudes g_k = Re(v_k·a_ext))
is the *same physical data* — just two different Cauer extractions:

  **Hiruma ladder** (Krylov-Padé impedance):
    α_n^Hiruma = (-1)^n · Σ_k g_k² · τ_k^(n+1)
    f(s) = Σ_k g_k² · τ_k / (1 + s · τ_k)
    Hankel-Padé(α_n^Hiruma) → R, L pairs in impedance convention

  **Kameari ladder** (χ-Foster susceptibility, what NGSolve Kameari
  accumulation computes from FE matrices):
    α_n^Kameari = (-1)^n · Σ_k g_k² · τ_k^n          (POWER LOWERED BY 1)
    g(s) = Σ_k g_k² / (1 + s · τ_k)
    Hankel-Padé(α_n^Kameari) → R, L pairs in susceptibility convention

The two satisfy f(s) = -s · g(s) + Σ_k g_k² (DC offset) — different
generating functions, different Cauer ladders.

Example: cylinder R=10mm t=2mm Cu disk (50 BEM modes,
mpmath 60 digit Hankel-Padé):

  BEM Hiruma  : τ_pair = [219.32, 78.65, 40.04, 23.74, 17.07, 14.73] μs
  BEM Kameari : τ_pair = [209.14, 68.59, 34.14, 22.41, 18.52, 15.41] μs
  NGSolve axisym Kameari accumulation: [208.4, 67.5, 32.6, 20.4, 15.7, 11.3]
  NGSolve 3D Kameari + snapshot     : [208.3, 67.3, 31.8, 19.8, 15.8, 12.9]

BEM Kameari leading τ matches NGSolve axisym/3D Kameari to 0.4 %
(remaining gap = BEM mode-count truncation, axisym/3D agree to 0.02 %).
Any BEM-Foster benchmark for a Kameari iteration code MUST drop the
τ_k power by 1.

### REFERENCE IMPLEMENTATION: BEM-Foster spectrum → Kameari Cauer-I

```python
import mpmath as mp
mp.mp.dps = 60   # 60 decimal digits sufficient for ~12 stages

def hankel_pade_cauer(alphas, max_stages):
    '''Continued-fraction inversion: Cauer-I rungs from Taylor moments.
    Returns p-coefficients [p_0, p_1, ...] where R_2k=p_2k, 1/L_2k+1=p_2k+1.
    '''
    c = [mp.mpf(a) for a in alphas]
    p_list = []
    for _ in range(max_stages):
        if len(c) < 2 or abs(c[0]) < mp.mpf(10) ** (-mp.mp.dps + 10):
            break
        n = len(c)
        e = [mp.mpf(0)] * n
        e[0] = 1 / c[0]
        for k in range(1, n):
            e[k] = -sum(c[j + 1] * e[k - j - 1]
                        for j in range(k) if k - j - 1 >= 0) / c[0]
        p_list.append(e[0])
        c = e[1:]
    return p_list

def kameari_cauer_from_bem(g_squared_list, tau_list_seconds, max_pairs=12):
    '''Extract Kameari Cauer-I from BEM-Foster spectrum {τ_k, g_k²}.
    Power of τ in moment formula is n (NOT n+1) - that's the conversion.
    '''
    n_moments = 2 * max_pairs + 4
    alphas = []
    for n in range(n_moments):
        a = sum(X * t ** n for X, t in zip(g_squared_list, tau_list_seconds))
        alphas.append((mp.mpf(-1)) ** n * a)
    p = hankel_pade_cauer(alphas, 2 * max_pairs)
    rungs = []
    for k in range(len(p) // 2):
        R = p[2*k]; Linv = p[2*k + 1]
        if abs(Linv) < mp.mpf(10) ** (-50): break
        L = 1 / Linv
        rungs.append({'k': k, 'R_2k': float(R), 'L_2k+1': float(L),
                      'tau_pair_us': float(L / R * mp.mpf('1e6'))})
    return rungs

# Hiruma extraction (for reference):
# replace "X * t ** n" with "X * t ** (n+1)" in the moment formula.
```

Results (Cu, B0 = 1 T uniform):
  Sphere R=10mm (Stoll spectrum, both Hiruma and Kameari work; Kameari
    matches NGSolve to 0.000% for 6 stages):
      Hiruma  τ_pair = [728.85, 171.51, 71.68, 38.41, 23.68, 15.96] μs
      Kameari τ_pair = [694.14, 154.60, 64.08, 34.47, 21.40, 14.54] μs
  Cylinder R=10mm t=2mm (50 BEM modes; Kameari matches NGSolve to 0.4%):
      Hiruma  τ_pair = [219.32, 78.65, 40.04, 23.74, 17.07, 14.73] μs
      Kameari τ_pair = [209.14, 68.59, 34.14, 22.41, 18.52, 15.41] μs

POLICY: For new CLN extraction code, prefer Kameari accumulation.
Hiruma 3-term Lanczos may be retained for verification but is NOT the
canonical Cauer-I extractor (its sequence diverges from the physical
χ-Foster representation).
References:
  S. Hiruma, H. Igarashi, IEEE Trans. Magn. 56(3), 2020.
  R. L. Stoll, The Analysis of Eddy Currents, Clarendon Press, Oxford,
  1974 (Bessel series for the sphere).
"""


def get_cln_3d_notebook(name: str = "list") -> str:
    """Retrieve raw Python code of a shipped 3D CLN example script.

    Args:
        name: One of:
            "list"       - list available scripts
            "AT"         - CLN_AT.py (A-T formulation, primary)
            "T_Omega"    - CLN_T_Omega.py (T-Ω formulation)
            "APhi"       - CLN_APhi.py (A-Φ formulation)
            "2D"         - CLN_2D.py (2D scalar reference)
            "production" - A_ICCG_production.py (A + ICCG)

    Returns:
        Python script content (or list of available scripts).
    """
    from .cln_notebooks import get_notebook, list_notebooks
    if name == "list":
        return list_notebooks()
    return get_notebook(name)


def get_cln_3d_documentation(topic: str = "all") -> str:
    """Return comprehensive 3D CLN (Cauer Ladder Network) documentation.

    Covers Tanimoto-Kameari iterative methods for eddy current analysis:
    A-T, T-Ω, A-Φ formulations with constraint variants (penalty, gauge)
    and solver variants (A + ICCG).

    Returns:
        Multi-section markdown string suitable for LLM agent consumption.
    """
    sections = {"overview": CLN_3D_OVERVIEW, "formulas": CLN_3D_KEY_FORMULAS,
                "notebooks": CLN_3D_NOTEBOOK_INDEX}
    if topic in sections:
        return sections[topic]
    if topic != "all":
        return get_cln_3d_notebook(topic)
    return (
        CLN_3D_OVERVIEW
        + "\n\n"
        + CLN_3D_NOTEBOOK_INDEX
        + "\n\n"
        + CLN_3D_KEY_FORMULAS
    )
