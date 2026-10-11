# Compatible high-order element curl recovery

The original one-pass failure is a shared-constraint problem, not a curved-
geometry restriction. Notebook 20 records a matched straight/curved cylinder
control, the local equations, and complete three-stage A-T runs. Earlier
notebooks 18 and 19 retain their original successful cube and failed wire
experiments; they are not overwritten with the new outcome.

## Comparison with the source

[Bíró and Köster, Compumag 2021, Section IV](https://www.compumag.org/Proceedings/2021_Cancun/www.conftool.pro/compumag2021/index.php/OA1-2-299ecbe.pdf?filename=OA1-2-299.pdf&form_id=299&form_version=final&ismobile=false&page=downloadPaper)
describes element equations K_e t_e = f_e, previously determined coefficients,
and SVD minimum-residual solutions of the resulting small systems. Its
demonstration uses hexahedra with 36 edge degrees of freedom. This source does
not supply a p=1-only restriction or an exact one-pass guarantee for arbitrary
tetrahedral meshes. Its global scalar-potential equations are separate from
local vector-potential generation.

The cited underlying article is O. Bíró and M. Gyimesi, *Finite Element
Implementation of the Iterative Scalar Potential Method for the Computation
of Eddy Currents*, IEEE Transactions on Magnetics 57(6), 6301704 (2021),
[DOI 10.1109/TMAG.2021.3063851](https://doi.org/10.1109/TMAG.2021.3063851).
Its full text was unavailable in this investigation. We do not attribute an
unverified traversal rule or theorem to it. This is an extension of the local
constraint idea, not a claimed exact reproduction of that article.

## Why freezing fails

On each element, let K_e = C_e^T C_e after removing the gradient kernel by an
eigendecomposition. The known physical curl defines y_e from the local load,
so the compatible constraint is C_e R_e t = y_e. R_e selects oriented global
HCurl degrees of freedom. NGSolve owns element orientation, curved mappings,
quadrature and the basis; no edge-only numbering is assumed.

Even when all constraints have a common solution, an arbitrary minimum-norm
choice on an early element need not extend to later elements around mesh
cycles. Freezing every visited coefficient removes the corrections required
there. A local least-squares answer then leaves a nonzero residual. Repeating
that frozen pass cannot fix it. Both K_e-equation and square-root-equation
minimum-residual controls, three traversal orders and p=1/2/3 are recorded on
the same topology with geometry orders 1 and 3. This tests equation weighting
as well as curvature; the failure is not evidence against Bíró's entire
scalar formulation.

## Conforming correction for p > 1

Keep essential boundary coefficients fixed, but allow every free shared
coefficient to be corrected. For each element, with F_e the free local columns,
apply the small SVD update

    t[F_e] += pinv(C_e[:, F_e]) (y_e - C_e R_e t).

Every update is an orthogonal projection onto a local affine constraint set.
When the intersection is nonempty, finite cyclic projections converge to that
intersection in exact arithmetic. They need not satisfy all other elements
after a single update or sweep. For any common solution t*, an exact projection
satisfies ||t_new-t*||^2 = ||t_old-t*||^2 - ||t_new-t_old||^2. Bounded iterates
therefore have vanishing updates; a subsequence over finite complete cycles
has a limit in every closed local constraint set, and the distance identity
then gives convergence to that common limit. This argument concerns exact
consistent constraints, not arbitrary noisy data or finite-precision ranks.
The same shared global HCurl vector is used
throughout, so tangential conformity holds after every update, including edge,
face and interior modes at higher order. Potential gauges may differ; the
physical curl is the quantity compared.

The implementation builds only element matrices and SVD factors for this
inverse. It checks local representability, all-element curl fit, integrated
vector curl error, essential boundary coefficients, and the recovered field
in the original uncondensed physical equation. Requested fit tolerance is
1e-10, physical curl tolerance 1e-8, and original equation residual limit 1e-6.
A capped or incompatible recovery raises; there is no original-potential
fallback. A trial 1e-11 fit tolerance encountered a roughly 1.17e-11 numerical
floor in one curved p=1 control; its failed diagnostic is retained outside Git.
No linear-complexity, fixed-sweep-count or universal conditioning claim follows.

## Full CLN scope and reproduction

`compatible_recovery_study.py` reuses the previously checked full A-T runner
and replaces its recovery adapter. Each recovered curl enters the following
CLN stage. Four matched curved-wire cohorts cover p=2/3/4, two mesh sizes,
seven coefficients, seven physical vector modes and all seven solves. Original
ICCG and condensed A-T controls separate physical solve cost from recovery
cost. Four runs per route retain the first observation and three warmed runs,
including setup and checks. Numerical values, energy checks, vector samples,
source hashes and times are saved as JSON and displayed in notebook 20.

The physical A-T trace solve remains global. This implementation does not yet
replace it with Bíró's V/Phi scalar Poisson route, and successful local curl
recovery is not a speed advantage or an independent continuum-accuracy proof.
Positive shifts, multiply connected domains, material interfaces and air
coupling need separate evidence. The A-T accuracy ranking in notebook 17 is
unchanged by this representation-level comparison.

Native reproduction, from the repository root with NGSolve, Radia, NumPy and
SciPy installed:

    python validation/curl_sweep_diagnosis.py --output docs/data/curl_sweep_diagnosis.json
    python validation/compatible_recovery_study.py --output docs/data/compatible_recovery_study.json

Portable saved-evidence checks:

    python validation/validate_compatible_recovery.py
    python validation/validate_notebooks.py
