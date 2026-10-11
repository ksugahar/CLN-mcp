# Faster compatible local curl recovery

The colored recovery engine uses the same element curl operators and local
right inverses as the archived compatible sweep. NGSolve supplies oriented
curved-element operators, free coefficients and physical integration.

For each element, the correction is

    a[free] += C_free^+ (target - C a[element])

The former implementation dispatches this operation separately for every
element on every sweep. The new implementation greedily colors the element
supports using **free global HCurl coefficients**. Elements within one color
have disjoint writable supports, so their projections commute. Matrices are
packed by shape within each color and applied with batched NumPy contractions.
Colors execute sequentially; boundary coefficients stay fixed at zero.
Grouping is asserted to have unique writable indices before execution.

The execution order changes, so iteration counts and potential gauges can
change. Equality of the reconstructed physical curl, the original scalar and
A-T weak equations, and the full CLN coefficients and vector fields remain
mandatory. This is an iterative local inverse, not a one-pass construction.
There is no assembled global curl matrix, global vector factorization or
A-T fallback. The globally factored physical unknown remains scalar facet
potential. Coloring adds setup and packed local storage; neither is omitted
from total timing.

The detailed numerical results and engineering conclusion are in
[executed notebook 22](22_batched_scalar_cln.ipynb). Previous results are
preserved in [notebook 21](21_scalar_potential_cln.ipynb), including its original
source hashes. The optimized route reuses the original equation implementation
through a temporary recovery-engine adapter. This validation adapter must not
be invoked concurrently with the reference route within one Python process.
Independent processes are unaffected.

## Measurement

The four identical-mesh cohorts retain p=2/3/4 at 3 mm and p=3 at 2 mm,
curved geometry order 3, quadrature bonus 8, zero shift, constant material and
three full CLN stages. Original ICCG A-T, condensed A-T, old scalar V/Phi and
colored scalar V/Phi each have one first observation and three warmed runs.
Route order reverses on alternate repeats. Four NGSolve threads and one BLAS
thread are fixed for every route. Setup, scalar solves, recovery, physical
residual checks and energy recursion are timed. Mesh construction, field
sampling and cross-route comparison are excluded identically.

The fast recovery separately records target assembly, projection iteration,
algebraic curl checks and integrated physical curl checks. Scalar solves,
setup and the remaining weak checks/energy work are also reported. A separate
cProfile control uses the curl of an HCurl-interpolated smooth potential to measure the old and new recovery without contaminating the
matched benchmark. Separate full p=3 construction profiles also split electric/magnetic scalar factors from the two local recovery setups. Raw observations, ranges, source hashes and runtime are
saved in JSON. One-host timings and variable warmed observations are not
scaling guarantees.

Six native negative controls cover p=2/3/4: a one-sweep limit rejects a
compatible field before convergence, and a field with nonzero divergence
fails the mandatory integrated physical curl-field check. The CLI adapter also runs all three stages
under explicit guards against global A-T or condensed vector fallbacks.
A seeded algebraic control compares batched updates against sequential updates in the identical color order at p=2/3/4, with a 1e-12 relative limit. The existing scalar parity checker is reused for both scalar routes; new
checks cover phase accounting, controls, profiling and deliberate corruptions.

## Reproduction

    python validation/batched_cln_study.py --output docs/data/batched_cln_study.json
    python validation/batched_curl_controls.py --output docs/data/batched_curl_controls.json
    python -m validation.cln3d.CLN_VPhi_fast --maxh .003 --order 3 --stages 3 --output result.json
    python validation/validate_batched_cln.py
    python validation/validate_notebooks.py

The scope remains the simply connected internal round wire. Air, variable
material, cohomology, shifted recursions and nonlinear problems are not
validated by these measurements. See [the scalar equations and boundaries](SCALAR_CLN.md).

The native cProfile records do not cover the entire recovery wall time. Notebook 22 reports their captured self-time sum alongside wall time. They are partial call diagnostics; phase attribution and speedup use explicit wall clocks and the complete matched runs.
