# Reciprocal review status

The 3D, surface/POD and Foster–Cauer conversion additions were reviewed by Codex and **Claude Opus 5.5** in three rounds, completed on 6 October 2026. The final reviewed code revision is `494f348cb8e76e727fd31128a5e2d560e8167f18`. The subsequent status update changes documentation only.

## Findings and corrections

- Exact scalar RL conversion with simple negative real poles now rejects floating coefficients unless the caller explicitly rationalizes them, certifies real poles and positive branch values, checks reconstruction, and handles a short terminal resistance and odd element budgets.
- The 3D volume/boundary quadrature is consistent. Right-hand-side projection is recorded as a load modification, with original and projected residuals checked separately. Source identities are captured before execution and checked afterward.
- Comparison reports distinguish actual-source snapshots from auxiliary surface loads, retained rank from effective excited rank, partial timings from total offline cost, and total linkage from conductor reaction. Hold-out field and Joule metrics are independently recomputed by the validator.
- Reciprocal review of a separate legacy implementation corrected its load-projection/gauge explanation. That implementation and its development history are outside this publication.

## Verification and limits

The local runtime was Python 3.12, NGSolve 6.2.2607 and Radia 5.2.3. Both reviewers reproduced relevant checks. Verification includes exact seven- and nine-element synthesis examples, six one-stage 3D formulation/mesh cases, nine penalty-weight robustness cases, both common-excitation comparison meshes, executed notebooks, source manifests and public-boundary lint. A separate clean environment with only the workflow dependencies also ran the documentation checks. Remote workflow results remain a separate check on GitHub.

The 3D accuracy result covers linear cylindrical conductor interiors at zero expansion point and R0/L1/R2. Higher-stage/general-3D accuracy is not established. Coarse three-stage T–Omega fails the strict compatibility gate; passing that gate in A–phi does not validate its higher circuit elements. Penalty-weight invariance checks solver robustness, not independent gauges or kernel completeness.

No equal-rank nonlinear superiority, CLN depth limit, universal Foster/CLN ranking or offline-cost ordering is established. Hold-outs measure in-band interpolation. The nonlinear enrichment experiment uses auxiliary Steklov loads and a full discrete surface operator; truncated surface helpers remain unvalidated.

Type2 retains its earlier independent derivation review. Type1 retains its separately stated Codex/Wolfram/Bessel/FEM validation; this review does not add an independent Claude derivation verdict for Type1. The introductory multiport, termination and nonlinear notebooks retain their explanatory scope.

Some stored JSON snapshots contain a `review` field recording the status when the numerical artifact was generated. Those historical fields are not the current review verdict; this page records the reviewed revision and scope. Numerical review is not a proof beyond the tested cases.

## Research follow-up: three review rounds

The follow-up research received independent **Claude Opus 5.5** review at `0b5baa62d3b56c9bf97095a8dced2289b2adc4ff`, correction review at `c2395b7296f9ee4555c140fa6daf1202f3503f8f`, and final scientific acceptance at **`ea9570706dbeb90dda1357368e39899ada78fbf5`** on 6 October 2026. Codex implemented the experiments and corrections. This status update changes documentation only. The earlier limits above describe the original publication; the following tested settings extend that scope.

- A three-stage, seven-element zero-shift round-wire ladder was derived in Wolfram Language and a second CAS using Bessel power series and continued-fraction extraction. The reviewer also checked Bessel contact independently at high precision. Twenty-four FE cases retain one strict compatibility-gate failure; at order 3, extra quadrature 8 and mesh size 1.5 mm, all seven elements of A–phi, T–Omega and A–T are within 0.2% of the analytic values. This does not validate arbitrary shifts, stage counts, meshes or general 3D geometry.
- Proper finite-pole Foster fits and field-based reductions use matched retained state counts and the stated common training grid. Offline information and costs differ. The four-state Foster/POD ordering reverses between the tested meshes; no universal ranking or global optimizer optimum is established. A nonzero fitted DC offset causes divergent relative error toward zero frequency for the stated normalized reaction transfer.
- Nonlinear trajectory POD receives a full training solve and all snapshots. Held-out amplitude 0.5 is within the training range and 1.5 is extrapolation. The explicit preserved two-bulk-mode control shows that adding auxiliary surface states can improve Joule error while slightly worsening linkage error. Same-space modal coordinates preserve the coupled nonlinear response; they do not remove nonlinear controlled sources.
- Full sparse mixed A–phi references include original-equation residuals, penalty-invariance tests, runtime identities and guarded source hashes. The final order-2 pair changes the response by about 0.14% at four sampled frequencies, but uses a much smaller element-count step than the order-1 pair. The finest cross-order gap is about 1.8%. No asymptotic convergence rate, continuum accuracy or unsampled-band bound is established.

The three review rounds closed ten disclosure/reproducibility items and a blocking overstatement of mesh convergence. All six lightweight documentation/validation commands passed for the final scientific revision in both reviewers' environments. These validate stored evidence and source identities; they do not run native FE or Wolfram computations in CI. Native results were genuinely regenerated when their producer sources changed. Independent numerical reruns are scoped in notebook 10.

Codex also reviewed a separate immutable legacy-pointer replacement, preserving text-retrieval names and distinguishing executable scripts from pointers. That development change is not included in this public research revision. TEAM 28 remains a separate implementation and review campaign; this verdict contains no TEAM 28, full 3D nonlinear or Model B claim. Remote publication and its CI result remain separate checks.

## TEAM 28: physical CLN and force

Codex implemented the TEAM 28 addition. Independent **Claude Opus 5.5** review
accepted its method and numerics at `834eaace2dc9aaae30d1e47d168028dcdd7d6982`;
the correction review gave scientific acceptance at
**`345041da92087a799d47d5eed74de659b91d8c96`** on 7 October 2026. This final
update changes status text only, preserving numerical sources, data, notebook
code and outputs.

- Axisymmetric Model A uses the actual prescribed winding current and a regular
  `w=A_theta/r` formulation. Six magnetic states give twelve physical Type1
  elements at zero expansion point, derived from magnetic-energy and Joule-loss
  norms. Energy orthogonality, field-volume element values, circuit/field
  equality and power balance are checked. No Arnoldi basis is relabelled CLN.
- Full and reconstructed CLN forces use the conjugate peak-phasor Lorentz
  average. All 25 heights pass the 1 mN same-mesh reduction gate. Only the
  official stationary height of 11.3 mm is an external measured reference.
- Fresh root refinement gives about **11.11 mm**, a gap of about **0.188 mm**,
  passing the 0.6 mm gate. The original 1 mm-grid interpolant is about 11.13 mm.
  Their difference, remeshing fluctuations and the remaining coarse settings
  probes are disclosed; no continuum or infinite-domain bound is established.
- The reviewer reproduced three native sweep samples and all five native root
  refinement samples bit-identically. The revised Wolfram output also reproduced
  byte-identically. The symbolic exhausted two-state proof and two exact rational
  truncated examples have separate scopes; the examples do not prove a general
  theorem for arbitrary semidefinite systems or stage counts.
- Seven lightweight validators passed independently. CI checks stored evidence,
  circuit reconstruction, independent symbolic instances and source identities;
  it does not run native FEM or Wolfram. Canonical LF evidence bytes are required
  for existing protected baseline hashes.

The two rounds closed the interpolation-precision, proof-scope, citation and
Type1-explanation findings. The tested scope excludes Model B, coupled transient
motion, winding resistance, shifted/Type2 TEAM 28 and general 3D nonlinear CLN.
The separate legacy-pointer review remains a development change outside this
public addition. Remote publication and its workflow result are separate checks.

## General 3D terminal CLN: scientific review accepted

Independent Claude Opus 5.5 design review accepted the terminal mixed blocks,
physical Type2 recurrence and positive-shift normalization on 7 October 2026.
An independent pilot check extracted the first four A–phi elements from full
assembled matrices using contour Taylor coefficients and continued-fraction
division. It confirmed the elements and contact order four for that pilot.
The eight-element numerical candidate then received independent scientific
acceptance at **`c18840af58cf83c672fadfba292c9aaaf1020117`** on 7 October
2026, following review at `7a9a8e160504bacf05918acc88d302bcace10858`.
The required resolution-scope correction and all four disclosure/validator
findings were closed with no new findings. Codex implemented the corrections;
Claude Opus 5.5 independently reviewed the exact revision.

Codex implemented the non-symmetric notched conductor, its one-volt terminal
port, eight-element ladders, same-mesh full references and h/p comparisons.
A–T and T–Omega use a free scalar current lift and zero-port curl(T)
corrections. The DC current lift is shared; T–Omega additionally shares the
A-derived magnetic port lift. Matching the first zero-shift elements is
therefore a built-in control. The two T forms also share zero-shift R2 by
construction: their common current space and magnetic lift give the same
first correction load. This is not independent cross-form evidence. Deep
R4-L7 elements are not cross-validated; their gaps and full-model gaps require
refinement. No exact continuum agreement or bracketing is claimed.

The positive-shift resistor coefficients use inverse Joule-plus-s0-magnetic
norms. Both physical energy integrals are retained. CI reconstructs the
small assembled example and checks contact order twice the retained electric
mode count, while refined FE data remain stored native evidence. Both 100 kHz
and 1 MHz fail the delta >= 2h/p resolution screen on every study mesh and
support only same-mesh reduction comparisons. The highest sampled frequency
passing that screen on any study mesh is 10 kHz; passage is not an accuracy
bound and still needs refinement evidence. There is no air, external return path or general nonlinear claim.

The independent checker confirmed the full terminal response, shallow
elements and contact order eight from original assembled matrices. All eight
CI validation commands passed on the exact-revision clean checkout, the
q7/q8 corruption controls were rejected, and the executed notebook had no
error outputs. This status-only update changes no code, data or execution
outputs. Scientific acceptance does not assert a new publication or remote
CI outcome.

## General 3D CLN with surrounding air — scientific review accepted

The surrounding-air design was independently accepted on 7 October 2026.
Independent Claude Opus 5.5 numerical review gave **SCIENTIFIC ACCEPT** at
`3ab2b33999e59ea8d9818cde5b1ff8cd6db14efa` on 7 October 2026, with no P1/P2
findings and three informational P3 notes. Codex implemented the candidate.
The source-break port models an ideal terminal drive and perfectly conducting
return enclosure; it excludes shell loss, lead capacitance and displacement
current. Air conductivity is exactly zero. A global gradient gauge is checked
against the original equations, and all ladder inductances use total magnetic
energy. Regional energy splits, coaxial analytic errors, geometry errors,
marked conductor/air refinement and same-mesh reduction errors are reported
separately in notebook 13. The A–T control shares the global magnetic operator
and DC lift; T–Omega is not implemented for this air topology. Passing a
skin-depth screen is not a continuum accuracy certificate. The reviewer independently reconstructed the full response and continued
fraction from the original exported matrices, confirmed Z(s0)=Rhat0 and
R0–L3 within 1.1e-10, and checked the coax oracle and executed Wolfram proof.
All nine validation gates passed on the exact-SHA clean checkout.

The informational notes retain the non-monotone deep coax elements and up to
8.3% notched full-response mesh change at 100 kHz; neither is a convergence
or continuum accuracy claim. The A–T magnetic orthogonality defect of
2.70e-9 has only a 3.7-fold margin to its 1e-8 gate. The conditioning audit
uses lower-bound estimates, not certified forward-error bounds.

This status-only update changes no producer code, numerical data, notebook
code cells or execution outputs. Scientific acceptance is limited to the
stated tests and does not assert publication or a remote CI outcome.

## Surface modes in the physical CLN — scientific review accepted

Independent Claude Opus 5.5 numerical review gave SCIENTIFIC ACCEPT at
`7a3b1606bd2a69e420fdba7b9efad0abb88f0743`, with no P1/P2 findings. Notebook 14 implements physical quotient air condensation and nested trace-space restriction; omitted trace modes are clamped. Fixed-current compliance increases from below as the space expands. The DC port trace is protected, so its exact L1 is a construction control, not an independent convergence result. Source-matched full-rank controls, dynamic Z and energy/element convergence are separate tests. The independent checker reconstructed the physical air Schur with explicit
gradient constraints, its trace kernel and Steklov spectrum, nested compliance
and full responses from original small-mesh matrices. It also checked
continued-fraction elements at three K values and both shifts. All ten gates
passed on the exact-SHA clean checkout. Coax and notched p2 evidence were
inspected as stored results; their original matrices were not exported.

The three informational wording notes clarify magnetic compliance versus
unit-current inductance. The maximum measured
coax orthogonality defect is about 9.97e-10, and the coax DC Schur residual
is about 2.25e-13. These pass the existing gates without changed numerical
outputs. Deep elements, protected-DC identities and coarse FE errors retain
the stated limits. This status/wording-only update changes no code, data or
execution outputs and does not assert publication or remote CI success.

## Air cohomology T–Omega — scientific review accepted

Codex implemented a curl-constrained physical T–Omega current pencil with
Radia absolute air cohomology, a free linked terminal-current channel and a
natural scalar magnetic correction. Air conductivity is zero and total
magnetic energy enters the Type2 recursion. Nine native cohorts and six
same-mesh A–T controls are stored; the small original-matrix NumPy checker
reconstructs the admissible field space, topology rank, current period,
physical stage fields, full responses and coefficients q0 through q8.
Four Taylor corruption controls (q7/q8, both shifts) are rejected.

The two vertex permutations return the same harmonic cochain; their comparison is a reproducibility check. A separate small native control changes the raw cocycle by a gradient with relative size 1.0 before independently assembling and solving each natural scalar equation. All eight elements at both shifts agree. The test covers the 118-tetrahedron notched cohort and does not construct different geometric cut surfaces. On the
order-1 notched meshes (244, 695, 1748 tetrahedra), full-response differences
and R2/L3 differences decrease. The finest full T–Omega/A–phi gaps remain
9.2% at 100 kHz and 14.5% at 1 MHz; shifted T–Omega/A–T R2/L3 gaps remain
19.3%/33.6%, with deep gaps up to 181%. These are not cross-validated
physical elements or established continuum accuracy. HCurl orders 0 and 1
share the curl-current dimension here; the comparison enriches magnetic
scalar fields. The coax 12-to-8 mm maxh pair does not increase the mesh
count and is explicitly not a refinement claim. Larger native cases are
stored evidence; original-matrix independent CI reconstruction is limited
to the small notched cohort. See notebook 15 for reproduction and references.

Independent Claude Opus 5.5 scientific review accepted revision
`61a8a241f72884a1ab4d09b616a0e17aa7f8b0d5` on 7 October 2026. The reviewer
independently solved both changed-cocycle scalar systems, checked all eight
elements at both shifts, and passed all workflow validators. Publication
and remote CI remain separate checks. This status update changes no producer,
data or execution outputs. C2 (a conducting
washer with a loop-current channel and coil-driven Type1 CLN) is not included.

The coax comparison supports low-frequency impedance magnitude only, not
analytic element-level validation. Order-1 R2/L3 gaps are about 26%/16%,
deep gaps reach about 1500%, and Re Z at 1 MHz differs by about -32%.
The Radia-oriented standalone mesh/region API extraction is a separate
maintenance follow-up; these validation helpers are not a generic Radia API.

### C2 dependency and publication boundary (work in progress)

C2 is planned as a coil-driven washer Type1 CLN: ladders, energy-normalized
elements, contact and orthogonality evidence, notebook and validator belong
in CLN-mcp. Generic region cohomology, natural magnetic representatives and
insulated minimum-Joule loop currents belong in Radia's `cohomology.py` and
`cohomology_cut.py`, without CLN code. Their publication follows the Radia
owners' separate process and requires explicit user instruction.

Until that Radia API is published, corresponding CLN-mcp validation helpers
are temporary and do not establish a second maintained API. C2 must use the
published Radia API once available, with the tested version pinned in README.
Radia 5.2.3 used for existing runs does not imply availability of the new API.
F2, the dependency migration and independent full T-Omega comparison remain open;
C2 is not included in the published C1 evidence.

The C2 candidate now has [an executed washer notebook](16_coil_washer_type1.ipynb)
and original small-mesh pencils/field stages. Primary A-phi and a current-space
A-T control use the actual closed-coil excitation and eight Type1 elements at
zero and positive shifts. Coefficient-wise H=Z/s contact q0..q7, energy/Joule
orthogonality, full-model power balance, two sampled radial cut fluxes, a
filled-hole control and omitted-loop responses are checked. A genuinely changed
representative is separately assembled and solved; its scalar systems, currents
and element comparisons are exported. The NumPy validator rejects q7/q8
coefficient corruption.

Native reruns include 1644/4501 tetrahedra and 1648/4860/40722 free A degrees
of freedom. Magnetic enrichment reduces the coil self-inductance gap to -5.62%
against the refined axisymmetric reference, but the 1 MHz loss gap remains
+24.0%. The conductor skin layer is unresolved at both 100 kHz and 1 MHz.
These results establish same-mesh CLN validation, not continuum accuracy.
The axisymmetric reference has a separate solid-cylinder Bessel/WLS limit
check; the washer/coil has no analytic oracle. A-T shares the A magnetic inverse
and does not establish an independent full T-Omega model. That cross-check
remains open. Claude independently accepted the same-mesh scientific scope at
`d96b614ad4b308741a8a74fabe26a655ea430207` on 2026-10-08 (C2 r09).
This acceptance does not establish continuum accuracy. FE convergence of the
loss, the independent full T-Omega cross-check, and F2/Radia API migration
(including the tested published version pin) remain open.

### Matched round-wire formulation ranking (2026-10-11)

[Notebook 17](17_formulation_ranking.ipynb) identifies the notebook-06 figure
as order 1, zero shift and maximum R0/L1/R2 error. Forty native field-recursion
cases reuse shared 3D meshes at maxh 3/2/1.5 mm, orders 1/2/3 and coefficient
windows of 3/5/7 elements, with one-space A-T enrichment and profile controls.
The matched quadrature-8 baseline completes all 27 cases. One quadrature-4
T-Omega load rejection is intentionally retained; no numerical gate is relaxed.

An independent full FE response/Taylor oracle contributes 36 zero/positive-
shift Type1/Type2 cases. It uses scalar and magnetic Schur complements with
NGSolve sparsecholesky subsolves, checked matrix-free CG and exact-gradient
gauges, without a physical mass perturbation. At zero shift its first-five
coefficients agree with field recursion within 2e-6 relative; independent
Taylor Pade evaluation checks the continued fractions. Recorded original-
system residual maxima are below 5.3e-9 (field study) and 1.5e-13 (outer
response pencils). Original magnetic subsolve residuals are below 2.9e-13.

The p=1 first-three ranking favors A-phi. A-T is better for the seven-element
coefficient metric on all nine matched baseline cohorts, and for positive-
shift first-five coefficients on all four pencil cohorts. At p=3, maxh=1.5 mm,
s0=2*pi*1e5, Type2 first-five errors are 1.1532% (A-phi), 1.2315% (T-Omega)
and 0.08909% (A-T), approximately a factor of thirteen advantage. Enriching
only A or only T changes the A-T error pattern toward the corresponding
one-potential route. This supports a discretization-space explanation,
not a universal formulation theorem or guaranteed error cancellation.

No general-shape, equal-cost, all-order, field-accuracy or terminal-response
superiority follows. The high-order terminal curves approach the same finite
analytic ladder and are often limited by truncation. Native evidence and
portable checks are Codex self-review; a new independent external scientific
review is not claimed. The producer sources, hashes, runtime, raw numerical
values and five corruption controls are retained in the repository.

### High-order element curl recovery (2026-10-11)

[Notebook 18](18_element_curl_recovery.ipynb) distinguishes local vector-
potential recovery from the global scalar Poisson problems in Biro and
Koster's Compumag 2021 paper. On one affine, simply connected 12-tetrahedron
cube, two compatible discrete fields at each of orders 1/2/3 are recovered
by local minimum-residual sweeps in forward/reverse/shuffled element order.
The largest high-order single-sweep curl error is 1.8e-13. An independently
assembled global pseudoinverse, element-operator LSMR, and integrated curl
errors check the result. Incompatible element curl data retain residuals
above 54%, as required; local solves do not repair compatibility.

This is a working high-order local-recovery prototype, not a proof for every
mesh or a replacement of the A-T CLN implementation. Curved elements, holes,
material interfaces, actual CLN source compatibility, scalar corrections,
and measured cost remain open. The paper consulted does not state a p=1-only
restriction, and cites a 36-edge hexahedral example; identifying the precise
claimed p=1 theorem remains open. No such theorem is attributed without its
source. Six native results, source identity and portable gates are saved.

### Full A-T CLN with trace/local recovery (2026-10-11)

[Notebook 19](19_element_cln_integration.ipynb) integrates element-local
recovery into the unchanged zero-shift A-T field recursion. Four curved-wire
cohorts (p=2/3/4, maxh 3 mm, plus p=3 at 2 mm) compare original ICCG, a cached
uncondensed sparsecholesky control, and NGSolve static condensation with local
harmonic recovery. Each route has one first observed run and three subsequent
runs; setup and original-equation verification are included in time. All 48
complete runs pass seven-coefficient, seven-volume-vector-mode and original
equation checks. The largest condensed/original coefficient difference is
3.77e-11; the volume field difference is 8.16e-10. The condensed route's largest
uncondensed original-equation residual is 7.47e-14. Saved field energies and
78-point vector samples provide independent portable checks.

At p=4, 3 mm, warmed median full-run times are 5.905 s (original ICCG), 3.485 s
(full Cholesky) and 3.266 s (trace/local). The approximately 1.81x improvement
against the original workflow includes backend and factor-reuse effects.
Against the matched Cholesky control the observed p=3/4 time reduction is
about 3-6%; this is three samples on one host, not a general speed theorem.
At p=2 there are no internal dofs; p=3/4 eliminate 3/11 per tetrahedron.
Full original matrices are still assembled for equation verification.

The earlier greedy curl sweep is also inserted before advancing each CLN
stage. All 16 attempts reject T stage 0 on the actual curved wire; relative
curl-field errors are approximately 2.01-2.53. These failures are preserved,
with no silent fallback and no fabricated later coefficients. The affine
cube prototype therefore does not establish arbitrary-mesh one-pass recovery.
The accepted integration retains a global trace system; it is distinct from
Biro/Koster's scalar-potential construction. Positive shifts, nonlinear/air
models, topology and a generally compatible one-pass algorithm remain open.

LAB/intel11 native evidence, source hashes, complete timings, raw fields and
five corruption controls are saved. Existing documentation CI executes the
portable gate. These additions are Codex self-review, without a new external
scientific acceptance claim. Raw failed-run diagnostics are recovered outside
Git with SHA256 verification before job-owned remote staging is removed.

### Compatible iterative element recovery (2026-10-11)

[Notebook 20](20_compatible_element_recovery.ipynb) and its
[source comparison and derivation](COMPATIBLE_ELEMENT_RECOVERY.md) resolve
the cause of the archived one-pass wire failure. The same mesh topology at
geometry orders 1 and 3, p=1/2/3, and three traversal orders is tested with
both local K-equation and square-root-equation least-squares solves. All 36
frozen-sweep controls fail; integrated relative curl errors range from 1.86
to 8.42. The forward K-equation first becomes unextendable at visited element
56 for every order and geometry control. Curvature is therefore not necessary
for the failure. An arbitrary early local gauge choice is not guaranteed to
extend once shared coefficients are frozen.

The correction revisits free shared degrees of freedom using element SVD
projections. NGSolve's full HCurl space preserves tangential conformity at
higher order. All 18 traversal/geometry/order controls converge in 200 sweeps,
with physical curl errors below 8.34e-11. Six one-sweep caps and six targets
with nonzero divergence reject. The local inverse assembles no global matrix;
there is no silent original-potential fallback. Multiple sweeps are required;
no exact one-pass or linear-complexity claim is made.

Four curved-wire cohorts (p=2/3/4 at 3 mm, plus p=3 at 2 mm), three stages,
four runs and three routes give 48 complete accepted CLN runs. All 112 local
recoveries feed the following recursion stages and pass the original equation
gate. The largest iterative/original coefficient difference is 1.054e-10;
the largest volume vector-mode difference is 1.055e-8. The largest recovered
uncondensed equation residual is 4.256e-10. Each recovery takes 100-360 sweeps.
Field energies, 78-point vector samples and all convergence histories are saved.

This is not a speed improvement. Original / condensed / iterative warmed
median full-run seconds are 0.836 / 0.665 / 5.190 (p=2, 3 mm),
2.022 / 1.551 / 9.205 (p=3, 3 mm), 6.071 / 4.648 / 37.486 (p=3, 2 mm),
and 5.944 / 3.302 / 18.635 (p=4, 3 mm). Setup and verification are included;
three warmed observations on LAB/intel11 do not establish general scaling.

The global physical A-T trace solve remains. The full V/Phi scalar route,
positive shifts, air/material interfaces and nontrivial topology remain open.
The accessible Bíró/Köster conference source is distinguished from the cited
Bíró/Gyimesi journal article whose full text was unavailable; no p=1-only
theorem is attributed to either. Saved source/coverage/energy/field/residual
gates include eight corruption controls in documentation CI. This is Codex
self-review, not a new external scientific acceptance.

### Scalar-global V/Phi CLN replacement (2026-10-11)

[Notebook 21](21_scalar_potential_cln.ipynb) and the
[scalar formulation note](SCALAR_CLN.md) replace the remaining global physical
A-T vector solves on the matched internal-wire cases. The Bíró V/Phi scalar
Poisson equations use a compatible hybrid-mixed discretization, rather than
the accessible paper's nodal implementation. NGSolve eliminates discontinuous
HDiv flux and cell scalar unknowns locally, leaving only scalar facet traces
for the global `sparsecholesky` factor. Repeated element curl projections
provide the cumulative particular potentials for subsequent CLN stages.

The scalar route receives no A-T reference field or coefficient. Runtime
guards forbid the original A-T solver and vector-condensation constructor
during its setup and recursion. Every retained global factor is scalar-sized;
no global curl-curl matrix is assembled on that route. Original A-T and its
condensed control execute separately on the same mesh. Scalar gauge fixing
adds no mass perturbation; a matrix-free uncondensed residual checks all
original equations, including the pure-Neumann gauge row.

All 48 complete runs and 112 scalar/local steps pass for p=2/3/4 at 3 mm and
p=3 at 2 mm, geometry order 3, quadrature bonus 8 and three full CLN stages.
Maximum scalar/original differences are 1.287e-10 for seven coefficients and
6.519e-9 for seven volume vector modes. The maximum original scalar-system
residual is 1.549e-14, the original A-T physical weak residual is 7.408e-11,
and the recovered original physical residual is 4.221e-10. Energy identities,
78-point vector samples, global factor dimensions and convergence histories
are retained. The standalone scalar CLI also passes a native smoke run.

Ordinary nodal Phi controls at p=2/3 and geometry orders 1/3 solve their scalar
systems to below 1.54e-14 but fail strict compatible local recovery. Even
unconstrained element projection leaves curl-image error lower bounds between
2.08e-4 and 1.352e-2. These controls motivate the compatible discretization;
they do not invalidate the published approximate nodal scalar method.

Measured original / condensed / scalar warmed median full-run seconds are
0.769 / 0.622 / 5.584 (p=2, 3 mm), 2.032 / 1.540 / 10.442 (p=3, 3 mm),
6.893 / 4.595 / 59.021 (p=3, 2 mm), and 5.760 / 3.212 / 21.299 (p=4, 3 mm).
NGSolve uses four threads and BLAS one; all routes include setup and residual
checks. At 2 mm, warmed ranges are 5.914-13.674 / 4.508-10.054 / 39.435-86.135
seconds, so those timing medians have substantial variability. Every raw
observation is retained. There is no measured speed advantage; element curl
iterations and scalar trace dimensions remain costs. A preliminary four-BLAS-
thread run was stopped after timing variation, with its sources, partial
results and logs retained outside Git.

The physical vector solve is replaced for these constant-material, simply
connected zero-shift wire cases. Air, variable material, cohomology, positive
shifts and nonlinear models remain unverified. The original paper's model and
speed comparison are not transferred to this discretization. Portable gates
include eight new corruptions and run in documentation CI. These additions
are Codex self-review, not a new independent scientific-review acceptance.

### Colored scalar CLN recovery (2026-10-11)

[Notebook 22](22_batched_scalar_cln.ipynb) and [the implementation note](BATCHED_SCALAR_CLN.md) retain the original scalar physics and replace element-by-element Python dispatch with colored batched projections. Each color has disjoint free global HCurl coefficients, so its updates commute. NGSolve's curved geometry, orientation and boundary constraints are preserved. Setup, all residual checks and complete recursion are included.

All 64 matched complete runs pass across four cohorts, with 112 optimized scalar/local steps and 112 old scalar/local steps. Maximum optimized/original differences are 1.2117e-10 for coefficients and 5.3505e-09 for volume vector fields; maximum recovered original physical residual is 4.1997e-10. Six native capped/incompatible controls reject, a standalone guarded fast entrypoint completes seven steps, and five new evidence corruptions are rejected.

Warmed medians, with every raw observation and range saved in JSON:

| Cohort | Original A-T [s] | Condensed A-T [s] | Old V/Phi [s] | Colored V/Phi [s] | Old/new scalar |
|---|---:|---:|---:|---:|---:|
| p=2, 3 mm | 0.875 | 0.703 | 6.122 | 3.679 | 1.66x |
| p=3, 3 mm | 2.011 | 1.523 | 10.030 | 8.009 | 1.25x |
| p=3, 2 mm | 6.060 | 4.494 | 40.061 | 25.952 | 1.54x |
| p=4, 3 mm | 5.939 | 3.374 | 22.155 | 19.842 | 1.12x |

The local inverse is accelerated, but the optimized scalar route remains slower than both A-T references in these cohorts. The older scalar implementation spends most time in local recovery; profiling and new phase timers distinguish projection iterations, element target assembly, algebraic curl checks, integrated checks, scalar solves and setup. Remaining target assembly and physical checks limit total speedup. No global vector solve or relaxed accuracy gate is introduced. The same internal-wire, constant-material, zero-shift scope applies. These additions are Codex self-review, not independent scientific-review acceptance.

### Bulk assembly and mexcuda validation (2026-10-11)

[Notebook 23](23_bulk_scalar_cln.ipynb) and [implementation/evidence](BULK_SCALAR_CLN.md) preserve quadrature order 13 and all scalar, curl, boundary and original physical residual gates. Bulk broken-space assembly and compound checks pass twelve native equivalence controls. The historical LAB block measures a 1.50–1.80x full-recursion improvement over colored recovery, with substantial timing variation; A-T remains faster in that block.

The selected GPU route is mexcuda. Four cohorts (p=2/3/4, curved meshes, three stages) pass complete coefficient and volume-field parity: maximum relative differences 1.21171e-10 and 5.34719e-09. Ten native input/lifecycle rejection controls pass. The standalone CUDA kernel uses checked handles, persistent operators, and a device fit reduction; FE assembly and scalar solves remain on CPU.

Whole-method memory is not automatically reduced by local recovery. At higher orders the CPU scalar route exceeds A-T memory in these cases. With the separate MATLAB process included, sampled simultaneous mexcuda working sets are approximately 2.0–2.4 GiB. Native per-process peaks and sampling limitations are saved separately.

The upgraded 100 host is authorized for CI, tests and numerical/memory validation. Formal performance measurements require an otherwise idle mdx. The current mdx is busy and lacks an NVIDIA runtime, so formal mexcuda speed remains unverified; the saved accuracy runs intentionally contain no timing claims. CuPy and earlier LAB timing data are exploratory evidence only. These results are Codex self-review, not independent scientific acceptance.
