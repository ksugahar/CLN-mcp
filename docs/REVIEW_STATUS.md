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
