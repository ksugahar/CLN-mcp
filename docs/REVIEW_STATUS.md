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
