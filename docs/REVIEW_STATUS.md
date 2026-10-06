# Reciprocal review status

The 3D, surface/POD and Foster窶鼎auer conversion additions were reviewed by Codex and **Claude Opus 5.5** in three rounds, completed on 6 October 2026. The final reviewed code revision is `494f348cb8e76e727fd31128a5e2d560e8167f18`. The subsequent status update changes documentation only.

## Findings and corrections

- Exact scalar RL conversion with simple negative real poles now rejects floating coefficients unless the caller explicitly rationalizes them, certifies real poles and positive branch values, checks reconstruction, and handles a short terminal resistance and odd element budgets.
- The 3D volume/boundary quadrature is consistent. Right-hand-side projection is recorded as a load modification, with original and projected residuals checked separately. Source identities are captured before execution and checked afterward.
- Comparison reports distinguish actual-source snapshots from auxiliary surface loads, retained rank from effective excited rank, partial timings from total offline cost, and total linkage from conductor reaction. Hold-out field and Joule metrics are independently recomputed by the validator.
- Reciprocal review of a separate legacy implementation corrected its load-projection/gauge explanation. That implementation and its development history are outside this publication.

## Verification and limits

The local runtime was Python 3.12, NGSolve 6.2.2607 and Radia 5.2.3. Both reviewers reproduced relevant checks. Verification includes exact seven- and nine-element synthesis examples, six one-stage 3D formulation/mesh cases, nine penalty-weight robustness cases, both common-excitation comparison meshes, executed notebooks, source manifests and public-boundary lint. A separate clean environment with only the workflow dependencies also ran the documentation checks. Remote workflow results remain a separate check on GitHub.

The 3D accuracy result covers linear cylindrical conductor interiors at zero expansion point and R0/L1/R2. Higher-stage/general-3D accuracy is not established. Coarse three-stage T窶徹mega fails the strict compatibility gate; passing that gate in A窶菟hi does not validate its higher circuit elements. Penalty-weight invariance checks solver robustness, not independent gauges or kernel completeness.

No equal-rank nonlinear superiority, CLN depth limit, universal Foster/CLN ranking or offline-cost ordering is established. Hold-outs measure in-band interpolation. The nonlinear enrichment experiment uses auxiliary Steklov loads and a full discrete surface operator; truncated surface helpers remain unvalidated.

Type2 retains its earlier independent derivation review. Type1 retains its separately stated Codex/Wolfram/Bessel/FEM validation; this review does not add an independent Claude derivation verdict for Type1. The introductory multiport, termination and nonlinear notebooks retain their explanatory scope.

Some stored JSON snapshots contain a `review` field recording the status when the numerical artifact was generated. Those historical fields are not the current review verdict; this page records the reviewed revision and scope. Numerical review is not a proof beyond the tested cases.
