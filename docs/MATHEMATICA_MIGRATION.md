# Mathematica symbolic reference migration

SciPy remains available for numerical analysis. Mathematica owns maintained
symbolic derivations and exact synthesis. Python reads source-bound records
and evaluates independent numerical circuit, Fraction Bessel-series and
small-matrix moment oracles; no SymPy import/fallback remains in live code.

The symbolic core received independent acceptance. The record producer
adds concrete conversion output, notebook gauge/termination identities and
InputForm check labels. It was evaluated with 39 successful checks and
four detected negative controls. The source-bound output is retained and the
changed notebook cells have been replayed from that output. These checks cover
finite scalar synthesis and illustrative identities, not general FE or passivity.

- input_ports_exact.py reads Mathematica port/caveat checks plus the immutable
  original illustration; an independent Fraction series checks wire/Bessel
  contact through degree8. The accepted RoundWireMaxwell module retains E-H/A/T.
- foster_cauer_exact.py is removed. SymbolicCLN.wl owns exact conversion;
  validate_foster_cauer.py checks stored exact results and independent complex
  responses at the original frequencies and1e-12 threshold.
- validate_team28.py uses the existing source-bound Mathematica energy proof,
  retaining all FE force/residual/power criteria and an independent exact
  rational numerical recurrence/Galerkin moment calculation for truncated cases.
- validate_higher_stages.py reads the existing Mathematica Bessel coefficient
  record. Existing FE records remain immutable; reference_migration.json binds
  their original bytes and historical producer hashes. This is provenance,
  not recertification of new FE runs or a rewritten baseline.
- Notebook02/04/09/16 read these records. CI runs portable record/numerical
  checks without a symbolic engine. Fresh CAS regeneration is explicit.

Caveats retained: regular Ampere field r/2; imposed subtraction1/2 produces
-1/(2r) defect; shifted branch s-1 has physical DC -1; Euclidean quotient1
and remainder-2. Positive shifted coefficients do not certify physical-s
passivity. Exact integer/rational input is mandatory; optional guessed
rationalization is deliberately removed. Historical SymPy metadata remains
producer-time provenance. No scientific threshold/reference is changed.

The gauge checks are bookkeeping identities, not independent physical oracles.
T/Omega H invariance follows from gradient linearity; shared curl definitions
do not detect a global sign reversal on both sides. Maxwell/Bessel reconstruction
and independent circuit/moment comparisons have separate roles.
