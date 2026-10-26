# Explicit positive-shift backend contract

This contract is implemented in the constructor. Numerical qualification of
the updated call sites is separate from the non-FE rejection checks.

For `s0 > 0`, require the caller to choose `solver="direct"`. Reject
`solver="sparsesolv"`, including the current default when the caller omits
`solver`, with `ValueError` before accessing the mesh or assembling operators.
The error should explain that positive-shift ICCG is not supported and that
direct factorization requires explicit selection. Do not silently change
`self.solver`, add an `auto` backend or fall back after a failed solve.

Keep `s0 == 0` behavior: `sparsesolv` selects shifted ICCG and `direct` selects
regularized Cholesky with original-system refinement. For `s0 > 0`, explicit
`direct` selects Cholesky of the positive-shift operator itself. Diagnostics
must record both the requested solver and the actual factorization backend
(`sparsecholesky`); `direct` names the caller's method choice. The true-residual
criterion and every existing numerical tolerance stay unchanged.

The constructor validates this choice before touching the mesh. Migrated callers include
the cylinder, reference, shifted-element, shifted-review, field-snapshot and
initial-magnetic tests, as well as saved notebook sources. Mixed zero/positive
test cases can explicitly select `direct` for both shifts if their purpose is
not ICCG qualification; the ICCG control cases must keep their solver choice.
Saved outputs must not be relabeled as rerun results.

Qualification checks for this contract:

- Omitted/default and explicit `sparsesolv` with positive shift raise before
  mesh access; an invalid backend never launches assembly.
- Explicit positive-shift `direct` passes the existing response/element tests
  and records requested `direct`, actual `sparsecholesky`, and true residuals.
- Zero-shift ICCG/direct regression cases retain their existing criteria.
- A temporary mutation removing the positive-shift guard must fail its
  rejection test. No admission or numerical acceptance transfers to a changed
  source revision automatically.

This proposal makes no claim about positive-shift ICCG capability. Adding that
capability would be a separate implementation and validation decision.
