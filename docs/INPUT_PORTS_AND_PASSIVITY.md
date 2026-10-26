# Input ports, boundary lifts and passivity

This note separates three questions: whether two formulations describe the
same terminal problem, whether their ladder coefficients agree, and whether a
finite approximation is passive. Agreement on one question does not establish
the others. The companion [executed notebook](16_input_ports_and_passivity.ipynb)
uses exact small-matrix algebra and a circular-wire Bessel-series oracle;
it does not add a general 3D, shifted-synthesis or homogenization validation.

The literature reference is S. Hiruma, Y. Takahashi and T. Matsuo,
“Homogenization Method Based on Cauer Ladder Network Representation of Unit
Cell,” *IEEE Transactions on Magnetics* 60(12), 7002304 (2024),
[doi:10.1109/TMAG.2024.3486683](https://doi.org/10.1109/TMAG.2024.3486683).
Equations (7)–(15) specify the boundary lift and B-input admittance; the
appendix defines the H/E transfer functions. The PDF and third-party example
notebooks are not redistributed here.

The field-recursion background is A. Kameari, H. Ebrahimi, K. Sugahara,
Y. Shindo and T. Matsuo, “Cauer Ladder Network Representation of Eddy-Current
Fields for Model Order Reduction Using Finite-Element Method,” *IEEE
Transactions on Magnetics* 54(3), 7201804 (2018),
[doi:10.1109/TMAG.2017.2743224](https://doi.org/10.1109/TMAG.2017.2743224).

## 1. B/H/E input is a port definition, not just an index

In the paper, B-input prescribes an external flux through a static magnetic
potential lift. H-input uses a magnetic boundary excitation, and E-input uses
an electric excitation. Their conjugate output, input normalization,
constrained unknown space and transfer function must be specified before
comparing coefficients. The appendix expands an H-input impedance and an
E-input admittance. For a matched conjugate port and the same normalization,
their B/H ladders do agree after shifting the indices by two, as shown below.

For the B-input definition in equation (13),

\[
Y_B(s)=\frac{c}{s}+Y_{\mathrm{response}}(s),\qquad
c=a_*^TKa_*,\quad L_{\mathrm{exc}}=1/c.
\]

The response is synthesized in the E-input form with the compatible static
lift as its excitation. Thus removing the parallel excitation inductance
leaves that *matched response admittance*. For a matched B/H conjugate port,
the unshifted energy recurrences give exactly

\[
L_{-1}^{B}=L_1^{H},\qquad R_0^{B}=R_2^{H},\qquad
L_1^{B}=L_3^{H},\qquad R_2^{B}=R_4^{H},\quad\ldots
\]

This holds while the corresponding modes have nonzero norms; exhaustion
terminates the recurrence. The caution about unequal coefficients applies
to *different or mismatched ports*, not to this matched case. An arbitrary
E-input excitation need not equal the matched B-response excitation.

An exact normalized one-boundary-degree-of-freedom example illustrates the
necessary port matching. Let \(M=K+sN\), eliminate free interior unknowns,
and denote the boundary Schur complement by \(S(s)\). For unit boundary
potential and its conjugate boundary reaction,

\[
Y_B=S(s)/s,\qquad Z_H=s\,e_b^TM^{-1}e_b=s/S(s),\qquad Y_BZ_H=1.
\]

The checker derives one transfer by a fixed-boundary solve and the other by
full matrix inversion. It then compares the constrained B-input and full-space
H-input element recurrences against scalar Laurent/DC extraction from that
independently inverted transfer. The matched toy gives
\(L_{-1}^{B}=L_1^{H}=5/17\) and \(R_0^{B}=R_2^{H}=25/39\).
Only a different free-space load gives \(L_1=3/5\); that is the
*mismatched-port* counterexample.
The example's \(Y_E\) label means the matched response term, not an
independent E-input boundary-value solve.

To see why the index shift persists, let \(b\) be the unit boundary load.
Then \(Ka_*=cb\), so the first H-input vector is \(a_1^H=a_*/c\)
and \(L_1^H=1/c\). The next electric vector is \(e_2^H=-a_*\).
After that, the H vectors are the negatives of the corresponding B-response
vectors in the homogeneous space: the static component cancels. Their
quadratic energy norms therefore agree. This argument assumes the same
scalar port, spaces, normalization and unshifted recurrence; it does not
identify unrelated terminal or shifted constructions.
The sketch assumes symmetric \(K,N\), an invertible full-space \(K\)
for H-input, and subsequent B-side electric modes N-orthogonal to the static
lift. A 3D curl-curl operator has a kernel and needs separate admissible-space
and gauge treatment before applying this argument.

## 2. Nonzero boundary data require a static lift

Write \(a=a_*+Pu\), where \(a_*\) has the prescribed boundary trace and
\(P\) injects free unknowns with zero trace. The static solve gives
\(P^TKa_*=0\). Consequently,

\[
(P^TKP+sP^TNP)u=-sP^TNa_*,\qquad a_*^TKPu=0.
\]

The first excitation carries the nonzero boundary value; correction fields
live in the homogeneous space. Restarting a recurrence while merely changing
the boundary condition can change its right-hand side, normalization and
energy identities. This is why the static excitation and later correction
modes must be explicitly distinguished. Static-lift orthogonality is separate
from mutual orthogonality of successive response modes.

Here \(K\) is the unconstrained energy operator. One must not identify its
lift quadratic form with the interior operator after boundary elimination.
In particular, the last compact product in the paper's equation (13) needs
its operator spaces specified. Reading
\(a_*^TK(K+sN)^{-1}Na_*\) with a full unconstrained inverse gives a
different transfer in this toy. Interpreting its left factor as
\(a_*^TKP\) in the homogeneous correction space makes it zero, by static
orthogonality. Neither substitution represents the response term here.
Use the intermediate energy expression, with \(a=a_*+Pu\), directly:

\[
Y_{\mathrm{response}}=a_*^TN(a_*+Pu)
=m-s\ell^T(K_{ff}+sN_{ff})^{-1}\ell,\quad
m=a_*^TNa_*,\quad\ell=P^TNa_*.
\]

The checker stores the literal full-product discrepancy; its nonzero value
is an explicit warning against silently changing the boundary problem.
In 3D, curl-curl kernels, gauges, air conductivity zeros and topology require
the corresponding admissible spaces; positive definiteness on every stored
degree of freedom cannot simply be assumed.

## 3. Energy positivity and floating-point safeguards

For symmetric nonnegative energy forms, \(x^TKx\) and \(x^TNx\) are
nonnegative in exact arithmetic. Where the particular recurrence defines
\(L=x^TKx\) or \(R=1/(x^TNx)\), a *nonzero* mode with a positive norm
gives a positive element. A zero norm requires termination or nullspace
handling; its reciprocal is not a finite positive resistance.

An ordinary unshifted RL circuit assembled from positive elements is passive.
For the toy port, a direct proof also follows from
\(Y_B=x^*Kx/s+x^*Nx\) for unit boundary trace and the free equilibrium
equations: \(\operatorname{Re}(1/s)>0\) when
\(\operatorname{Re}s>0\). This structural argument is stronger than a
frequency sample, but its assumptions still have to hold for an actual FE
discretization and retained circuit.

Positive input matrices are not a floating-point accuracy certificate.
Finite-precision implementation checks should separately expose symmetry,
original-equation residuals, energy norms and their scale, mode
orthogonality, boundary traces, and recurrence breakdown. Report negative or
numerically unresolved norms instead of taking absolute values or clipping
them. Residual and orthogonality tolerances must be declared for a particular
validation; this note does not introduce or relax any acceptance threshold.
Orthogonality loss can spoil accuracy even while reported element values
remain positive. Frequency samples alone do not prove positive-realness on
the entire right half-plane.

The Euclidean algorithm is an algebraic division procedure. It does not
guarantee positive circuit coefficients for arbitrary rational functions.
For example, dividing \(s-1\) by \(s+1\) gives quotient 1 and remainder
\(-2\); the ratio is negative at DC. Changing remainder signs without
compensating elsewhere changes the represented function. Passivity requires
the function class, port convention and synthesis structure, not a chosen
remainder sign.

## 4. Positive-real shifts need their original physical variable

The existing [expansion-point notebook](01_expansion_and_energy.ipynb)
already distinguishes Type1 expansion of \(Z/s\) from Type2 expansion of
\(Z\), with \(q=s-s_0\). Preserve that distinction. A branch \(qL\)
becomes \(sL-s_0L\) in physical \(s\); positive \(L\) in the shifted
variable does not by itself imply a passive physical circuit. The exact
counterexample \(s_0=L=1\) has negative DC value.

The observation that a Type1 energy quantity involving division by \(s\)
must be used should therefore be
checked against the precise port, phasor convention and analytic transfer
function. Complex average power contains conjugates and cannot be blindly
treated as an analytic function of \(s\). This note does not claim that
dividing any transfer function by \(s\) universally restores passivity, or
that it proves a shifted finite ladder passive. Existing shifted studies
retain their explicitly limited scope.

## 5. Circular-wire E–H, A and T correspondence

For a homogeneous wire, the physical Laplace-domain radial equations are

\[
\frac1r(rH_\theta)'=J_z,\qquad E_z'=sB_\theta.
\]

The checker sets \(a=\mu=\sigma=1\) and applies the radial integration
operators to recurrence modes. For an electric correction the stage
multiplier is \(1/L\), rather than the physical phasor multiplier \(s\).

Regularity at the axis fixes
\(H_\theta=r^{-1}\int_0^r tJ_z(t)dt\). Its surface value is determined by
the total current; it cannot be independently imposed for an arbitrary
fixed \(J_z\). The homogeneous integration freedom is \(C/r\), not an
additive constant. For \(J_z=1\), replacing \(H=r/2\) by \((r-1)/2\)
to impose a zero surface value introduces an Ampere defect \(-1/(2r)\).
Compatible zero-total-current modes can make that adjustment vanish exactly;
that special compatibility must not be assumed for arbitrary inputs.

The checker reconstructs A from \(-A'=H\), with homogeneous A trace, and T
from \((rT)'/r=E\), retaining the impressed/history contribution in H.
It verifies the exact energy orthogonality of five electric and four magnetic
modes. Their coefficients are

\[
R_{2k}=(2k+1)R_{dc}\ (0\le k\le4),\qquad
L_{2k+1}=L_{dc}/(k+1)\ (0\le k\le3),
\]

where \(R_{dc}=1/(\pi\sigma a^2)\) and
\(L_{dc}=\mu/(8\pi)\), both per unit length. The nine-element unshifted
Type2 truncation agrees through \(z^8\), \(z=s\mu\sigma a^2\), with an
independently formed power series of
\(\sqrt z I_0(\sqrt z)/(2I_1(\sqrt z))\).
The oracle uses the defining modified-Bessel series in
[NIST DLMF 10.25.2](https://dlmf.nist.gov/10.25.E2).
This checks the circular-wire algebra, not general finite-element
equivalence, high-frequency accuracy, or the homogenized unit-cell example.

Run `python validation/input_ports_exact.py --output docs/data/input_ports_exact.json`.
No original supplied notebook is executed by this command.

## 6. Independence and hash conventions

A/H and T/E differentiation identities follow from how their potentials are
constructed. They check signs, boundary traces and impressed/history terms;
they are not independent physical or FE solutions. Likewise the static lift
is solved to satisfy its free equations. Independent comparisons are the
boundary Schur response versus the full inverse, element recurrences versus
scalar rational extraction, and wire-field recurrence versus the defining
Bessel series. An injected terminal \(z^8\) error must fail the last comparison.

The record's `source_sha256_lf` hashes UTF-8 source after removing an optional
BOM and normalizing CRLF/CR to LF, preserving all other text and the final
newline. It is not a hash of raw Windows worktree bytes. For this producer
those bytes equal the LF Git blob. Submission manifests separately label
`git_blob_sha256` (bytes from `git cat-file blob SHA:path`) and
`worktree_raw_sha256` (unmodified on-disk bytes). Private evidence-file hashes
always identify raw artifact bytes. A reviewer must not compare a CRLF
worktree hash with a claimed LF/blob hash.

CI installs the declared SymPy version in its workflow so that saved symbolic
string formatting remains reproducible. Identity checks remain explicit
exceptions under Python optimization; `python -O` does not disable them.
