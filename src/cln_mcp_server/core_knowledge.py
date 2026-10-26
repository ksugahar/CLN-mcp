"""Knowledge base of the CLN MCP server (plain data, no solver import)."""

OVERVIEW = """\
Cauer Ladder Network (CLN): model-order reduction of quasi-static eddy-current
fields. The impedance of a conductor is written as the Cauer continued fraction

    Z(s) = R0 + 1/( 1/((s-s0) L1) + 1/( R2 + 1/( 1/((s-s0) L3) + ... )))

whose elements R_{2n}, L_{2n+1} are obtained stage by stage from static-type FE
problems (no frequency sweep). Each stage also yields a field mode, so the
ladder is a reduced-order model of the field, not only of the impedance.

s0 = 0  : classic CLN (expansion at DC).
s0 > 0  : CLN with an expansion point; accuracy is concentrated around s0.
          In the documented conductor-only problem the magnetic mass term
          removes the zero-shift gradient kernel; air/topology kernels still
          require their own treatment. Positive shifted elements do not by
          themselves certify passivity in physical s.

This package implements CLN on NGSolve / radia-ngsolve (module `cln`).
For B/H/E input, matched index shifts, static boundary lifts and passivity,
use cln_knowledge(area="input_ports", topic="ports"), then boundary_lift,
homogenization, passivity, shifted_energy, wire and evidence in that same area.
"""

ALGORITHM_T_OMEGA = """\
T-Omega method, conductor-only domain, T = H inside the conductor.

  a(u,v) = (1/sigma curl u, curl v) + s0 (mu u, v)
  m(u,v) = (mu u, v)

 1: T0 : a(T0, N) = <drive, N>          (unit voltage; see boundary_conditions)
 2: for n = 0, 2, 4, ...
 3:   R_n     = 1 / a(T_n, T_n)                       (per unit length)
 4:   (s0=0)  gauge: (mu grad W, grad psi) = -(mu T_n, grad psi);  T_n <- T_n + grad W
 5:   H_{n+1} = H_{n-1} + R_n T_n,   L_{n+1} = m(H_{n+1}, H_{n+1})
 6:   T_{n+2} : a(T_{n+2}, N) = -m(H_{n+1}, N)/L_{n+1} + a(T_n, N)

Step 6 contains the accumulation T_{n+2} = T_n + T' (Algorithm 2, line 9 of the
s0 = 0 formulation) through the term a(T_n, N). Writing it this way, one formula
covers the documented conductor-only cases with matching gauges and port
normalization. The current-drive path's initial GridFunction carries the
prescribed-trace lift; _solve corrects its free residual. Later step fields
start with homogeneous trace. Changing boundary flags alone is not equivalent.
At the first Dirichlet update, a(T0,N)=0 for free test functions. The weak-form
accumulation in Step 6 does not mean adding T0's nonzero boundary trace to a
homogeneous correction or prescribing that initial trace at later stages.
"""

BOUNDARY_CONDITIONS = """\
Neumann (voltage drive): no Dirichlet on T. Drive by the natural boundary term
  <drive, N> = -int_lateral (E_t x N) . n dS,  E_t = (0,0,1) V/m.
  Kernel of a (s0 = 0) = grad(phi) with phi free on the WHOLE boundary
  -> the Omega gauge must have NO Dirichlet condition.

Dirichlet (current drive): n x T = I/(2 pi a) on the lateral surface, I = 1 A,
  then scale T0 by 1/R0 to unit voltage.
  Kernel of a (s0 = 0) = grad(phi) with phi = 0 on the lateral surface
  -> Omega = 0 on the lateral surface, preserving the prescribed tangential trace.

In both cases the end faces carry the natural condition n x E = 0.

Do not identify B/H/E input solely from a Dirichlet/Neumann label on different
potentials. B-input uses a static flux lift; later response modes have zero
trace. For the same conjugate scalar port and unshifted normalization,
L_-1^B=L_1^H, R_0^B=R_2^H, L_1^B=L_3^H, etc. hold exactly until exhaustion.
Different ports need not share coefficients. See input_ports/boundary_lift
and input_ports/ports via cln_knowledge, citing doi:10.1109/TMAG.2024.3486683.
"""

PITFALLS = """\
1. Omega gauge with Dirichlet on the lateral surface in the Neumann case
   -> the T-stage right-hand side is not in the range of the singular matrix;
   ICCG stalls (relative residual 0.1-0.8) and R, L are wrong.
2. Missing a(T_{n-2}, N) = (1/sigma J_{n-2}, curl N) in the s0 = 0 branch
   -> no accumulation. Adding T[n] += T[n-2] afterwards is equivalent, but only
   for n > 2 in the Dirichlet case; in the Neumann case n = 2 also needs it
   (otherwise R2/R0 = 0.75 instead of 3 on the wire benchmark).
3. The mass term of the previous stage must carry the factor s0
   (s0 mu T_{n-2} . N); without it the term is dimensionally wrong.
4. Calling the FE-space setup twice can silently override the boundary
   condition (e.g. always Dirichlet).
5. Impedance-map plots evaluated at s = 2 s0 divide by zero when s0 = 0.
6. Reaching an ICCG/refinement iteration limit is not convergence. The implementation
   checks the final residual of the original operator, records solve_diagnostics,
   and rejects an unconverged solve. Zero-shift ICCG never switches to a direct
   method. Select solver='direct' explicitly for regularized direct refinement;
   the original-system residual check still applies to that selected method.
7. Use the same quadrature for curved-element assembly and energy normalization.
   For the zero-shift scalar gauge, pin the constant without a mass perturbation
   and retain T+grad(Omega) without interpolation into another HCurl space.
8. Magnetic orthogonality uses m(H_i,H_j), not s0*m(H_i,H_j). The latter is
   identically zero at s0=0 and cannot verify orthogonality.
9. Positive shifted R/L elements do not alone certify global passivity in s.
   Check the exported poles and the actual j*omega band against the full model.
10. Energy positivity needs nonzero norms in the admissible spaces. A zero
    norm is exhaustion/nullspace handling, not a finite resistance; never
    clip negative norms or infer positivity from a Euclidean remainder sign.
11. Type1 expands Z/s, Type2 expands Z. Complex average power contains
    conjugates; calling it an analytic function of s can invalidate an
    energy/passivity argument. See input_ports/shifted_energy and passivity.
"""

WIRE_BENCHMARK = {
    "geometry": "round wire, radius a = 10 mm, segment length h = 10 mm, sigma = 1e6 S/m, mu = mu0",
    "exact_s0_zero": "R_{2k} = (2k+1)/(pi a^2 sigma),  L_{2k+1} = mu/(8 pi (k+1))   (per unit length)",
    "exact_impedance": "Z(s) = k/(2 pi a sigma) J0(ka)/J1(ka),  k^2 = -s mu sigma",
    "reference_results": {
        "note": "cln.TOmegaCLN, order 3, maxh a/2, conductor maxh a/10; s0=0: radia SparseSolv ICCG; s0>0: sparsecholesky (current API requires explicit solver=direct)",
        "s0=0 Neumann": {"R2/R0": 3.0000, "L3*8pi/mu": 0.5000, "R4/R0": 5.0000,
                         "L5*8pi/mu": 0.3333, "R6/R0": 7.0011, "L7*8pi/mu": 0.2501, "R8/R0": 9.0226},
        "s0=2pi*1e5 Neumann, n=8, |Z_ladder/Z_exact-1|": {"1e3 Hz": 5.9e-3, "1e5 Hz": 6e-6, "1e6 Hz": 2.8e-3},
        "s0=0 Neumann, n=8, |Z_ladder/Z_exact-1|": {"1e3 Hz": 0.0, "1e5 Hz": 6.2e-5, "1e6 Hz": 5.8e-2},
    },
}

REFERENCES = [
    "A. Kameari, H. Ebrahimi, K. Sugahara, Y. Shindo, T. Matsuo, \"Cauer Ladder Network "
    "Representation of Eddy-Current Fields for Model Order Reduction Using Finite-Element "
    "Method,\" IEEE Trans. Magn., vol. 54, no. 3, 2018, doi:10.1109/TMAG.2017.2743224.",
    "S. Hiruma, Y. Takahashi, T. Matsuo, \"Homogenization Method Based on Cauer "
    "Ladder Network Representation of Unit Cell,\" IEEE Trans. Magn., 60(12), "
    "2024, doi:10.1109/TMAG.2024.3486683.",
]

DOCS = {
    "docs/01_round_wire.ipynb": "T-Omega CLN of a round wire with expansion point: elements, direct-solve and analytic checks, LTspice netlist",
    "docs/02_diffusion_rc.ipynb": "current-source RC CLN of a uniformly heated disk: closed-form elements and exact transfer function",
}
