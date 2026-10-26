"""Cited input-port and energy guidance; static data, no solver execution."""
import json


_REFERENCES = {
    "field_recursion": "https://doi.org/10.1109/TMAG.2017.2743224",
    "unit_cell_B_input": "https://doi.org/10.1109/TMAG.2024.3486683",
    "Bessel_definition": "https://dlmf.nist.gov/10.25.E2",
}

_PORTS = {
    "schema": "cln.knowledge.input-ports.v1",
    "conditions": ["same conjugate scalar port", "same normalization and admissible spaces",
                   "unshifted energy recurrence", "nonzero corresponding mode norms"],
    "matches": [{"quantity": kind, "B_index": index, "H_index": index + 2}
                for kind, index in [("L", -1), ("R", 0), ("L", 1), ("R", 2), ("L", 3), ("R", 4)]],
    "terminal_relation": "Y_B(s) * Z_H(s) = 1 for the matched scalar conjugate port",
    "B_response": "Y_B = c/s + Y_response; c = a_star^T K a_star; L_excitation = 1/c",
    "E_input": "Definition: Y_response=Y_B-c/s is the matched E-type response admittance; this defines a response term, not an independently solved E-input port or a claim about arbitrary E excitations.",
    "mismatched_ports": "Only different ports invalidate this index-shift comparison. A boundary condition's Dirichlet/Neumann label alone does not identify the port.",
    "toy": {"K": [[4, 1, 1], [1, 3, 1], [1, 1, 2]], "N_diagonal": [1, 2, 3],
            "boundary_index": 0, "unit_boundary_value": 1,
            "matched_L": "5/17", "matched_R": "25/39", "different_port_free_load": [1, 1],
            "different_port_L": "3/5"},
    "references": _REFERENCES,
}

_WIRE = {
    "schema": "cln.knowledge.wire-correspondence.v1",
    "conditions": "homogeneous round wire, regular axis, conductor interior, zero shift; per unit length",
    "scales": {"R_dc": "1/(pi*sigma*a^2)", "L_dc": "mu/(8*pi)", "z": "s*mu*sigma*a^2"},
    "R_even_over_Rdc": [1, 3, 5, 7, 9],
    "L_odd_over_Ldc": ["1", "1/2", "1/3", "1/4"],
    "exact_transfer": "Z/R_dc = sqrt(z)*I0(sqrt(z))/(2*I1(sqrt(z)))",
    "check": "The nine-element unshifted truncation has Bessel-series contact through z^8; this is local series contact, not all-frequency accuracy.",
    "field_maps": "Physical equations: (r*H_theta)'/r=J_z, E_z'=s*B_theta. A/H and T/E mode maps must retain impressed/history terms and the same traces.",
    "axis": "H_theta=(1/r)*integral_0^r(t*J_z(t) dt). Its surface value follows from total current; the homogeneous freedom is C/r, not an additive constant.",
    "limits": "Does not establish general 3D A-phi/T-Omega/A-T equivalence, shifted passivity, or continuum FE convergence.",
    "references": _REFERENCES,
}

_WIRE["polynomial_family"] = {
    "name": "radial Zernike polynomials, normalized by R_n^m(1)=1",
    "electric": "E_(2k)(rho)=R_(2k)^0(rho)=P_k(2*rho^2-1)",
    "magnetic": "H_(2k+1)(rho)=R_(2k+1)^1(rho)/(2*pi*a)",
    "Jacobi_form": "R_(m+2k)^m(rho)=rho^m*P_k^(0,m)(2*rho^2-1)",
    "norm": "integral_0^1 rho*R_n^m(rho)*R_j^m(rho) drho=delta_nj/(2*(n+1)) for fixed m and admissible indices",
    "elements": "Disk energy measure gives R_(2k)=(2*k+1)*R_dc and L_(2k+1)=L_dc/(k+1)",
    "vector_symmetry": "m=1 labels the radial polynomial of H_theta. Axisymmetric H_theta*e_theta has sin/cos angular factors in Cartesian components; this is not a nonaxisymmetric scalar H_theta excitation.",
    "reference": "https://reference.wolfram.com/language/ref/ZernikeR.html",
    "execution_status": "identification from definitions and energy norms; new Mathematica built-in correspondence check pending",
}

_SLAB = {
    "schema": "cln.knowledge.symmetric-slab.v1",
    "conditions": "homogeneous slab -a<x<a; identical surface electric drive; even E, odd H; zero shift; no edge/end effects",
    "port": {"current": "K=integral_-a^a sigma*E_z dx=2*H_y(a) [A/m]",
             "transfer": "Z_sheet=E_z(a)/K [ohm]",
             "terminal_scaling": "Z_terminal=(length/width)*Z_sheet, neglecting edges and ends"},
    "scales": {"R_dc": "1/(2*a*sigma) [ohm]", "L_dc": "mu*a/6 [henry]", "z": "s*mu*sigma*a^2"},
    "exact_transfer": "Z_sheet/R_dc=sqrt(z)*coth(sqrt(z)); removable limit at z=0 is 1",
    "fields": {"kappa": "sqrt(s*mu*sigma)",
               "E_z": "E_surface*cosh(kappa*x)/cosh(kappa*a)",
               "H_y": "E_surface*sigma*sinh(kappa*x)/(kappa*cosh(kappa*a))"},
    "R_even_over_Rdc": [1, 5, 9, 13, 17],
    "L_odd_over_Ldc": ["1", "3/7", "3/11", "1/5"],
    "mode_normalization": "a=mu=sigma=1; E_(2k)=P_(2k)(x), H_(2k+1)=P_(2k+1)(x)/2; full thickness integral dx",
    "energy_argument": "integral_-1^1 P_n(x)^2 dx=2/(2*n+1), hence R_(2k)=(4*k+1)/2 and L_(2k+1)=1/(2*(4*k+3))",
    "potential_maps": "H=-A' with A(1)=A(-1)=0; T'=E with odd T; retain history terms when forming total H",
    "verification_status": "analytic definition and Legendre derivation; new Mathematica/FE execution not performed; no all-band accuracy or timing claim",
    "references": {"Legendre_norm": "https://dlmf.nist.gov/18.3", "polynomial_relations": "https://dlmf.nist.gov/18.9", "field_recursion": _REFERENCES["field_recursion"]},
}

_TOPICS = {
    "overview": """# Input ports, boundary lifts and passivity
Use this guidance when interpreting B/H/E-input CLN, the unit-cell paper,
positive-energy element arguments or the E-H/A/T round-wire derivation.
The matched B/H case has an exact two-index coefficient shift. The question
is which conjugate port, spaces and normalization are being held fixed.
Topics ports, wire, slab and radial_modes return structured JSON; the other topics explain the
assumptions and failure modes. This knowledge server does not run those
derivations or certify a solver merely by returning this text.
References: Kameari et al. (2018), doi:10.1109/TMAG.2017.2743224;
Hiruma, Takahashi and Matsuo (2024), doi:10.1109/TMAG.2024.3486683.
""",
    "ports": json.dumps(_PORTS, indent=2),
    "boundary_lift": """# Nonzero boundary data and homogeneous corrections
Write a=a_star+P*u, where a_star carries the prescribed trace and P injects
zero-trace unknowns. The static lift satisfies P^T*K*a_star=0. Then
(K_ff+s*N_ff)*u=-s*P^T*N*a_star. Keep the static excitation separate from
later homogeneous corrections; changing only the boundary flag mid-recurrence
does not preserve normalization or orthogonality automatically.
Use Y_response=a_star^T*N*(a_star+P*u). The last compact product in the 2024
paper's eq13 needs its operator spaces specified: a literal full inverse can
give a different transfer, whereas a_star^T*K*P vanishes in the homogeneous
space. The intermediate energy expression and direct boundary Schur solve
avoid that ambiguity. Static K-orthogonality is not N-orthogonality. Gauge,
air conductivity zeros and topology still need their admissible spaces.
Reference: doi:10.1109/TMAG.2024.3486683, eqs7-15 and appendix.
""",
    "homogenization": """# B-input unit-cell homogenization
Hiruma, Takahashi and Matsuo (2024) use a linear spatially periodic unit cell
to replace a resolved fine conductor/air structure by a frequency-dependent
effective magnetic material. A prescribed external flux defines the B-input
port. Solve its static lift first, then generate homogeneous response modes;
the excitation inductance remains as the separate parallel c/s admittance.
For their uniform-input energy definition, nu_eff(s)=s*Y_B(s)/Q_B, where
Q_B=integral_cell |B0|^2 dV (or the declared per-length 2D analogue), and
mu_eff=1/nu_eff. This is the scalar response along the imposed B0, not a
permeability tensor. The geometry, imposed-flux normalization, phasor convention
and units must be recorded; a generic terminal impedance cannot simply be
relabeled effective permeability. This describes the paper's definition,
not a newly implemented homogenization solver.
The field-energy construction gives nonnegative ordinary RL parameters under
its operator assumptions, whereas a generic numerical continued-fraction
conversion need not preserve them. This is not a universal speed or accuracy
ranking over conversion/projection methods. The paper uses an electric-field
reconstruction estimator to select stage count for a stated operating band.
Do not treat that as mesh convergence, arbitrary-geometry certification or
a transferable acceptance threshold. FE accuracy and reduction error remain
separate; 3D semidefinite curl-curl operators require admissible/gauged modes.
Source: doi:10.1109/TMAG.2024.3486683, sectionsII-III, eqs1,7-21 and appendix.
""",
    "passivity": """# Positive energy is a structural argument with assumptions
For symmetric nonnegative energy forms, nonzero positive-energy modes give
positive ordinary unshifted RL elements. A zero norm means exhaustion or a
nullspace issue: stop or resolve it, rather than dividing by zero. Never
clip a negative norm or take its absolute value to manufacture passivity.
The Euclidean algorithm alone does not ensure positivity: dividing s-1 by
s+1 leaves remainder -2 and its ratio has negative DC value. A chosen
remainder sign is not a physical passivity proof.
Floating-point checks must separately report true original-equation residuals,
energy scales, traces, orthogonality and breakdown. Positive coefficients do
not establish field accuracy. Sampled frequency positivity is not a proof
on the full right half-plane. This guidance changes no acceptance threshold.
Reference: doi:10.1109/TMAG.2024.3486683; positive-energy reasoning assumes the
actual constrained/gauged operators satisfy the stated hypotheses.
""",
    "shifted_energy": """# Type1, Type2 and the physical Laplace variable
Distinguish Type1 expansion of Z/s from Type2 expansion of Z and state the
port and normalization. For a positive real shift s0, q=s-s0: a branch q*L
is s*L-s0*L in physical s. Positive q-elements therefore do not alone certify
a passive physical circuit. s0=L=1 gives a negative physical DC branch.
An energy quantity involving division by s must be tied to the declared
analytic transfer. Complex average power contains conjugates and must not
be substituted blindly into analytic continuation in s. Dividing an arbitrary
transfer by s is not a general passivity restoration theorem. The 2024
unit-cell paper does not itself validate every shifted Type1 construction.
Check the intended finite circuit and operating-band error separately.
""",
    "wire": json.dumps(_WIRE, indent=2),
    "slab": json.dumps(_SLAB, indent=2),
    "radial_modes": json.dumps(_WIRE["polynomial_family"], indent=2),
    "geometry_ports": """# Match geometry measures before comparing ladders
Round wire: I=2*pi*a*H_theta(a) [A], so Z_per_length=E_z(a)/I [ohm/m].
The circumference factor a is essential: E/(2*pi*H) is not the same per-length port.
Symmetric slab: K=2*H_y(a) [A/m], so Z_sheet=E_z(a)/K [ohm].
Recover a terminal impedance using length for the wire, length/width for the slab.
For a slab, use the full thickness -a..a (or a half-domain with its factor two),
even E and odd H. An arbitrary constant in H violates the odd symmetry.
For the wire, the regular-axis integral fixes H; an additive boundary offset
is incompatible with cylindrical Ampere, whose homogeneous freedom is C/r.
Geometry-specific quadrature and DC scales are necessary; replacing unknown
parameters by 1 or reusing cylindrical defaults can conceal a normalization error.
Analytic E-H/A/T identities in these 1D settings do not rank general 3D FE methods.
""",
    "evidence": """# Evidence and open questions
Distinguish a literature result, a checked special-case identity, a numerical
validation and a hypothesis. The unit-cell B-input construction is cited to
Hiruma et al. (2024); field recursion is cited to Kameari et al. (2018).
Exact toy checks compare boundary Schur elimination to a full rational inverse,
and B/H recurrences to scalar Laurent extraction. The wire recurrence is
checked against the defining Bessel series. A/H and T/E differentiation maps
are construction-consistency checks, not independent FE solutions.
Returning this knowledge does not execute those checks. General shifted
passivity, arbitrary geometry and full numerical accuracy need their own
proof or validation. The informal questions motivate these distinctions
but are not themselves new theorems.
""",
}
_TOPICS["all"] = "\n\n".join(_TOPICS.values())


def get_input_port_documentation(topic: str = "all") -> str:
    """Return cited guidance without acquiring a solver.

    Topics:
      overview - assumptions and navigation
      ports - matched B/H indices and E-type response, as JSON
      boundary_lift - static excitation and homogeneous correction spaces
      homogenization - unit-cell B-input and effective magnetic material
      passivity - energy positivity, Euclid and floating-point breakdown
      shifted_energy - Z/s, shifted branches and analytic-transfer caveats
      wire - E-H/A/T correspondence and Bessel oracle, as JSON
      slab - symmetric slab analytic port, modes and coefficients, as JSON
      geometry_ports - wire/slab measures, units and boundary assumptions
      radial_modes - radial Zernike/Jacobi identification of circular modes, as JSON
      evidence - independent checks, construction identities and limits
      all - all sections
    """
    topic = topic.strip().lower()
    if topic in _TOPICS:
        return _TOPICS[topic]
    raise ValueError("Unknown topic '" + topic + "'. Available: " + ", ".join(sorted(_TOPICS)))
