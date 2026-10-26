"""Bounded numerical derivation tools; no arbitrary code or file execution."""
import json

from .diffusion_tools import ANALOGIES


ENERGY_FORMULATION = """Physical diffusion CLN, not just a transfer-function fit
PDE: -div(a grad u) + b du/dt = f F, with a,b positive and homogeneous
coercive boundaries. FE coordinates: K u + M udot = load F, y=load^T u.
This implementation requires real SPD K,M and one collocated source/output.
Remove fixed DOFs before calling. A pure-Neumann zero mode requires separate
treatment; nonsymmetric advection, nonlinear coefficients and multiple ports
are not implemented by this scalar recurrence.

Construct q0=load, h_previous=0. At each stage:
  K t_i=q_i; R_i=t_i^T K t_i; e_i=t_i/R_i;
  h_i=h_previous+e_i; C_i=h_i^T M h_i;
  q_next=q_i-M h_i/C_i.
Even modes E=[e_i] are dissipation-orthogonal; odd modes H=[h_i] are
capacity-orthogonal. In the continuum, flux mode Qhat_i=-a grad(t_i),
R_i=int (1/a)|Qhat_i|^2 and C_i=int b h_i^2. Thus the elements come from
material-weighted field integrals, and Qhat_i=-a R_i grad(e_i).
These volume-only formulas assume no dissipative boundary term. For an
assembled Robin problem, K and the dissipation also include its boundary form.

B has +1 on the diagonal and -1 on its upper diagonal. E=H B.
Node coordinates v and capacitor drops w=Bv are DIFFERENT coordinates:
  u=E v=H w; flux=-a grad(u)=sum (v_i/R_i) Qhat_i.
Certificates: E^T K E=G=diag(1/R), H^T M H=diag(C),
E^T M E=M_c=B^T diag(C) B, E^T load=[1,0,...]^T.
Circuit: M_c vdot+G v=port F; y=v[0]. Shunt resistors, series capacitors,
and the final capacitor to reference. This full capacitor termination retains
all stored modes. The older rc_ladder_impedance is RESISTOR-terminated and
omits the last capacitor; it is not this Galerkin energy model.

W=1/2 int b u^2=1/2 sum C_i w_i^2;
D=int a|grad u|^2=sum v_i^2/R_i; dW/dt+D=F*y.
Also: d(1/2 u^T K u)/dt+udot^T M udot=F*ydot.
Both identities are certified on the retained subspace, not a guarantee that
truncation reproduces every full-order frequency or arbitrary initial field.

HEAT: u is excess temperature. W has units J K, D and F*y have units W K.
Do NOT call W heat in Joules or F*y injected heating power. Actual reconstructed
excess heat is U=int rho*c*u [J], a separate LINEAR functional. Supply the
storage_integral vector int b phi_j, or use DiffusionCLN.energy_model, to obtain
it. Do not replace it by sum(C_i*w_i). W/T_ref is a near-equilibrium availability
approximation for small temperature deviations, not a universal equality.
For other physics, attach physical units/constitutive meaning to each metric;
for magnetic vector-potential diffusion the rate-tested identity can represent
magnetic energy and Joule loss. The code does not assign Joules universally.

PRIMA can preserve passivity and quadratic storage by congruence projection.
For the same Krylov subspace, the full CLN and a PRIMA-type congruence have the
same transfer function; there is no accuracy advantage by changing coordinates.
The additional deliverable here is the material-normalized alternating field
basis, positive R/C elements, explicit RC topology, field reconstruction and
the measured field/circuit certificates. PRIMA is not alleged to violate energy.
References: Tanimoto et al., IEEJ SA-25-014/RM-25-014 (2025);
Odabasioglu, Celik, Pileggi, PRIMA, IEEE TCAD 17(8), 645-654 (1998),
DOI 10.1109/43.712097.
"""


def energy_formulation(physics="heat"):
    if physics not in ANALOGIES:
        raise ValueError("physics must be one of " + ", ".join(ANALOGIES))
    return ENERGY_FORMULATION + "\n\nConstitutive labels: " + json.dumps(ANALOGIES[physics])


def physical_model(stiffness, capacity, load, nstages=6, physics="heat",
                   storage_integral=None, nodes=None, drive=0.0, initial_field=None):
    """Return a JSON energy model for a bounded dense assembled diffusion system."""
    if physics not in ANALOGIES:
        raise ValueError("physics must be one of " + ", ".join(ANALOGIES))
    if not 1 <= len(load) <= 256:
        raise ValueError("MCP supports 1..256 full-order DOFs; use the Python/NGSolve API for larger models")
    n = len(load)
    if (len(stiffness) != n or len(capacity) != n or
            any(len(row) != n for row in stiffness) or any(len(row) != n for row in capacity)):
        raise ValueError("stiffness/capacity must be square matrices matching the load")
    if isinstance(nstages, bool) or not isinstance(nstages, int) or not 1 <= nstages <= 64:
        raise ValueError("MCP nstages must be an integer in 1..64")
    try:
        from cln.energy import physical_cln
    except ImportError as exc:
        raise RuntimeError('Install the numerical tool dependency with pip install "CLN-mcp[energy]"') from exc
    model = physical_cln(stiffness, capacity, load, nstages, storage_integral=storage_integral)
    out = model.to_dict()
    out["summary"] = {
        "requested_stages": nstages, "effective_stages": int(model.R.size),
        "stop_reason": model.metadata["stop_reason"],
        "notice": ("Fewer stages retained than requested; inspect rejected_larger_prefixes and verify full-order accuracy."
                   if model.R.size < nstages else
                   "Retained-subspace identities certified; full-order approximation accuracy requires separate validation."),
    }
    out["physics"] = physics
    out["normalization_note"] = (
        "load is used as supplied; if F means total injected heat/flux, assemble from a unit-integral "
        "continuum source. The sum of free-DOF loads is not that continuum integral.")
    out["interpretation"] = {
        "quadratic_storage": "material-weighted quadratic functional, not universally thermodynamic energy",
        "linear_storage": "integral b*u, only supplied when storage_integral is given",
        "source": "one collocated port; homogeneous boundary data; real SPD operators",
        "reconstruction": "even_basis times node coordinates; odd_basis times capacitor drops",
        "stages": "effective_stages may be lower than requested after invariant-space breakdown or energy-audit failure; inspect stop_reason and rejected_larger_prefixes",
    }
    if physics == "heat":
        out["units"] = {"R": "K/W", "C": "J/K", "quadratic_storage": "J K",
                        "dissipation_and_supply": "W K", "linear_storage_content": "J"}
    else:
        out["units"] = {"note": "determined by the supplied physical coordinates and spatial measure; no universal Joule label"}
    if nodes is not None:
        out["state"] = model.balance(nodes, drive)
    if initial_field is not None:
        lti = model.to_lti(initial_field=initial_field)
        out["initial_projection"] = lti.metadata["initial_projection"]
        out["initialized_lti"] = lti.to_dict()
        out["matlab_function"] = lti.matlab_function("cln_initial_model")
        out["initialization_note"] = "Capacity-metric projection onto retained modes; nonzero initial state requires state-space execution, not a Laplace element."
    return json.dumps(out, allow_nan=False)
