"""MCP tools for the physical CLN of scalar diffusion-type PDEs."""
import json
from numbers import Integral
import pathlib
import shutil
import subprocess

ANALOGIES = {
    "heat": {"u": "temperature T [K]", "a": "thermal conductivity k [W/(m K)]",
             "b": "volumetric heat capacity rho c [J/(m^3 K)]", "F": "injected power [W]",
             "R": "thermal resistance [K/W]", "C": "heat capacity [J/K]"},
    "mass_diffusion": {"u": "concentration c [mol/m^3]", "a": "diffusivity D [m^2/s]", "b": "1",
                       "F": "injected amount rate [mol/s]", "R": "diffusion resistance [s/m^3]",
                       "C": "storage [m^3]"},
    "groundwater": {"u": "hydraulic head h [m]", "a": "hydraulic conductivity K [m/s]",
                    "b": "specific storage S_s [1/m]", "F": "injected flow [m^3/s]",
                    "R": "flow resistance [s/m^2]", "C": "storage [m^2]"},
    "magnetic_diffusion": {"u": "vector potential A_z [Wb/m]", "a": "reluctivity 1/mu [m/H]",
                           "b": "conductivity sigma [S/m]", "F": "impressed current [A]",
                           "R": "inductance-like [H]", "C": "conductance-like [S]",
                           "note": "2-D eddy currents with an impressed current; the classic CLN of "
                                   "the field is the voltage-driven RL ladder (cln_algorithm)."},
}

FORMULATION = """\
PDE:   -div(a grad u) + b du/dt = f(r) F(t)   in Omega,   u = 0 on the Dirichlet boundary,
       f normalized to int f dOmega = 1, so F is the injected flux (power, flow, ...).
Port:  u_port = int f u dOmega; Z(s) = u_port / F. This pairing is conjugate
to the quadratic storage balance; for heat F*u_port has units W K, not W.

Algorithm (Tanimoto, Sugahara, Takahashi, Matsuo, IEEJ SA-25-014, 2025):
   That_{-1} = 0,  qhat_0 = f
   for n = 0, 1, ...
     solve  -div(a grad Tt_2n) = qhat_2n               (static problem, u = 0 on boundary)
     R_2n    = int a |grad Tt_2n|^2                      [resistance]
     That_2n = Tt_2n / R_2n ;   That_{2n+1} = That_{2n-1} + That_2n
     C_2n+1  = int b That_{2n+1}^2                        [capacity]
     qhat_{2n+2} = qhat_2n - b That_{2n+1} / C_2n+1

Circuit (flux/current source F into node 1): R_2n from node 2n+1 to the reference,
C_2n+1 in series between nodes 2n+1 and 2n+3:
   Z(s) = 1/(1/R0 + 1/(1/(s C1) + 1/(1/R2 + 1/(1/(s C3) + ...))))
Exact properties (verified symbolically for slab / cylinder / sphere):
   int b That_{2i+1} That_{2j+1} = C_{2i+1} delta_ij,
   int a grad That_{2i} . grad That_{2j} = delta_ij / R_{2i},
   R0 = Z(0) (static resistance); Z(s) converges stage by stage to the exact transfer function.
Field reconstruction: let v_i be node coordinates and w_i=v_i-v_{i+1}, v_N=0.
u=sum_i v_i That_{2i}=sum_i w_i That_{2i+1}. Odd modes multiply capacitor drops,
not node coordinates. The full energy model retains the final capacitor to reference;
the resistance-terminated impedance above omits it. Use cln_diffusion_energy_formulation
and cln_diffusion_physical_model for certified energy matrices, units and RC netlists.
Implementations: cln.diffusion.DiffusionCLN (NGSolve, any mesh); cln_mcp_server/resources/DiffusionCLN.wl
(closed forms for slab/cylinder/sphere)."""

_PKG = pathlib.Path(__file__).resolve().parent / "resources" / "DiffusionCLN.wl"   # shipped in the wheel
MAX_SYMBOLIC_STAGES = 8


def formulation(physics: str = "heat") -> str:
    if physics not in ANALOGIES:
        raise ValueError("physics must be one of " + ", ".join(ANALOGIES))
    return FORMULATION + "\n\nQuantities for '" + physics + "':\n" + json.dumps(ANALOGIES[physics], indent=1)


def derive(geometry: str = "cylinder", nstages: int = 5) -> str:
    """Closed forms via Mathematica; at most eight symbolic stages per call."""
    d = {"slab": 1, "cylinder": 2, "disk": 2, "sphere": 3}.get(geometry)
    if d is None:
        raise ValueError("geometry must be slab, cylinder, disk or sphere")
    if isinstance(nstages, bool) or not isinstance(nstages, Integral) or not 1 <= nstages <= MAX_SYMBOLIC_STAGES:
        raise ValueError(f"nstages must be an integer from 1 to {MAX_SYMBOLIC_STAGES} for symbolic derivation")
    exe = shutil.which("wolframscript")
    if exe is None:
        raise RuntimeError("Mathematica wolframscript is not available")
    if not _PKG.is_file():
        raise FileNotFoundError("DiffusionCLN.wl package is not available")
    package_literal = _PKG.as_posix().replace('\\', '\\\\').replace('"', '\\"')
    code = (f'Get["{package_literal}"]; r = DiffusionCLN`DiffusionCLN[{d}, {nstages}, a, b, L]; '
            'Print[ToString[Simplify[r["R"]], InputForm]]; Print[ToString[Simplify[r["C"]], InputForm]];')
    out = subprocess.run([exe, "-code", code], capture_output=True, text=True, timeout=600)
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip().startswith("{")]
    if out.returncode or len(lines) < 2:
        raise RuntimeError("derivation failed:\n" + out.stdout + out.stderr)
    return json.dumps({"geometry": geometry, "R_2n": lines[0], "C_2n+1": lines[1],
                       "units": "a, b, L symbolic; slab per unit face area (both halves), "
                                "cylinder per unit length, sphere total"}, indent=1)
