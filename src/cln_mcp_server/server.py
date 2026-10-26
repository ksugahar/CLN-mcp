"""MCP server for the Cauer Ladder Network (CLN): theory, algorithms, pitfalls, benchmarks."""
import json
from typing import Annotated
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from . import core_knowledge as K
from .diffusion_tools import MAX_SYMBOLIC_STAGES

mcp = FastMCP("cln", instructions=(
    "Knowledge server for the Cauer Ladder Network (CLN) with expansion point. "
    "Start with cln_overview. The numerical implementation is the `cln` Python "
    "module (NGSolve / radia-ngsolve). cln_diffusion_physical_model derives bounded "
    "matrix-based energy models; symbolic derivation uses Mathematica. "
    "Use cln_diffusion_energy_formulation for assumptions, coordinates and energy units."))


@mcp.tool()
def cln_overview() -> str:
    """What CLN is, the ladder form with expansion point s0, and where things live."""
    return K.OVERVIEW


@mcp.tool()
def cln_algorithm(formulation: str = "T-Omega") -> str:
    """Stage-by-stage algorithm. formulation: 'T-Omega' (only one available now)."""
    if formulation.replace("_", "-").lower() not in ("t-omega", "t-ω"):
        raise ValueError(f"formulation '{formulation}' not documented; available: T-Omega")
    return K.ALGORITHM_T_OMEGA


@mcp.tool()
def cln_boundary_conditions() -> str:
    """Voltage drive (Neumann) vs current drive (Dirichlet) and the matching Omega gauge."""
    return K.BOUNDARY_CONDITIONS


@mcp.tool()
def cln_pitfalls() -> str:
    """Known implementation mistakes and their symptoms."""
    return K.PITFALLS


@mcp.tool()
def cln_wire_benchmark() -> str:
    """Round-wire benchmark: exact elements, exact impedance, stored reference results."""
    return json.dumps(K.WIRE_BENCHMARK, ensure_ascii=False, indent=1)


@mcp.tool()
def cln_references_and_docs() -> str:
    """Literature and the tutorial notebooks shipped in docs/."""
    return json.dumps({"references": K.REFERENCES, "docs": K.DOCS}, ensure_ascii=False, indent=1)


_MODULES = {
    "input_ports": ("input_ports", "get_input_port_documentation"),
    "basics": ("cln_knowledge", "get_cln_documentation"),
    "multiport": ("cln_multiport_knowledge", "get_cln_multiport_documentation"),
    "advanced": ("cln_advanced_knowledge", "get_cln_advanced_documentation"),
    "specialty": ("cln_specialty_knowledge", "get_cln_specialty_documentation"),
    "3d": ("cln_3d", "get_cln_3d_documentation"),
    "bem": ("bem_cln", "get_bem_cln_documentation"),
    "sibc_orthogonal": ("cln_sibc_orthogonal", "get_cln_sibc_orthogonal_documentation"),
    "sphere_dd": ("cln_sphere_dd", "get_cln_sphere_dd_documentation"),
    "from_radia_mcp": ("from_radia_mcp", "get_from_radia_mcp"),
}


@mcp.tool()
def cln_knowledge(area: str = "index", topic: str = "") -> str:
    """Extended CLN knowledge base (theory, lab lessons, citations).

    area: 'index' (list areas) or one of basics, multiport, advanced, specialty,
          3d, bem, sibc_orthogonal, sphere_dd, from_radia_mcp, input_ports.
    topic: passed to the area's dispatcher (empty = the area's default/overview).
    3d: overview, formulas, notebooks, list, or a listed script name.
    bem: all, overview, 2d, 3d, notebooks. sibc_orthogonal: list or a section.
    sphere_dd: all or overview. Unknown selections raise an MCP error.
    """
    import importlib
    if area == "index":
        return "areas: " + ", ".join(_MODULES)
    if area not in _MODULES:
        raise ValueError(f"Unknown knowledge area '{area}'; available: " + ", ".join(_MODULES))
    mod, fn = _MODULES[area]
    f = getattr(importlib.import_module(f"cln_mcp_server.knowledge.{mod}"), fn)
    return f(topic) if topic else f()


@mcp.tool()
def cln_notebook(name: str = "") -> str:
    """Tanimoto's CLN scripts. Empty name lists them.

    2D and production are runnable scripts. AT, T_Omega and APhi return pointer text to the
    maintained public implementation (CLN-mcp validation/cln3d), not runnable code.
    """
    from cln_mcp_server.knowledge.cln_notebooks import get_notebook, list_notebooks
    return get_notebook(name) if name else str(list_notebooks())


@mcp.tool()
def cln_diffusion_formulation(physics: str = "heat") -> str:
    """Physical CLN for -div(a grad u) + b du/dt = f: algorithm, circuit topology, units.

    physics: heat | mass_diffusion | groundwater | magnetic_diffusion
    """
    from .diffusion_tools import formulation
    return formulation(physics)


@mcp.tool()
def cln_diffusion_derive(geometry: str = "cylinder",
                         nstages: Annotated[int, Field(strict=True, ge=1, le=MAX_SYMBOLIC_STAGES)] = 5) -> str:
    """Symbolic RC-CLN elements for slab/cylinder/disk/sphere; integer nstages 1..8, via Mathematica."""
    from .diffusion_tools import derive
    return derive(geometry, nstages)


@mcp.tool()
def cln_diffusion_energy_formulation(physics: str = "heat") -> str:
    """Physical CLN derivation, energy identities, circuit/field coordinates and PRIMA distinction."""
    from .energy_tools import energy_formulation
    return energy_formulation(physics)


@mcp.tool()
def cln_diffusion_physical_model(stiffness: list[list[float]], capacity: list[list[float]],
                               load: list[float], nstages: int = 6, physics: str = "heat",
                               storage_integral: list[float] | None = None,
                               nodes: list[float] | None = None, drive: float = 0.0,
                               initial_field: list[float] | None = None) -> str:
    """Derive an energy-certified RC CLN from K u + M udot = load F (1..256 DOFs).

    Real SPD matrices after fixed-DOF elimination; one collocated port. Returns
    positive R/C, full capacitor-terminated circuit matrices/netlist, even/odd
    reconstruction bases and measured material-weighted certificates. Supply
    int(b*phi_j) as storage_integral for actual reconstructed heat content.
    Optional node coordinates and drive evaluate dynamics and energy balances.
    initial_field is in original free-DOF coordinates. Returns its capacity-metric
    projection, lost storage, and a MATLAB ss factory carrying projected x0.
    Requires CLN-mcp[energy] (NumPy), not NGSolve or Mathematica.
    """
    from .energy_tools import physical_model
    return physical_model(stiffness, capacity, load, nstages, physics,
                          storage_integral, nodes, drive, initial_field)


@mcp.tool()
def cln_export_lti(resistance: list[float], storage: list[float], kind: str = "rc",
                   termination: str = "auto", s0: float = 0.0, length: float = 1.0,
                   quantity: str = "impedance", name: str = "cln_model") -> str:
    """Export CLN elements as LTspice Laplace subcircuits and a Simulink ss model.

    Ordered arrays: RC R0,R2,... / C1,C3,...; RL R0,R2,... / L1,L3,... .
    RC defaults to full capacitor termination, already scaled physical elements.
    RL infers termination from element counts; length scales per-length R/L.
    quantity='impedance' maps current/flux to voltage/potential; RL admittance
    is voltage to current and is required for inductance-terminated RL ladders.
    Returns A/B/C/D, pole/stability diagnostics and .m/.lib file CONTENTS.
    No file writing, MATLAB launch or polynomial fit. Up to 64 R/L/C elements.
    """
    from .export_tools import export_model
    return export_model(resistance, storage, kind, termination, s0, length, quantity, name)


@mcp.tool()
def cln_tiles(resistance: list[float] = (), inductance: list[float] = (), s: float = 0.0, s0: float = 0.0,
              ladder: str = "type2", euclid: list[int] = (), title: str = "") -> str:
    """Tile picture (SVG text) of a ladder or of Euclid's algorithm (EMT2017-65).

    Each resistor is a rectangle of width = current, height = voltage (aspect R : 1); series stacks,
    parallel sits side by side.  Orange = resistive (Joule), blue = inductive ((s - s0) L at real s > s0).
      type2: resistance = [R0, R2, ...], inductance = [L1, L3, ...]
      type1: resistance = [L-1, R0, L1, R2, ...] (the Type1 element list), inductance unused
      euclid = [p, q]: unit-resistor network of p/q ohm; its tiles are Euclid's squares.
    Returns {"svg", "impedance", "tiles"} as JSON.  At most 64 elements.
    """
    import json
    from cln.tiles import ladder_network, euclid_network, svg, impedance, layout
    if ladder not in ("type2", "type1"):
        raise ValueError("ladder must be 'type2' or 'type1'")
    if euclid:
        if len(euclid) != 2 or any(isinstance(v, bool) or int(v) != v for v in euclid):
            raise ValueError("euclid needs exactly two positive integers [p, q]")
        p, q = (int(v) for v in euclid)
        net, cur = euclid_network(p, q), float(q)      # rejects > 64 tiles before building any
    else:
        if not 1 <= len(resistance) + len(inductance) <= 64:
            raise ValueError("give 1..64 ladder elements")
        if ladder == "type2" and len(resistance) not in (len(inductance), len(inductance) + 1):
            raise ValueError("Type2 alternates R0, L1, R2, ...: len(resistance) must equal "
                             "len(inductance) or len(inductance) + 1")
        if ladder == "type2":
            R = {2*i: float(v) for i, v in enumerate(resistance)}
            L = {2*i + 1: float(v) for i, v in enumerate(inductance)}
            net = ladder_network(R, L, s, s0)
        else:
            net = ladder_network([float(v) for v in resistance], {}, s, s0, ladder="type1")
        cur = None
    z = impedance(net)
    tiles = [dict(label=leaf.label, kind=leaf.kind, x=x, y=y, width=w, height=h)
             for x, y, w, h, leaf in layout(net, current=1.0 if cur is None else cur)]
    return json.dumps(dict(svg=svg(net, title=title, current=cur), impedance=z, tiles=tiles))


def main():
    # Load the optional native numerical runtime before stdio reader threads
    # start. On Windows, first importing NumPy while a reader holds the CRT
    # stream lock can deadlock in extension initialization.
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
    mcp.run()


if __name__ == "__main__":
    main()

