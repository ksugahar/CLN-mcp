"""Cauer Ladder Network (CLN) with expansion point, NGSolve implementation."""
from importlib import import_module
from ._version import __version__

_EXPORTS = {
    "TOmegaCLN": "t_omega", "ladder_impedance": "ladder",
    "wire_internal_impedance": "analytic", "wire_cln_s0_zero": "analytic",
    "cylinder_mesh": "geometry", "ladder_netlist": "spice",
    "reference_impedance": "reference", "reference_sweep": "reference",
    "DiffusionCLN": "diffusion", "rc_ladder_impedance": "diffusion",
    "EnergyCLN": "energy", "physical_cln": "energy",
    "CLNLTI": "export", "rc_lti": "export", "rl_lti": "export",
    "InitialProjection": "initial", "MagneticFieldCLN": "magnetic",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    # Matrix-only energy models need NumPy, not a finite-element installation.
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
