"""CLN scripts for a round conductor (original notebooks by N. Tanimoto).

Standalone, runnable Python scripts that compute the Cauer ladder network
parameters R_0, L_1, R_2, ... of a round conductor with NGSolve and compare
them with the analytic round-wire values.  Linear systems use the public
sparsesolv ICCG shipped with Radia (helper module ``_iccg.py``) or NGSolve's
built-in CG; every solve is checked and non-convergence raises.  Each script
takes command-line options (mesh size, order, stages, output path, --quick)
and writes its results as JSON.

Available scripts:
  - CLN_2D.py            : 2D cross-section reference, scalar A_z formulation
  - A_ICCG_production.py : 3D, A formulation with H1 gauge correction, ICCG

Pointers (superseded here; maintained in the public CLN-mcp repository as
validation/cln3d/*.py with docs/06_3d_round_wire.ipynb):
  - CLN_AT.py            : 3D, A-T formulation
  - CLN_T_Omega.py       : 3D, T-Omega formulation
  - CLN_APhi.py          : 3D, A-phi formulation
"""

from pathlib import Path

_NOTEBOOK_DIR = Path(__file__).parent

NOTEBOOK_FILES = {
    "AT": "CLN_AT.py",
    "T_Omega": "CLN_T_Omega.py",
    "APhi": "CLN_APhi.py",
    "2D": "CLN_2D.py",
    "production": "A_ICCG_production.py",
}


def get_notebook(name: str) -> str:
    """Retrieve a Tanimoto CLN notebook code as text.

    Args:
        name: One of "AT", "T_Omega", "APhi", "2D", "production".

    Returns:
        Python script content as string. Unsupported or missing scripts raise.
    """
    if name not in NOTEBOOK_FILES:
        raise ValueError(f"Unknown notebook '{name}'. Available: " + ", ".join(NOTEBOOK_FILES))
    path = _NOTEBOOK_DIR / NOTEBOOK_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Notebook script not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def list_notebooks() -> str:
    """List all available Tanimoto CLN notebooks with sizes."""
    lines = ["# Tanimoto CLN scripts (AT, T_Omega, APhi are pointers to the public "
             "validation/cln3d implementation)\n"]
    for key, fname in NOTEBOOK_FILES.items():
        path = _NOTEBOOK_DIR / fname
        if path.exists():
            size = path.stat().st_size
            lines.append(f"- **{key}** ({fname}): {size:,} bytes")
        else:
            lines.append(f"- **{key}** ({fname}): MISSING")
    return "\n".join(lines)
