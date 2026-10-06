"""Repository policy lint for CLN-mcp.

Public documentation release policy:
  P1  Implementations are published only as Python (.py), Mathematica (.wls / .wl) or executed
      Jupyter notebooks (.ipynb, under docs/ only).
  P2  No files of commercial or other-language tool chains (MATLAB, COMSOL, FEMM, JMAG, FreeFEM, Julia).
  P3  No commercial tool names (COMSOL, FEMM, JMAG) in published text or code.
  P4  No absolute local paths (drive letters, user directories) in published files.
  P5  Notebooks in docs/ are executed (every code cell has an execution count).
  P6  Data files are JSON (preferred) or HDF5 (.h5/.hdf5) for large arrays; no .mat/.dat/.npy/.npz/.pkl.
  P7  Nothing refers to the lab archive: no drive-letter tokens such as "W:" in code, no <archive> placeholder.
  P8  No private solver code: no EMPY names or imports; no binary executables or libraries (.pyd/.dll/.exe/.so).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "build", "dist", ".venv"}
CODE_EXT = {".py", ".wls", ".wl", ".ipynb"}
FORBIDDEN_EXT = {".m", ".mlx", ".slx", ".mat", ".mex", ".mexw64", ".mph", ".fem", ".ans", ".jmdl", ".jproj",
                 ".jfiles", ".jcf", ".edp", ".idp", ".jl"}
DATA_FORBIDDEN = {".dat", ".npy", ".npz", ".pkl", ".pickle"}       # .mat is already in FORBIDDEN_EXT
BINARY_FORBIDDEN = {".pyd", ".dll", ".exe", ".so", ".dylib", ".obj", ".o", ".a", ".lib.bin"}
PRIVATE = re.compile(r"\bEMPY\b|import\s+empy|from\s+empy|EMPY_Solver", re.I)
ARCHIVE = re.compile(r"[\"'][A-Za-z]:[\"']|<archive>|lab archive", re.I)
TEXT_EXT = CODE_EXT | {".md", ".txt", ".json", ".toml", ".cfg", ".yml", ".yaml", ".lib", ".cir", ".svg"}
COMMERCIAL = re.compile(r"\b(COMSOL|FEMM|JMAG)\b", re.I)
ABSPATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\\\/][^\s\"'<>|]+|/(?:Users|home)/[A-Za-z0-9_.-]+|\\\\Users\\\\")
# files whose job is to name the forbidden things (the policy itself and this lint)
SELF = {Path("tools/policy_lint.py"), Path("docs/REPOSITORY_POLICY.md"), Path("tests/test_policy.py"),
        Path("tests/test_knowledge_contracts.py")}   # the last one asserts the names are ABSENT
ALLOWED_PATHS = re.compile(r"[A-Za-z]:[\\\\/]+Program")   # standard install locations (Program Files) are not private


def files(base=ROOT):
    for p in base.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(base).parts):
            yield p


def check(root=None):
    base = ROOT if root is None else Path(root)
    problems = []
    for p in files(base):
        rel = p.relative_to(base)
        ext = p.suffix.lower()
        if ext in FORBIDDEN_EXT:
            problems.append(("P2", str(rel), "forbidden file type"))
            continue
        if ext in BINARY_FORBIDDEN:
            problems.append(("P8", str(rel), "binary executable/library"))
            continue
        if ext in DATA_FORBIDDEN:
            problems.append(("P6", str(rel), "data must be JSON or HDF5"))
            continue
        if ext == ".ipynb" and rel.parts[0] != "docs":
            problems.append(("P1", str(rel), "notebooks are published under docs/ only"))
        if rel in SELF or ext not in TEXT_EXT:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in COMMERCIAL.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(("P3", f"{rel}:{line}", m.group(0)))
        for m in ABSPATH.finditer(text):
            if ALLOWED_PATHS.match(m.group(0)):
                continue
            line = text.count("\n", 0, m.start()) + 1
            problems.append(("P4", f"{rel}:{line}", m.group(0)[:60]))
        for m in PRIVATE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(("P8", f"{rel}:{line}", m.group(0)))
        for m in ARCHIVE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(("P7", f"{rel}:{line}", m.group(0)))
        if ext == ".ipynb" and rel.parts[0] == "docs":
            nb = json.loads(text)
            for i, c in enumerate(nb.get("cells", [])):
                if c.get("cell_type") == "code" and "".join(c.get("source", [])).strip() and c.get("execution_count") is None:
                    problems.append(("P5", f"{rel} cell {i}", "not executed"))
    return problems


if __name__ == "__main__":
    probs = check(sys.argv[1] if len(sys.argv) > 1 else None)
    for rule, where, what in probs:
        print(f"{rule}  {where}  {what}")
    print(f"{len(probs)} problem(s)")
    sys.exit(1 if probs else 0)
