# CLN-mcp

Reproducible Cauer ladder network derivations and finite-element validation notebooks.

| Notebook | Verified scope |
|---|---|
| [Type2: A and T give the same circuit](docs/type2_same_circuit.ipynb) | Circular wire, positive real shift, R0 / L1 / R2 / L3; symbolic boundary and normalization proof; energy vs Cauer |
| [Type1: derivation and NGSolve reproduction](docs/type1_same_circuit.ipynb) | Circular wire, positive real shift, L-1 / R0 / L1 / R2; symbolic proof, A/T finite elements, mesh convergence and radia-ngsolve runtime reproduction |

## Reproduce

Install `ngsolve`, `numpy`, `mpmath`, `nbformat`, `nbclient` and `ipykernel` for Python; symbolic checks require `wolframscript` on PATH (or `WOLFRAMSCRIPT`). Open the executed notebooks under `docs/` and rerun their cells.

Standalone commands:

```text
wolframscript -file mathematica/derive_type1_same_circuit.wls type1_results.json
python validation/type1_ngsolve.py --output type1_fe.json
```

- `docs/*.ipynb`: primary derivations and executed reproduction entry points.
- `mathematica/*.wls`: symbolic and high-precision checks.
- `validation/`: public finite-element reproduction code.
- `docs/data/`: recorded second-runtime numerical evidence.

The public repository currently contains this focused documentation release. The broader MCP application is not yet included. These four-element continuum results do not prove arbitrary stage counts or general 3D mixed formulations. A finite shifted ladder matches locally near its expansion point, not at all frequencies. Positive elements in a shifted representation alone do not prove passivity in physical s.

Type2's underlying derivation received an independent Claude Opus 5.5 review. Type1 was verified by Codex using Wolfram, independent Bessel evaluation and two FE formulations; no new Opus review is claimed.

BSD-3-Clause license.
