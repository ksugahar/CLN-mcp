# CLN-mcp

**CLNは、電磁場を少数の抵抗・インダクタンスからなる回路で表現する方法です。** 回路の各段を場のモードと結びつけ、損失や磁気エネルギーの積分から素子値を導きます。

初めて読む方は **[CLNとは何か](docs/00_what_is_cln.ipynb)** から。表皮効果、はしご回路、エネルギー、段数と精度を図と実行例で説明しています。[全ノートの読み順](docs/README.md)も用意しました。

Reproducible Cauer ladder network derivations and finite-element validation notebooks.

| Notebook | Content / verified scope |
|---|---|
| [CLNとは何か](docs/00_what_is_cln.ipynb) | Introductory field/circuit diagrams, skin effect, energy integrals and truncation error |
| [展開点とエネルギー](docs/01_expansion_and_energy.ipynb) | Expansion point, Type1/Type2, normalization and recorded FEM convergence |
| [Type2: A and T give the same circuit](docs/type2_same_circuit.ipynb) | Circular wire, positive real shift, R0 / L1 / R2 / L3; symbolic boundary and normalization proof; energy vs Cauer |
| [Type1: derivation and NGSolve reproduction](docs/type1_same_circuit.ipynb) | Circular wire, positive real shift, L-1 / R0 / L1 / R2; symbolic proof, A/T finite elements, mesh convergence and radia-ngsolve runtime reproduction |

## Reproduce

Install `ngsolve`, `numpy`, `matplotlib`, `mpmath`, `nbformat`, `nbclient` and `ipykernel` for Python; symbolic checks require `wolframscript` on PATH (or `WOLFRAMSCRIPT`). Open the executed notebooks under `docs/` and rerun their cells.

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
