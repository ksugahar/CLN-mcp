# CLN-mcp

**A Cauer Ladder Network (CLN) represents electromagnetic fields with a small resistor–inductor circuit.** Its stages correspond to field modes; losses and magnetic-energy integrals determine circuit elements.

Start with [What is CLN?](docs/00_what_is_cln.ipynb), then follow the [notebook reading guide](docs/README.md). The primary public material is the executed `docs/*.ipynb` notebooks.

## Derivations and learning notebooks

| Notebook | Content and verified scope |
|---|---|
| [What is CLN?](docs/00_what_is_cln.ipynb) | Field/circuit diagrams, skin effect, normalization, energy, truncation |
| [Expansion points and energy](docs/01_expansion_and_energy.ipynb) | Positive real shifts, Type1/Type2, passivity caveats, stored FEM convergence |
| [Type2: the same circuit from A and T](docs/type2_same_circuit.ipynb) | Circular wire, positive real shift, R0/L1/R2/L3, symbolic field/boundary correspondence |
| [Type1 derivation and NGSolve reproduction](docs/type1_same_circuit.ipynb) | Circular wire, L-1/R0/L1/R2, separate A/T FEM, mesh convergence |
| [Formulations, gauges, and boundaries](docs/02_formulations_gauge_boundary.ipynb) | A–φ/T–Ω/A–T, gauge identities, physical-field reconstruction |
| [Nonlinear FP-CLN and surface modes](docs/03_nonlinear_fp_surface_modes.ipynb) | Fixed linear kernel, nonlinear correction sources, exterior surface reduction |
| [Termination, conversion, and tiles](docs/04_termination_conversion_tiles.ipynb) | Termination inversion, circuit definitions, exact versus approximate conversion |
| [Multiport models](docs/05_multiport.ipynb) | Coupling, power-preserving coordinates, reciprocity and dissipation |
| [Foster–Cauer conversion and error](docs/09_foster_cauer_conversion_error.ipynb) | Cited synthesis/error literature, exact round trip, truncation and error budgets |

## Actual finite-element experiments

[3D circular wire](docs/06_3d_round_wire.ipynb) reproduces A–φ/T–Ω/A–T from student notebooks. It tests conductor-interior R0/L1/R2 at s₀=0 on two meshes, including terminal normalization, physical fields, load projection, and gauge sensitivity.

[Surface enrichment and Foster comparisons](docs/07_surface_hybrid_foster.ipynb) separates Foster training-band allocation from DC constraints and tests nonlinear 2D enrichment. High-band fitting is effective with more offline full-model information here; freeing DC adds little. Candidate counts and independent ranks, linkage, and conductor Joule loss are reported separately.

[Common-excitation CLN, boundary response, and POD](docs/08_3d_surface_pod_foster.ipynb) compares single/multipoint CLN, CLN2 plus boundary-response enrichment, and POD at matched state counts. **CLN and POD use the same physical excitation.** They differ in shifts, frequency information, and basis construction; offline costs are not matched. Boundary-response POD is not exterior Steklov modes. Foster terminal equivalence is not the derivation of a physical mixed Cauer ladder.

The notched-3D full responses differ by about 3.46% between meshes. Small reduction error against one discrete model is not established continuum/device accuracy. No universal Foster/CLN ranking is claimed.

[Research follow-up](docs/10_research_followup.ipynb) verifies three-stage round-wire elements against independent Bessel-series coefficients, compares matched-count reaction-transfer fits on the POD training grid, and tests nonlinear bases against both matched-rank models and the preserved two-mode bulk control. It also separates first- and second-order full-model mesh changes from reduction errors. These added results have explicit discretization and information-budget limits.

[TEAM 28 force](docs/11_team28_force.ipynb) derives a six-state, twelve-element Type1 CLN from magnetic-energy and Joule-loss norms using the actual winding excitation. It compares reconstructed lift with newly computed full FEM at 25 heights, checks a 1 mN reduction gate and a 0.6 mm equilibrium gate against the published 11.3 mm, and records fresh local root refinement and mesh/outer-air sensitivity separately. The scope is axisymmetric Model A at fixed height and 50 Hz. See [physical field recursion and implementation checks](docs/CLN_FIELD_RECURSION.md).

[General 3D terminal CLN](docs/12_general_3d_cln.ipynb) constructs energy-normalized Type2 field modes in a non-symmetric notched conductor with a non-trivial scalar potential. It compares two- through eight-element ladders at zero and positive real expansion points, full A–phi/A–T/T–Omega terminal responses, and h/p refinement. The T forms use a free scalar terminal-current mode and side-only tangential constraints. Shared DC lifts and the shared zero-shift R2 of the two T forms are disclosed. Deep R4-L7 elements are not cross-validated. Both 100 kHz and 1 MHz fail the skin-depth resolution screen on every study mesh and support only same-mesh reduction comparisons; passing the screen does not establish FE accuracy. The interior magnetic boundary excludes air and return-path inductance. Independent Claude Opus 5.5 numerical review gave scientific acceptance at `c18840a` on 7 October 2026; see the reviewed scope and limits in [REVIEW_STATUS](docs/REVIEW_STATUS.md).

## Reproduce

Install `ngsolve`, `numpy`, `scipy`, `matplotlib`, `mpmath`, `sympy`, `nbformat`, `nbclient`, and `ipykernel` for Python. Symbolic derivation cells require `wolframscript` on PATH (or `WOLFRAMSCRIPT`). Stored notebook outputs are readable on GitHub. The 3D circular-wire scripts additionally require a Radia build that provides `radia.sparsesolv_ngsolve` with NGSolve interoperability enabled; see [Radia](https://github.com/ksugahar/Radia). Installing NGSolve alone does not provide this solver. The current reciprocal-review reruns use Radia 5.2.3; singular-system breakdown behavior differs across Radia versions, so record the installed version when reproducing.

```text
wolframscript -file mathematica/derive_type1_same_circuit.wls type1_results.json
python validation/type1_ngsolve.py --output type1_fe.json
python validation/validate_3d.py --output three_dimensional.json
python tools/policy_lint.py
python validation/validate_foster_cauer.py
python validation/surface_hybrid/validate_evidence.py
python validation/surface_hybrid/validate_same_excitation_3d.py
wolframscript -file mathematica/derive_energy_recursion.wls docs/data/energy_recursion.json
python validation/team28_cln.py --output docs/data/team28_cln.json
python validation/team28_refine_equilibrium.py --output docs/data/team28_equilibrium.json
python validation/validate_team28.py
python validation/cln3d/general_shape.py --maxh .006 .004 .003 .002 --modes 4 --with-t --output docs/data/general_3d_cln_p1.json --export docs/data/general_3d_cln_matrices.json
python validation/cln3d/general_shape.py --order 2 --maxh .004 .003 --modes 4 --with-t --output docs/data/general_3d_cln_p2.json
python validation/validate_general_3d.py
```

- `docs/*.ipynb`: derivations and executed examples.
- `mathematica/*.wls`: symbolic and high-precision checks.
- `validation/`: public reproduction code.
- `docs/data/`: recorded numerical evidence and source identifiers.

This release focuses on documentation and reproduction; the broader MCP application is not distributed here. Four-element continuum results do not prove arbitrary stages or general 3D mixed formulations. A finite shifted ladder is a local approximation, not an exact response at all frequencies. Positive shifted elements alone do not establish passivity in physical s.

Type2's underlying derivation received an independent Claude Opus 5.5 review. Type1 was checked by Codex using Wolfram, independent Bessel evaluation, and separate FE formulations. The later 3D, surface/POD, and conversion additions received reciprocal Codex and Claude Opus 5.5 review. TEAM 28 received two independent Claude Opus 5.5 review rounds, including native force/root and exact-proof reproduction. See [reviewed revisions, corrections and limits](docs/REVIEW_STATUS.md). Stored JSON `review` fields record producer-time status; the linked review note gives the current verdict.

BSD-3-Clause license.
