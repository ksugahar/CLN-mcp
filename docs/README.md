# Notebook reading guide

| Order | Notebook | What it explains | Rerun requirements |
|---|---|---|---|
| 1 | [What is CLN?](00_what_is_cln.ipynb) | Skin effect, fields/circuits, energy integrals, stages and errors | numpy, matplotlib, mpmath |
| 2 | [Expansion points and energy](01_expansion_and_energy.ipynb) | Types, shifts, normalization, passivity, stored FEM results | Same as above |
| 3 | [Formulations, gauges, and boundaries](02_formulations_gauge_boundary.ipynb) | A–φ/T–Ω/A–T, reconstruction, gauge identities | sympy |
| 4 | [Nonlinear FP-CLN and surface modes](03_nonlinear_fp_surface_modes.ipynb) | Fixed-point correction, surface reduction, convergence versus model error | numpy, matplotlib |
| 5 | [Termination, conversion, and tiles](04_termination_conversion_tiles.ipynb) | Termination inversion, exact/approximate conversion, circuit tiles | sympy, matplotlib |
| 6 | [Multiport models](05_multiport.ipynb) | Coupling, port work, reciprocity, dissipation | numpy |
| 7 | [Type2 derivation](type2_same_circuit.ipynb) | A/T field and boundary correspondence, same four elements | Python, Wolfram |
| 8 | [Type1 derivation and FEM](type1_same_circuit.ipynb) | Same four elements, separate A/T finite elements | Python, Wolfram, NGSolve |
| 9 | [3D circular wire](06_3d_round_wire.ipynb) | Initial A–φ/T–Ω/A–T stages, field/element convergence, gauge sensitivity | Plots: numpy/matplotlib; FEM: NGSolve/scipy/Radia |
| 10 | [Surface enrichment and Foster](07_surface_hybrid_foster.ipynb) | Band/DC factors, 3D reduction, nonlinear 2D enrichment, ranks and Joule loss | Plots: numpy/matplotlib; solves: NGSolve/scipy/mpmath |
| 11 | [Common-excitation CLN and POD](08_3d_surface_pod_foster.ipynb) | Single/multipoint CLN, boundary-response POD, Steklov distinction, Foster equivalence | Plots: numpy/matplotlib; solves: NGSolve/scipy |
| 12 | [Foster–Cauer conversion and error](09_foster_cauer_conversion_error.ipynb) | Exact round trip, termination, shifted representations, error measures, cited papers | sympy, numpy, matplotlib |

All notebooks contain executed outputs for reading on GitHub. The two introductory notebooks use analytic solutions and need neither FEM nor Wolfram.

This public release centers on derivations and reproduction, not distribution of the full MCP application. Analytic continued-fraction examples, field-derived circuit proofs, and finite-element experiments have explicitly different scopes.

The formulation, nonlinear introduction, termination, and multiport notebooks use minimal explanatory examples. They do not implement every general 3D/nonlinear/matrix-CLN extension. The 3D circular-wire reproduction is restricted to linear material, s₀=0, conductor interior, and R0/L1/R2.

The comparison experiments use a linear internal-conductor reaction transfer in 3D and a nonlinear 2D full discrete/time-discrete reference. They do not establish general 3D nonlinear performance or universal Foster superiority. Exact terminal conversion does not reconstruct missing physical field modes.

These additions received reciprocal Codex and Claude Opus 5.5 review. See [reviewed revisions, corrections and limits](REVIEW_STATUS.md) and each notebook for its evidence.

[Research follow-up](10_research_followup.ipynb): three-stage refinement, same-grid Foster fits, and matched-rank nonlinear held-out amplitudes.

Notebook 10 also records the original two-mode bulk control and second-order full-model mesh refinement at four representative frequencies. These tests distinguish adding modes from reallocating a fixed rank, and intermesh agreement from certified continuum accuracy.

[TEAM 28 force](11_team28_force.ipynb): actual coil-driven Type1 energy recursion, physical L/R elements, conjugate phasor lift, 25-height same-mesh full-FEM comparison, static equilibrium and settings probes. Reading needs numpy/matplotlib/sympy; native recomputation needs NGSolve/scipy. [Implementation knowledge](CLN_FIELD_RECURSION.md) explains energy normalization, saved-field storage, boundary design and diagnostic limits.

[General 3D terminal CLN](12_general_3d_cln.ipynb): a notched conductor with non-trivial phi, physical eight-element Type2 recursions at zero/positive real shifts, equivalent current-port T formulations, same-mesh full references, contact order and h/p comparisons. Shared DC port lifts and unresolved skin layers are explicit limitations. Reading needs numpy/matplotlib; native recomputation needs NGSolve/scipy. Independent numerical review gave scientific acceptance at c18840a; see REVIEW_STATUS for the precise scope.

[General 3D CLN with surrounding air](13_general_3d_air.ipynb): a source-break terminal port inside a perfectly conducting enclosure, global gradient gauge with zero air conductivity, total magnetic-energy normalization, coaxial oracle, conductor/air refinement and return-current diagnostics. A–T shares the magnetic operator and DC lift. Numerical reciprocal review is pending; the notebook distinguishes reduction accuracy from mesh and geometry error. Reading needs numpy/matplotlib/mpmath; native recomputation needs NGSolve/scipy and the symbolic proof needs Wolfram.
