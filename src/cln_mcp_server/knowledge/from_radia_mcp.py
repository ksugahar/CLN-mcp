"""CLN / Cauer material absorbed from radia-mcp (removed there in radia-mcp 2.0.0).

Removed passages, grouped by their former radia-mcp module (source:
ksugahar/Radia a96c07b26).  References to unpublished manuscripts and to
non-public files were edited out for the public package.
"""

TOPICS = {
    "bayesian_opt_knowledge": ("bayesian_opt/knowledge.py", """  surrogate" alternative (PRIMA, Cauer ladder)
"""),
    "bem_sommerfeld_layered_knowledge": ("bem/sommerfeld_layered_knowledge.py", """Sugahara lab CLN papers, and most engineering EM literature use
"""),
    "fem_nonconforming_mesh_coupling_knowledge": ("fem/nonconforming_mesh_coupling_knowledge.py", """## 6. BEM-CLN multi-conductor refinement

Status: lab uses single-mesh BEM-FEM coupling
(radia_mcp.bem.bem_fem_bem_hybrid).

If multi-mesh BEM coupling is needed (e.g., each transformer winding
with its own surface mesh), mortar on BEM panels is possible
(De Gersem-Weiland series).
"""),
    "fem_xfem_em_hiruma_knowledge": ("fem/xfem_em_hiruma_knowledge.py", """Topic: asymptote-preserving Schur augmentation of CLN (see the
sibc_orthogonal area, xfem_vs_sibc section).

The XFEM reproduction also independently confirms:
"""),
    "figure_server": ("figure/server.py", """           {"y_data":   8.5, "text": "CLN-N",  "color": "C0"},
"""),
    "figure_tools": ("figure/tools.py", """        ...     {"y_data": 141, "text": "CLN (R-term)", "color": "C1"},

[...]

        ...     {"y_data": 0.55, "text": "CLN (L-term)", "color": "C0"},

[...]

# ROM-paper helpers (Cauer / Multi-K / Schur-F asymptote plots)

[...]

    rational ROMs (CLN-N, Multi-K) decay as r ~ omega^{-(N-1/2)} above
"""),
    "gnn_knowledge": ("gnn/knowledge.py", """| Circuit reduction | PRIMA / Cauer (`radia_mcp.mor`) |
"""),
    "magnetic_materials_iron_loss_knowledge": ("magnetic_materials/iron_loss_knowledge.py", """- Cauer 等価回路とプレイモデルを用いた...鉄損特性評価

[...]

- Cauer等価回路とプレイモデルを用いた数値解析による
  インバータ励磁下における鉄損特性評価 (PWM-specific)
"""),
    "mathematica_basis_functions_eddy_parent_order_admissibility": ("mathematica/basis_functions/eddy_parent_order_admissibility.wls", """  EVRS/CLN rank plus topology and surface branches:

[...]

  chk["parent p requirement is max(bulk, bridge, surface), not CLN rank",

[...]

  (* Symbolic statement: CLN/Krylov depth nEVRS is independent of p unless the
     spatial moment degree requested by that CLN construction changes. *)
"""),
    "mathematica_basis_functions_eddy_topology_reduction": ("mathematica/basis_functions/eddy_topology_reduction.wls", """      interior HCurl bulk       -> EVRS / CLN response class.

[...]

     numerical p=6 slogan.  The actual requirements come from the intended CLN
"""),
    "mathematica_basis_functions_evrs_derham_bridge": ("mathematica/basis_functions/evrs_derham_bridge.wls", """     matrices standing in for resistance, inductance, BEM/VIM, or CLN moments.
"""),
    "mathematica_basis_functions_readme": ("mathematica/basis_functions/README.md", """(The CLN/Foster work used **arbitrary-precision adaptive integration**,

[...]

| `eddy_parent_order_admissibility.wls` | **parent HCurl order ledger**: `p` is selected by spatial requirements, not by the CLN/Krylov rank; `p >= max(r_bulk, r_bridge, r_surface)`, surface-Omega scalar degree is `r_surface+1`, and enriched bridge traces have `cycle_rank * dim P_r(face)` modes | done, self-test PASS (`dim P_r(face)=Binomial[r+2,2]`, constant bridge trace gives one mode per cycle, quadratic trace multiplies by 6, p=6 can be admissible/conservative without being optimal, high-order bridge trace can force p=6 symbolically, current coarse count `24+7+3=34`) |
"""),
    "meta_catalog": ("meta/catalog.py", """                       "dynamic hysteresis MEC (Play + Cauer)",
"""),
    "pcb_applications_knowledge": ("pcb/applications_knowledge.py", """   `radia_mcp.mor.systematic` (CLN, Lanczos, PRIMA)
"""),
    "radia_ngsolve_knowledge_axifem": ("radia_ngsolve/knowledge/axifem.py", """Reference (published background):
  Nagamine et al., "Convergence analysis of the Cauer ladder network
  method using eigenfunction expansion", IEEE CEFC 2022.

Check: three-way Cauer cross-validation (axisymmetric FEM, BEM-Foster,
analytic) in the Kameari convention (see the 3d area).
"""),
    "radia_ngsolve_knowledge_kelvin": ("radia_ngsolve/knowledge/kelvin.py", """  from the 2026-05 CLN study, so corners are a known risk area, not a

[...]

- The goldens validate the FEM formulation layer.  CLN / Cauer-ladder
"""),
    "radia_ngsolve_knowledge_technical_reports": ("radia_ngsolve/knowledge/technical_reports.py", """coefficient, carry its uncertainty, then re-evaluate the loss in the Cauer or

[...]

## Play plus Cauer

The report combines a play model for DC hysteresis with a Cauer ladder for
skin-effect and dynamic minor-loop behavior.  One ladder stage can overestimate
PWM minor-loop behavior; two stages improve the reported case.  A useful
workflow is to identify static hysteresis first, identify the dynamic ladder
from frequency-dependent loss, and validate an unseen waveform rather than
only the calibration loop.
"""),
    "rna_mec_knowledge": ("rna_mec/knowledge.py", """  - Hane, Nakamura, "Dynamic Hysteresis Modeling for Magnetic Circuit
    Analysis by Incorporating Play Model and Cauer's Equivalent Circuit
    Theory", IEEE Trans. Magn., DOI 10.1109/TMAG.2020.3004355, 2020.

[...]

  - Kameari, Ebrahimi, Sugahara, Shindo, Matsuo, "Cauer Ladder Network
    Representation of Eddy-Current Fields for Model Order Reduction
    Using Finite-Element Method", IEEE Trans. Magn. 54(3), 7201804,
    2018.

[...]

    "dynamic_hysteresis": "Play + Cauer dynamic hysteresis MEC (lab specialty)",

[...]

  - Hane 2019 -- Japanese technical-meeting version of Play + Cauer

[...]

  - Hane, Nakamura, IEEE Trans. Magn. 2020 -- the English IEEE version
    of the Play + Cauer MEC.

[...]

integral leakage coupling, and the Play + Cauer dynamic hysteresis MEC.

[...]

This is how lab-scale Hane MEC + Cauer + Play models are typically

[...]

DYNAMIC_HYSTERESIS = r\"\"\"
# Dynamic hysteresis in MEC -- Play + Cauer (Sugahara-Nakamura-Hane lineage)

Static-BH MEC (Hopkinson with mu(B)) is fine for slowly-varying
fields, but FAILS at PWM-driven hysteresis loops with minor-loop
patches at carrier frequencies of 1-10 kHz.  The eddy-current
distribution inside the lamination is not uniform at those
frequencies (skin depth < lamination thickness), and the apparent
permeability becomes complex and frequency dependent.

The Sugahara-Nakamura-Hane lineage solves this with a dynamic
hysteresis MEC that combines:
  - PLAY MODEL for the DC hysteresis (Bobbio-Visone 1997, extended to
    Tohoku-Kindai shape-function decomposition).
  - CAUER LADDER NETWORK (Cauer-I) for the eddy-current skin effect
    inside the lamination.
  - Separate dependent source for ANOMALOUS eddy-current loss
    (Bertotti excess loss).

Key references:

  Hane, Nakamura, "Dynamic Hysteresis Modeling for Magnetic Circuit
  Analysis by Incorporating Play Model and Cauer's Equivalent Circuit
  Theory", IEEE Trans. Magn., DOI 10.1109/TMAG.2020.3004355, 2020.

  Tanaka, Nakamura, Ichinokura, "Magnetic Circuit Model combined with
  Play Model Obtained from Landau-Lifshitz-Gilbert Equation", JPCS
  903, 012047, 2017.

  Shindo, Noro, "Simple Circuit Simulation Models for the Eddy Current
  in Magnetic Sheets and Wires", IEEJ Trans. FM 134(4), 173-181,
  2014.

  Shindo, Miyazaki, Matsuo, "Cauer Circuit Representation of the
  Homogenized Eddy-Current Field Based on the Legendre Expansion for
  a Magnetic Sheet", IEEE Trans. Magn. 52(3), 6300504, 2016.

## The MEC per-branch structure

Each branch in the RNA is replaced by:

    Branch i = [ Play hysterons ] in series with [ Cauer ladder ]

The PLAY HYSTERON block (Hane 2020 Fig. 1) takes the branch flux Phi
as input and produces the MMF F as output via:

    H = sum_k zeta_k(eta_k, B)                                       (1)

where eta_k are play widths (Tesla), zeta_k are play hysterons, and
the sum is weighted by shape functions f_k(r) derived from a family
of MEASURED DC hysteresis loops at different B_max amplitudes
(Hane Fig. 2).

The CAUER LADDER block (Hane 2020 Fig. 6, 7) is a Cauer-I network
truncated to 2 or 3 stages with parameters (Hane eq. 2-3):

    L_2n+1 = mu * geometric_factor                                   (2)
    R_2n   = 4 / (sigma d^2)                                         (3)

where d is lamination thickness, sigma conductivity, and mu the
LINEARIZED differential permeability evaluated on the centerline of
the DC hysteresis loop at B_m = 1 T.  This linearization is a
practical recipe Hane validates empirically on grain- and non-
oriented Si-Fe.

The CLASSICAL eddy-current loss equals 1/L_1 in the classical limit
(first stage of ladder = single inductance).  Adding more stages
captures the skin-effect-driven reduction at high frequency.

The ANOMALOUS eddy-current loss is represented as a separate
dependent source proportional to (dB/dt)^1.5 with coefficient gamma_2
calibrated from low-frequency core-loss measurements (Hane Fig. 3 and
Table I).

## Time stepping

At each time step t:

  1. Read winding currents from external SPICE circuit -> MMF sources
     on the source branches.
  2. Solve nonlinear KCL at all nodes (modified-Newton with current
     branch tangents).
  3. For each branch:
     - Update the Play state (which hysterons are pushed)
     - Update the Cauer state variables (currents in each ladder
       inductor)
     - Sum the resulting H, the dB/dt contribution, and the anomalous
       term.
  4. Compute total flux per branch + per-branch iron-loss breakdown:
       P_total = P_hyst + P_classical + P_anomalous

The Play state update is O(K) per branch (K = number of hysterons,
typically 5-20).  The Cauer state update is O(N_stage) per branch
(N_stage = 2-5).  Both are constant-cost-per-step, no matrix solve
needed for the hysteresis-loss part itself.

## Iron-loss decomposition (Hane Fig. 9)

The output of the model gives:

  - P_hyst (Joule loss in static hysteresis loop) -- frequency
    independent.
  - P_classical (Joule loss in distributed eddy currents) -- scales
    as f^2 below the skin-depth corner, falls off above.
  - P_anomalous (Joule loss in domain-wall fluctuations) -- scales
    as f^1.5.

This decomposition is the SUGAHARA-NAKAMURA-HANE selling point: not
just total loss but a physically-interpretable breakdown that
guides material selection (high-Si steel reduces P_classical;
domain-refined steel reduces P_anomalous).

## Complex permeability output (Hane Fig. 10)

By exciting the model with a small sinusoidal H at frequency f and
fitting the response, one extracts:

    mu*(f) = mu'(f) - j mu"(f)                                       (4)

The model reproduces measured mu'(f) and mu"(f) across 10 Hz - 10 kHz
on grain- and non-oriented Si-Fe.  This validates the model not just
for total loss but for the full frequency-domain magnetic response,
which is what a power-electronics circuit simulator actually sees.

## Forward extension: full motor RNA

Hane's 2019 technical-meeting paper applies the per-branch Play + Cauer
structure to a multi-branch RNA of a permanent-magnet motor / three-
phase variable inductor.  Each branch's hysteresis and eddy-current
state evolves independently; the nonlinear KCL couples them through
the flux-conservation constraint.

This is THE most physically detailed RNA available for PWM iron-loss
prediction in rotating machines, and it is the production model of
the Tohoku-Kindai collaboration.

## Cross-references

- `radia_mcp.magnetic_materials.hysteresis_models.play` -- Play model
  forward / inverse evaluation API.
- `radia_mcp.magnetic_materials.hysteresis_models.lab_core` -- the
  Tohoku-Kindai energy / play core lineage.
- Hane, Nakamura, IEEE Trans. Magn. 2020, DOI 10.1109/TMAG.2020.3004355
  -- English IEEE version.
\"\"\"

[...]

armature, iron loss with optional Play / Cauer extensions.

[...]

- 1-D laminated sheet eddy current -> **Cauer ladder** (Hane 2020).

[...]

- PWM iron-loss prediction in laminated machines -> **dynamic
  hysteresis MEC** = RNA + Play + Cauer (Hane / Sugahara-Nakamura
  lineage).

[...]

PEEC + Cauer + Play in a unified framework; for any specific problem

[...]

        dynamic_hysteresis

[...]

        "dynamic": "dynamic_hysteresis",
        "play_cauer": "dynamic_hysteresis",
        "hysteresis_mec": "dynamic_hysteresis",
        "hane": "dynamic_hysteresis",

[...]

        "dynamic_hysteresis": DYNAMIC_HYSTERESIS,

[...]

            "dynamic_hysteresis",
"""),
    "rna_mec_server": ("rna_mec/server.py", """            "dynamic_hysteresis"             - Lab Play + Cauer dynamic MEC

[...]

    description='RNA / Magnetic Equivalent Circuit. ★ Lab specialty: dynamic hysteresis MEC (Play + Cauer)',
"""),
}


def get_from_radia_mcp(topic: str = "index") -> str:
    """Return one absorbed passage, or the index of available topics."""
    if topic == "index":
        return "\n".join(f"{k}: former radia_mcp/{v[0]}" for k, v in TOPICS.items())
    if topic not in TOPICS:
        raise ValueError(f"Unknown topic '{topic}'. Available: " + ", ".join(TOPICS))
    path, text = TOPICS[topic]
    return f"# from former radia_mcp/{path}\n\n{text}"
